//! Loopback JSON-lines control server (`--control <port>`): one request per line,
//! `{"id", "method", "params"}`, one reply per line. The first request must be
//! `{"method":"auth","params":{"token":"…"}}` with the token from `TEXTURECRAFT_CONTROL_TOKEN`
//! or, when that is unset, the token written (owner-only) to
//! `<temp>/texturecraft-control-<port>.token`. Token handling, request framing and connection
//! limits are shared with `texturecraft-cli serve` (`texturecraft_automation::security`).

use std::io::{BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use serde_json::{Value, json};
use texturecraft_automation::security::{
    AUTH_TIMEOUT, AuthDeadline, ConnectionLimiter, LineRead, MAX_CONNECTIONS, read_bounded_line, server_token, token_matches,
};
use texturecraft_ui_egui::ControlRequest;

/// How long an authenticated connection may stay idle.
const IDLE_TIMEOUT: Duration = Duration::from_secs(600);

/// Listen on `127.0.0.1:port` and return the requests for the UI thread to answer
/// ([`ControlRequest::reply`]). With `confined` (`--read-root`/`--write-root` given) `command.run` refuses
/// host paths, as `texturecraft-cli serve` does. Bind or token failures are printed and leave the channel empty.
pub fn start(port: u16, ctx: egui::Context, confined: bool) -> Receiver<ControlRequest> {
    let (tx, rx) = channel::<ControlRequest>();
    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("texturecraft: control server failed to bind 127.0.0.1:{port}: {e}");
            return rx;
        }
    };
    // After binding, so an app that cannot listen never replaces another server's token.
    let token = match server_token(port) {
        Ok((token, Some(path))) => {
            eprintln!("texturecraft: control token written to {}", path.display());
            token
        }
        Ok((token, None)) => token,
        Err(e) => {
            eprintln!("texturecraft: control server disabled: {e}");
            return rx;
        }
    };
    eprintln!("texturecraft: control server listening on 127.0.0.1:{port}");
    std::thread::spawn(move || accept(listener, &token, &tx, &ctx, confined));
    rx
}

/// Serve connections from `listener` (at most [`MAX_CONNECTIONS`] at once).
fn accept(listener: TcpListener, token: &str, tx: &Sender<ControlRequest>, ctx: &egui::Context, confined: bool) {
    let limiter = ConnectionLimiter::default();
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let Some(permit) = limiter.try_acquire(MAX_CONNECTIONS) else {
            let _ = writeln!(stream, "{}", json!({"id": null, "ok": false, "error": "too many connections"}));
            continue;
        };
        let (tx, ctx, token) = (tx.clone(), ctx.clone(), token.to_string());
        std::thread::spawn(move || {
            let _permit = permit;
            serve(stream, &token, &tx, &ctx, confined);
        });
    }
}

/// Send `reply` with the request's `id`; false when the client is gone.
fn reply(out: &mut TcpStream, id: Value, mut reply: Value) -> bool {
    if let Some(o) = reply.as_object_mut() {
        o.insert("id".into(), id);
    }
    writeln!(out, "{reply}").is_ok() && out.flush().is_ok()
}

/// One connection: `auth` first (within [`AUTH_TIMEOUT`], one attempt), then requests
/// forwarded to the UI thread one at a time until the client disconnects or idles out.
fn serve(stream: TcpStream, token: &str, tx: &Sender<ControlRequest>, ctx: &egui::Context, confined: bool) {
    let Ok(mut deadline) = AuthDeadline::start(&stream, AUTH_TIMEOUT).map(Some) else { return };
    let _ = stream.set_read_timeout(Some(IDLE_TIMEOUT));
    let Ok(read) = stream.try_clone() else { return };
    let mut reader = BufReader::new(read);
    let mut out = stream;
    let mut line = String::new();
    loop {
        match read_bounded_line(&mut reader, &mut line) {
            Ok(LineRead::Line) => {}
            Ok(LineRead::TooLong) => {
                reply(&mut out, Value::Null, json!({"ok": false, "error": "request too large"}));
                return;
            }
            Ok(LineRead::Eof) | Err(_) => return,
        }
        if line.trim().is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let sent = reply(&mut out, Value::Null, json!({"ok": false, "error": format!("bad JSON: {e}")}));
                // Before `auth` a malformed line is the one refused attempt, as on `serve --port`.
                if !sent || deadline.is_some() {
                    return;
                }
                continue;
            }
        };
        let id = msg.get("id").cloned().unwrap_or(Value::Null);
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("").to_string();
        let params = msg.get("params").cloned().unwrap_or(json!({}));
        if deadline.is_some() {
            let supplied = params.get("token").and_then(Value::as_str).unwrap_or_default();
            if method != "auth" || !token_matches(token, supplied) {
                // One attempt per connection, as on `texturecraft-cli serve`.
                reply(&mut out, id, json!({"ok": false, "error": "authenticate first: {\"method\":\"auth\",\"params\":{\"token\":…}}"}));
                return;
            }
            drop(deadline.take());
            if !reply(&mut out, id, json!({"ok": true})) {
                return;
            }
            continue;
        }
        let command = params.get("id").and_then(Value::as_str).unwrap_or_default();
        if confined
            && method == "command.run"
            && texturecraft_automation::workspace::authorize_command(command, params.get("params").unwrap_or(&Value::Null)).is_err()
        {
            // (the workspace tools its own message names are not on this channel: the file actions are in the window)
            let error = format!(
                "`{command}` with host paths is disabled on this control channel while --read-root/--write-root is set; use the file actions in the application window"
            );
            if !reply(&mut out, id, json!({"ok": false, "error": error})) {
                return;
            }
            continue;
        }
        let (req, rrx) = ControlRequest::new(method, params);
        if tx.send(req).is_err() {
            return;
        }
        ctx.request_repaint();
        let answer = rrx.recv_timeout(Duration::from_secs(120)).unwrap_or_else(|_| json!({"ok": false, "error": "timeout"}));
        if !reply(&mut out, id, answer) {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::BufRead;

    use super::*;

    const TOKEN: &str = "0123456789abcdef0123456789abcdef";

    /// A server on a free port whose app side answers every request with its method name.
    fn server() -> std::net::SocketAddr {
        server_with(false)
    }

    fn server_with(confined: bool) -> std::net::SocketAddr {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = channel::<ControlRequest>();
        std::thread::spawn(move || accept(listener, TOKEN, &tx, &egui::Context::default(), confined));
        std::thread::spawn(move || {
            for r in rx {
                let _ = r.reply.send(json!({"ok": true, "result": r.method}));
            }
        });
        addr
    }

    fn connect(addr: std::net::SocketAddr) -> (TcpStream, BufReader<TcpStream>) {
        let s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
        let r = BufReader::new(s.try_clone().unwrap());
        (s, r)
    }

    fn call(w: &mut TcpStream, r: &mut BufReader<TcpStream>, line: &str) -> Option<Value> {
        writeln!(w, "{line}").ok()?;
        let mut reply = String::new();
        (r.read_line(&mut reply).ok()? > 0).then(|| serde_json::from_str(&reply).unwrap())
    }

    #[test]
    fn requests_need_the_token() {
        let addr = server();
        // Anything before `auth` is refused and the connection closed.
        let (mut w, mut r) = connect(addr);
        let denied = call(&mut w, &mut r, r#"{"id":1,"method":"command.run","params":{"id":"file.open"}}"#).unwrap();
        assert_eq!(denied["ok"], false);
        assert_eq!(denied["id"], 1);
        assert!(call(&mut w, &mut r, r#"{"id":2,"method":"ping"}"#).is_none(), "closed after a refused request");
        // A wrong token is refused and gets no second try.
        let (mut w, mut r) = connect(addr);
        let wrong = format!(r#"{{"id":1,"method":"auth","params":{{"token":"{}x"}}}}"#, &TOKEN[1..]);
        assert_eq!(call(&mut w, &mut r, &wrong).unwrap()["ok"], false);
        let right = format!(r#"{{"id":2,"method":"auth","params":{{"token":"{TOKEN}"}}}}"#);
        assert!(call(&mut w, &mut r, &right).is_none());
        // So is a malformed first line: no further lines before `auth`.
        let (mut w, mut r) = connect(addr);
        assert_eq!(call(&mut w, &mut r, "not json").unwrap()["ok"], false);
        assert!(call(&mut w, &mut r, &right).is_none(), "closed after malformed JSON before auth");
        // The right token unlocks the connection.
        let (mut w, mut r) = connect(addr);
        let ok = call(&mut w, &mut r, &right).unwrap();
        assert_eq!((ok["ok"].clone(), ok["id"].clone()), (json!(true), json!(2)));
        let pong = call(&mut w, &mut r, r#"{"id":3,"method":"ping"}"#).unwrap();
        assert_eq!((pong["result"].clone(), pong["id"].clone()), (json!("ping"), json!(3)));
        assert_eq!(call(&mut w, &mut r, "not json").unwrap()["ok"], false);
        assert_eq!(call(&mut w, &mut r, r#"{"id":4,"method":"ping"}"#).unwrap()["id"], 4, "bad JSON keeps an authenticated connection");
    }

    #[test]
    fn roots_refuse_host_paths_over_the_control_channel() {
        let right = format!(r#"{{"id":1,"method":"auth","params":{{"token":"{TOKEN}"}}}}"#);
        let run = r#"{"id":5,"method":"command.run","params":{"id":"file.open","params":{"path":"/etc/passwd"}}}"#;
        let (mut w, mut r) = connect(server_with(true));
        call(&mut w, &mut r, &right).unwrap();
        let denied = call(&mut w, &mut r, run).unwrap();
        assert_eq!((denied["ok"].clone(), denied["id"].clone()), (json!(false), json!(5)));
        let why = denied["error"].as_str().unwrap_or_default();
        assert!(why.contains("application window") && !why.contains("project_open"), "{why}");
        assert_eq!(call(&mut w, &mut r, r#"{"id":6,"method":"command.run","params":{"id":"layer.add.fill","params":{}}}"#).unwrap()["ok"], true);
        // Without roots the command is forwarded unchanged.
        let (mut w, mut r) = connect(server());
        call(&mut w, &mut r, &right).unwrap();
        assert_eq!(call(&mut w, &mut r, run).unwrap()["ok"], true);
    }

    #[test]
    fn connections_are_limited_and_idle_ones_time_out() {
        let addr = server();
        let idle: Vec<_> = (0..MAX_CONNECTIONS).map(|_| connect(addr)).collect();
        // Let the server take every idle connection before the one over the limit.
        std::thread::sleep(Duration::from_millis(300));
        // Read without writing: the server closes at once, and unread input would reset the
        // connection before its reply arrives.
        let (_, mut r) = connect(addr);
        let mut full = String::new();
        r.read_line(&mut full).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&full).unwrap()["error"], "too many connections");
        // Unauthenticated connections are closed after AUTH_TIMEOUT, which frees their slots.
        for (_, mut r) in idle {
            let mut rest = String::new();
            assert_eq!(r.read_line(&mut rest).unwrap(), 0);
        }
        // A slot frees once its thread has ended, which can trail the client's EOF a little.
        let right = format!(r#"{{"id":1,"method":"auth","params":{{"token":"{TOKEN}"}}}}"#);
        let authed = (0..50).any(|_| {
            let (mut w, mut r) = connect(addr);
            let ok = call(&mut w, &mut r, &right).is_some_and(|v| v["ok"] == true);
            if !ok {
                std::thread::sleep(Duration::from_millis(100));
            }
            ok
        });
        assert!(authed, "slots are freed after the auth deadline");
    }
}
