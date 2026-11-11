use std::process::{Command, Stdio};
use std::time::Duration;

use tokio::io::AsyncReadExt;

use crate::{EngineError, Result};

/// Most stderr retained for a failed cook; additional output is drained without retaining it.
const MAX_STDERR: u64 = 64 << 10;

/// Run a cook with one deadline covering both process exit and stderr completion.
pub(super) fn run(command: Command, timeout: Duration) -> Result<()> {
    // Engine commands may already be inside an async runtime; use an independent worker
    let worker = std::thread::Builder::new()
        .name("material-cook".into())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().map_err(process_error)?;
            runtime.block_on(run_async(command, timeout))
        })
        .map_err(process_error)?;

    worker.join().map_err(|_| EngineError::Other("MaterialCraft cook worker failed".into()))?
}

async fn run_async(command: Command, timeout: Duration) -> Result<()> {
    let program = command.get_program().to_string_lossy().into_owned();
    let mut child =
        tokio::process::Command::from(command).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped()).kill_on_drop(true).spawn().map_err(
            |error| {
                EngineError::Other(format!(
                    "MaterialCraft's command line ({program}) did not start: {error}. Install MaterialCraft next to TextureCraft or set MATERIALCRAFT_CLI"
                ))
            },
        )?;

    let stderr = child.stderr.take();
    let completion = async {
        let drain = async {
            let mut kept = Vec::new();
            if let Some(mut stderr) = stderr {
                (&mut stderr).take(MAX_STDERR).read_to_end(&mut kept).await?;
                tokio::io::copy(&mut stderr, &mut tokio::io::sink()).await?;
            }

            Ok::<_, std::io::Error>(kept)
        };
        tokio::join!(child.wait(), drain)
    };

    let (status, stderr) = match tokio::time::timeout(timeout, completion).await {
        Ok(result) => result,
        Err(_) => {
            // Cancelling completion closes the pipe even when a descendant still holds it
            let _ = child.kill().await;
            return Err(EngineError::Other(format!("MaterialCraft did not finish cooking within {} s", timeout.as_secs_f64())));
        }
    };

    let status = status.map_err(process_error)?;
    let stderr = stderr.map_err(process_error)?;
    if !status.success() {
        return Err(EngineError::Other(format!("MaterialCraft could not cook the material: {}", String::from_utf8_lossy(&stderr).trim())));
    }

    Ok(())
}

fn process_error(error: std::io::Error) -> EngineError {
    EngineError::Other(format!("MaterialCraft's command line: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::Instant;

    fn child_command(mode: &str) -> std::io::Result<Command> {
        let mut command = Command::new(std::env::current_exe()?);
        command.args(["--exact", "link_cmds::cook_process::tests::cook_child", "--nocapture"]).env("TC_TEST_COOK_CHILD", mode);
        Ok(command)
    }

    #[test]
    fn cook_child() {
        match std::env::var("TC_TEST_COOK_CHILD").as_deref() {
            Ok("chatty") => {
                std::io::stderr().write_all(&vec![b'x'; 256 << 10]).unwrap();
                std::process::exit(4);
            }
            Ok("sleep") => std::thread::sleep(Duration::from_secs(2)),
            Ok("inherited") => {
                let _child = child_command("sleep").unwrap().stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::inherit()).spawn().unwrap();
                std::process::exit(4);
            }
            _ => {}
        }
    }

    #[test]
    fn successful_cook_and_spawn_failure_are_reported() {
        assert!(run(child_command("success").unwrap(), Duration::from_secs(5)).is_ok());
        let error = run(Command::new("/nonexistent/texturecraft-test-cli"), Duration::from_secs(5)).unwrap_err();
        assert!(error.to_string().contains("did not start"));
    }

    #[test]
    fn chatty_cook_drains_output_and_retains_only_the_error_budget() {
        let error = run(child_command("chatty").unwrap(), Duration::from_secs(5)).unwrap_err().to_string();

        assert!(error.contains("could not cook"));
        assert_eq!(error.bytes().filter(|byte| *byte == b'x').count(), MAX_STDERR as usize);
    }

    #[test]
    fn deadline_covers_running_children_and_inherited_stderr() {
        for mode in ["sleep", "inherited"] {
            let started = Instant::now();

            let error = run(child_command(mode).unwrap(), Duration::from_millis(300)).unwrap_err();

            assert!(error.to_string().contains("did not finish cooking"), "{mode}: {error}");
            assert!(started.elapsed() < Duration::from_millis(1500), "{mode}: the deadline must include stderr draining");
        }
    }

    #[test]
    fn cook_can_run_inside_an_existing_async_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();

        runtime.block_on(async { run(child_command("success").unwrap(), Duration::from_secs(5)).unwrap() });
    }
}
