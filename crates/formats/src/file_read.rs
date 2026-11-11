use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::{FormatError, Result};

/// Read a regular file, enforcing `cap` even if it grows while being read.
pub fn read_capped(path: &Path, cap: u64) -> Result<Vec<u8>> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);

    // A replaced FIFO must not block before its handle can be rejected as non-regular
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(&mut options, rustix::fs::OFlags::NONBLOCK.bits() as i32);

    read_open_file(options.open(path)?, path, cap)
}

/// Use the same handle for metadata and contents, including capability-opened files.
pub(crate) fn read_open_file(file: File, path: &Path, cap: u64) -> Result<Vec<u8>> {
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(FormatError::Invalid(format!("{} is not a regular file", path.display())));
    }

    if metadata.len() > cap {
        return Err(FormatError::TooLarge(format!("{} is {} bytes (limit {cap})", path.display(), metadata.len())));
    }

    read_with_cap(file, path, cap, metadata.len())
}

fn read_with_cap(reader: impl Read, path: &Path, cap: u64, expected_bytes: u64) -> Result<Vec<u8>> {
    let capacity = usize::try_from(expected_bytes.min(cap)).map_err(|_| FormatError::TooLarge(path.display().to_string()))?;
    let mut bytes = Vec::new();

    // Reserve the known file size so probing EOF does not double an already full buffer
    bytes.try_reserve_exact(capacity).map_err(|error| FormatError::TooLarge(format!("{}: {error}", path.display())))?;

    reader.take(cap.saturating_add(1)).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > cap {
        return Err(FormatError::TooLarge(format!("{} exceeds {cap} bytes", path.display())));
    }

    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn actual_read_is_bounded_when_more_bytes_arrive_after_metadata() {
        let mut reader = Cursor::new(vec![7; 128]);

        let result = read_with_cap(&mut reader, Path::new("growing-file"), 16, 16);

        assert!(matches!(result, Err(FormatError::TooLarge(_))));
        assert_eq!(reader.position(), 17, "only one byte beyond the budget may be read");
    }

    #[test]
    fn exact_empty_and_maximum_caps_are_supported() {
        let path = Path::new("file");

        assert_eq!(read_with_cap(&b"abc"[..], path, 3, 3).unwrap(), b"abc");
        assert!(read_with_cap(&b""[..], path, 0, 0).unwrap().is_empty());
        assert!(matches!(read_with_cap(&b"a"[..], path, 0, 0), Err(FormatError::TooLarge(_))));
        assert_eq!(read_with_cap(&b"abc"[..], path, u64::MAX, 3).unwrap(), b"abc");
    }

    #[test]
    fn known_file_size_avoids_doubling_the_read_buffer() {
        let contents = vec![7; 8192];

        let bytes = read_with_cap(Cursor::new(&contents), Path::new("stable-file"), 16384, contents.len() as u64).unwrap();

        assert_eq!(bytes, contents);
        assert!(bytes.capacity() < contents.len() * 2, "checking EOF must not double the file allocation");
    }

    #[test]
    fn size_hint_does_not_change_the_read_limit() {
        let path = Path::new("changing-file");

        assert_eq!(read_with_cap(&b"abc"[..], path, 3, 0).unwrap(), b"abc");
        assert_eq!(read_with_cap(&b"abc"[..], path, 3, u64::MAX).unwrap(), b"abc");
        assert!(matches!(read_with_cap(&b"abcd"[..], path, 3, 1), Err(FormatError::TooLarge(_))));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_non_regular_files_without_waiting_for_a_fifo_writer() {
        use std::hash::{BuildHasher, Hasher};
        let nonce = std::collections::hash_map::RandomState::new().build_hasher().finish();
        let directory = std::env::temp_dir().join(format!("tc-capped-read-{nonce:016x}"));
        std::fs::create_dir(&directory).unwrap();
        let fifo = directory.join("fifo");
        rustix::fs::mkfifoat(rustix::fs::CWD, &fifo, rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR).unwrap();
        std::fs::write(directory.join("regular"), b"pixels").unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        let reader_path = fifo.clone();
        let worker = std::thread::spawn(move || sender.send(read_capped(&reader_path, 16)).unwrap());

        let result = receiver.recv_timeout(std::time::Duration::from_secs(1));
        if result.is_err() {
            // Release a regressed blocking open so a failed test never leaves a stuck thread
            drop(std::fs::OpenOptions::new().write(true).open(&fifo).unwrap());
        }

        worker.join().unwrap();

        assert!(matches!(result.unwrap(), Err(FormatError::Invalid(_))));
        assert!(matches!(read_capped(&directory, 16), Err(FormatError::Invalid(_))));
        assert_eq!(read_capped(&directory.join("regular"), 6).unwrap(), b"pixels");
        assert!(matches!(read_capped(&directory.join("regular"), 5), Err(FormatError::TooLarge(_))));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
