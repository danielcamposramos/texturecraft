use std::cell::Cell;
use std::io::{self, Read, Seek, SeekFrom};

/// Restrict ZIP indexing to a bounded tail and read allowance before enabling payload access.
pub(super) struct MetadataReader<'a, R> {
    reader: R,
    metadata_only: &'a Cell<bool>,
    metadata_start: u64,
    remaining_bytes: u64,
    position: u64,
}

impl<'a, R: Read + Seek> MetadataReader<'a, R> {
    pub(super) fn new(mut reader: R, metadata_only: &'a Cell<bool>, metadata_bytes: u64) -> io::Result<Self> {
        let end = reader.seek(SeekFrom::End(0))?;
        Ok(Self { reader, metadata_only, metadata_start: end.saturating_sub(metadata_bytes), remaining_bytes: metadata_bytes.saturating_mul(2), position: end })
    }
}

impl<R: Read> Read for MetadataReader<'_, R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }

        let metadata_only = self.metadata_only.get();
        if metadata_only && (self.position < self.metadata_start || self.remaining_bytes == 0) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "project ZIP metadata exceeds the indexing limit"));
        }

        let length = if metadata_only { bytes.len().min(usize::try_from(self.remaining_bytes).unwrap_or(usize::MAX)) } else { bytes.len() };
        let read = self.reader.read(bytes.get_mut(..length).unwrap_or_default())?;
        self.position = self.position.saturating_add(read as u64);
        if metadata_only {
            self.remaining_bytes -= read as u64;
        }

        Ok(read)
    }
}

impl<R: Seek> Seek for MetadataReader<'_, R> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.position = self.reader.seek(position)?;
        Ok(self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn indexing_cannot_read_payloads_outside_the_metadata_window() {
        let metadata_only = Cell::new(true);
        let mut reader = MetadataReader::new(Cursor::new(vec![7; 100]), &metadata_only, 8).unwrap();
        reader.seek(SeekFrom::Start(91)).unwrap();

        assert!(reader.read(&mut [0; 1]).is_err());
        reader.seek(SeekFrom::Start(92)).unwrap();
        reader.read_exact(&mut [0; 8]).unwrap();

        metadata_only.set(false);
        reader.seek(SeekFrom::Start(0)).unwrap();
        reader.read_exact(&mut [0; 100]).unwrap();
    }

    #[test]
    fn repeated_metadata_scans_share_one_read_allowance() {
        let metadata_only = Cell::new(true);
        let mut reader = MetadataReader::new(Cursor::new(vec![7; 8]), &metadata_only, 8).unwrap();
        for _ in 0..2 {
            reader.seek(SeekFrom::Start(0)).unwrap();
            reader.read_exact(&mut [0; 8]).unwrap();
        }
        reader.seek(SeekFrom::Start(0)).unwrap();

        assert!(reader.read(&mut [0; 1]).is_err());
    }
}
