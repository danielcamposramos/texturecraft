/// Decompressed bytes accepted by one project load, fixed when the load starts.
pub(super) struct LoadBudget {
    limit_bytes: u64,
    used_bytes: u64,
}

impl LoadBudget {
    pub(super) fn new(limit_bytes: u64) -> Self {
        Self { limit_bytes, used_bytes: 0 }
    }

    pub(super) fn remaining(&self) -> u64 {
        self.limit_bytes - self.used_bytes
    }

    /// Reserve bytes before retaining decoded data so entries share one allocation budget.
    pub(super) fn take(&mut self, bytes: u64) -> Result<(), String> {
        if bytes > self.remaining() {
            return Err(format!(
                "project file decompresses to more than {} bytes ({:.2} GiB); increase Max project size on open in Settings > Project to open larger files",
                self.limit_bytes,
                self.limit_bytes as f64 / (1u64 << 30) as f64,
            ));
        }

        self.used_bytes += bytes;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::LoadBudget;

    #[test]
    fn entries_share_the_limit_and_the_exact_boundary_is_allowed() {
        let mut budget = LoadBudget::new(10);

        budget.take(4).unwrap();
        budget.take(6).unwrap();
        let error = budget.take(1).unwrap_err();

        assert_eq!(budget.remaining(), 0);
        assert!(error.contains("10 bytes"), "{error}");
        assert!(error.contains("Settings > Project"), "{error}");
    }

    #[test]
    fn a_rejected_entry_does_not_consume_the_remaining_budget() {
        let mut budget = LoadBudget::new(10);
        budget.take(4).unwrap();

        assert!(budget.take(7).is_err());

        assert_eq!(budget.remaining(), 6);
        assert!(budget.take(6).is_ok());
    }

    #[test]
    fn accounting_cannot_wrap_at_the_integer_boundary() {
        let mut budget = LoadBudget::new(u64::MAX);
        budget.take(u64::MAX).unwrap();

        assert!(budget.take(1).is_err());
        assert!(budget.take(0).is_ok());
        assert_eq!(budget.remaining(), 0);
    }
}
