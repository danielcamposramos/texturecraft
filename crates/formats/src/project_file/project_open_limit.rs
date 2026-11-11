use std::sync::atomic::{AtomicU32, Ordering};

/// Default maximum decompressed project size, in GiB.
pub const DEFAULT_PROJECT_OPEN_LIMIT_GIB: u32 = 8;

static PROJECT_OPEN_LIMIT_GIB: AtomicU32 = AtomicU32::new(DEFAULT_PROJECT_OPEN_LIMIT_GIB);

/// Set the limit for subsequent project loads, in GiB; zero is clamped to one GiB.
/// An in-progress load keeps the limit it started with. Individual entry limits still apply.
pub fn set_project_open_limit_gib(limit_gib: u32) {
    PROJECT_OPEN_LIMIT_GIB.store(limit_gib.max(1), Ordering::Relaxed);
}

/// Maximum decompressed project size accepted by subsequent loads, in GiB.
pub fn project_open_limit_gib() -> u32 {
    PROJECT_OPEN_LIMIT_GIB.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project_file::load_budget::LoadBudget;

    #[test]
    fn preferences_apply_to_new_loads_in_gib_without_disabling_existing_guards() {
        set_project_open_limit_gib(8);
        let mut active_load = LoadBudget::new(u64::from(project_open_limit_gib()) << 30);

        set_project_open_limit_gib(16);
        let mut next_load = LoadBudget::new(u64::from(project_open_limit_gib()) << 30);

        assert!(active_load.take(9 << 30).is_err());
        assert!(next_load.take(9 << 30).is_ok());

        set_project_open_limit_gib(0);
        assert_eq!(project_open_limit_gib(), 1);
        set_project_open_limit_gib(u32::MAX);
        assert_eq!(u64::from(project_open_limit_gib()) << 30, 4_611_686_017_353_646_080);
        set_project_open_limit_gib(DEFAULT_PROJECT_OPEN_LIMIT_GIB);
    }
}
