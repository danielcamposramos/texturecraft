//! A new Path tool path previewed in the document before it is applied

use serde_json::Value;

use crate::Document;

/// A new path previewed in the document before it is applied
pub(crate) struct PathDraft {
    /// The document before the preview
    pub(crate) base: Document,
    /// What painting the preview last reported (the commit's report when nothing changed since)
    pub(crate) painted: Value,
}
