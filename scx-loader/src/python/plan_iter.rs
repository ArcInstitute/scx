//! Python-iterator → Rust-iterator adapter shared by both plan-driven loaders.
//!
//! Split out of `python.rs` by ORG-9.10-2; a pure move.

use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;

use crate::error::LoaderError;

/// How one plan type is read out of the object a Python generator yielded.
///
/// A function pointer, not a `T: for<'py> FromPyObject<'py>` bound: the two plan
/// types differ in the *diagnostic* they owe a caller who yields the wrong
/// shape, and that bound would collapse both onto pyo3's generic extract error.
/// The cell-set arm in particular names its four arrays and their dtypes, which
/// is the difference between a usable message and "expected tuple of length 4".
type PlanExtract<T> = fn(&Bound<'_, PyAny>) -> std::result::Result<T, LoaderError>;

/// Adapter: Python iterator → Rust `Iterator<Item = Result<T, LoaderError>>`,
/// for both plan-driven loaders.
///
/// `Py<PyAny>` and `fn` pointers are `Send` + `Sync`, so this struct crosses the
/// thread boundary to the plan-pull worker without a `PhantomData` or an
/// `unsafe` impl. Each `next` reacquires the GIL just for the `__next__` call so
/// the GIL is freely available between pulls.
///
/// What is shared is the awkward part — the `__next__` call, telling
/// `StopIteration` from a real exception, and the fact that a `PyErr` cannot
/// cross the plan channel into the consumer's `Result`, so it travels as
/// message text. What stays per-plan-type is the extraction, in [`PlanExtract`].
pub(super) struct PyPlanIterator<T> {
    // `pub(super)` because both plan-driven loaders build this as a struct
    // literal from their own modules; there is no constructor to widen instead.
    pub(super) py_iter: Py<PyAny>,
    pub(super) extract: PlanExtract<T>,
}

impl<T> Iterator for PyPlanIterator<T> {
    type Item = std::result::Result<T, LoaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        Python::attach(|py| {
            let bound = self.py_iter.bind(py);
            match bound.call_method0("__next__") {
                Ok(obj) => Some((self.extract)(&obj)),
                Err(e) => {
                    if e.is_instance_of::<PyStopIteration>(py) {
                        None
                    } else {
                        // Forward the Python exception text. We can't pass
                        // a PyErr through to the consumer's Result type, so
                        // wrap as a ChannelError carrying the message.
                        Some(Err(LoaderError::ChannelError(format!(
                            "plan iterator raised: {e}"
                        ))))
                    }
                }
            }
        })
    }
}
