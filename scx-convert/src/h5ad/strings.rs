//! One traversal for every HDF5 string dataset on the ingest side.
//!
//! HDF5 has four string flavours and producers pick freely: h5py/anndata emit
//! variable-length UTF-8, while PyTables `create_carray` (CellRanger and
//! CellBender outputs) emits **fixed-length ASCII**. The two families need
//! different read types — `read_1d::<VarLenUnicode>()` on a fixed-length
//! dataset fails with an opaque "no conversion paths found" — so the fixed
//! widths have to go through a const-generic type, dispatched up a width
//! ladder.
//!
//! That ladder used to be duplicated by hand at every site that wanted the
//! strings in a different shape, and **two sites had already drifted off it**:
//! `read_categorical_values` and `read_nullable_string_group` both matched
//! `FixedUnicode(_) | FixedAscii(_)` in their arms while reading the body as
//! var-length, so a PyTables-written `categories` or nullable `values` dataset
//! failed with exactly the error the ladder exists to prevent. Routing every
//! consumer through one traversal is what makes that class of drift
//! unrepresentable.
//!
//! The traversal reads each element **by reference** and pushes it into a
//! [`StringSink`]. Nothing clones the HDF5 element type: `VarLenUnicode` is a
//! bare `{ ptr }` with a deep-copying `Clone`, so the `read_1d()?.to_vec()`
//! this replaces paid a `malloc` + `memcpy` per row before `to_string()` paid
//! a second one. A sink that appends into an Arrow buffer copies the payload
//! **once**.
//!
//! Sinks are constructed by the traversal, not by the caller, because the
//! exact item count *and* the exact payload byte count are only known after
//! the read — and passing both is what lets [`Utf8Sink`] pre-size its value
//! buffer instead of growing it by doubling the way
//! `StringArray::from_iter_values` does.
//!
//! # What this does and does not buy, measured
//!
//! It buys read-phase wall and a lower read-phase transient. It does **not**
//! lower the peak RSS of a full `from_h5ad`, and the reason is worth writing
//! down so nobody re-derives it: on a 10M-row obs-heavy conversion the process
//! high-water mark lands at ~0.85 of total wall — in the *writer* — and is
//! identical either way. Sampling RSS at 5 ms through the conversion, the read
//! phase plateaus ~400 MB lower here than it did through the four-copy chain,
//! and then both curves climb past it to the same peak while obs is
//! serialised. A read-only workload (`read_h5ad_metadata`, no writer
//! downstream) does show it: ~1.06x lower peak and ~1.24x faster.
//!
//! So the remaining obs term is on the **write** side, not this one.

use arrow::array::{GenericStringBuilder, StringArray};
use hdf5::types::TypeDescriptor;

use crate::pipeline::ConvertError;

/// An HDF5 string element that can be borrowed as `&str`.
///
/// Implemented for the three read types the ladder below uses.
/// `TypeDescriptor::VarLenAscii` datasets are read as `VarLenUnicode`, which
/// is what the pre-existing reader did and what HDF5's own conversion path
/// supports.
trait H5Str {
    fn as_str(&self) -> &str;
}

impl H5Str for hdf5::types::VarLenUnicode {
    fn as_str(&self) -> &str {
        // Fully qualified deliberately. The inherent method wins resolution
        // over the trait one, so the short form is not a recursion -- but it
        // reads like one, and it would become one if hdf5 ever moved `as_str`
        // onto a trait.
        hdf5::types::VarLenUnicode::as_str(self)
    }
}

impl<const N: usize> H5Str for hdf5::types::FixedAscii<N> {
    fn as_str(&self) -> &str {
        hdf5::types::FixedAscii::<N>::as_str(self)
    }
}

impl<const N: usize> H5Str for hdf5::types::FixedUnicode<N> {
    fn as_str(&self) -> &str {
        hdf5::types::FixedUnicode::<N>::as_str(self)
    }
}

/// Where a string traversal puts what it reads.
///
/// `Config` is whatever the sink needs that the traversal cannot know — the
/// validity mask, for [`MaskedUtf8Sink`]. It is `Copy` so the width ladder can
/// hand it to whichever arm fires without an `Option` dance.
trait StringSink: Sized {
    type Config: Copy;

    /// Called exactly once, with the exact element count and the exact total
    /// payload size in bytes, before any [`StringSink::push`].
    fn with_capacity(config: Self::Config, items: usize, bytes: usize) -> Self;

    fn push(&mut self, s: &str);
}

/// Collects owned `String`s — the shape the five non-Arrow consumers need
/// (barcode-join hashmap keys, `serde_json::Value::String`).
struct OwnedSink(Vec<String>);

impl OwnedSink {
    fn into_vec(self) -> Vec<String> {
        self.0
    }
}

impl StringSink for OwnedSink {
    type Config = ();

    fn with_capacity(_config: (), items: usize, _bytes: usize) -> Self {
        Self(Vec::with_capacity(items))
    }

    fn push(&mut self, s: &str) {
        self.0.push(s.to_owned());
    }
}

/// Builds an Arrow `Utf8` array with **no validity buffer**.
///
/// Byte-identity with the `StringArray::from(Vec<&str>)` this replaces rests
/// on `NullBufferBuilder` allocating lazily: `append_value` records a `true`
/// bit but the buffer materialises only once a `false` arrives, so `finish()`
/// yields `nulls: None` exactly as `from_iter_values` does. **Never call
/// `append_null` / `append_option(None)` here** — an all-valid null buffer is
/// not the same array.
struct Utf8Sink(GenericStringBuilder<i32>);

impl Utf8Sink {
    fn finish(mut self) -> StringArray {
        self.0.finish()
    }
}

impl StringSink for Utf8Sink {
    type Config = ();

    fn with_capacity(_config: (), items: usize, bytes: usize) -> Self {
        Self(GenericStringBuilder::with_capacity(items, bytes))
    }

    fn push(&mut self, s: &str) {
        self.0.append_value(s);
    }
}

/// Builds an Arrow `Utf8` array carrying validity bits from an external mask,
/// for anndata's `nullable-string-array` group form (`mask[i] == true` ⇔
/// null, null positions filled with `""` on disk).
struct MaskedUtf8Sink<'a> {
    builder: GenericStringBuilder<i32>,
    mask: &'a [bool],
    next: usize,
}

impl MaskedUtf8Sink<'_> {
    fn finish(mut self) -> StringArray {
        self.builder.finish()
    }
}

impl<'a> StringSink for MaskedUtf8Sink<'a> {
    type Config = &'a [bool];

    fn with_capacity(mask: &'a [bool], items: usize, bytes: usize) -> Self {
        Self {
            builder: GenericStringBuilder::with_capacity(items, bytes),
            mask,
            next: 0,
        }
    }

    fn push(&mut self, s: &str) {
        // `mask` length is checked against the dataset shape before the read,
        // so an out-of-range index here is unreachable; treat a short mask as
        // "not null" rather than panicking in a reader.
        let is_null = self.mask.get(self.next).copied().unwrap_or(false);
        self.next += 1;
        if is_null {
            self.builder.append_null();
        } else {
            self.builder.append_value(s);
        }
    }
}

/// Read the whole dataset as `T`, size the sink from what came back, then
/// append every element by reference.
fn read_and_drain<T, S>(ds: &hdf5::Dataset, config: S::Config) -> Result<S, ConvertError>
where
    T: hdf5::H5Type + H5Str,
    S: StringSink,
{
    let array = ds.read_1d::<T>()?;
    let bytes: usize = array.iter().map(|s| s.as_str().len()).sum();
    let mut sink = S::with_capacity(config, array.len(), bytes);
    for s in array.iter() {
        sink.push(s.as_str());
    }
    Ok(sink)
}

/// Fixed-length HDF5 strings are read through a const-generic type, so the
/// width must be known at compile time. Dispatch the runtime width up to the
/// next size in this ladder: HDF5 performs the string-size conversion, so
/// reading an `N`-byte dataset as `M >= N` is lossless.
///
/// The ladder is keyed on **bytes**, which is why a multi-byte UTF-8 fixture
/// belongs in its tests: a `FixedUnicode(24)` column holding eight
/// three-byte characters is a 24-byte dataset, not an 8-byte one.
/// Expanded only inside [`read_string_dataset_as`], and it reads that
/// function's `S` and `ConvertError` from the expansion site rather than
/// taking them as macro arguments -- which is why it is defined here and not
/// exported.
macro_rules! fixed_width_ladder {
    ($ds:expr, $config:expr, $ty:ident, $size:expr, $($cap:literal),+) => {{
        let size = $size;
        let mut out: Option<Result<S, ConvertError>> = None;
        $(
            if out.is_none() && size <= $cap {
                out = Some(read_and_drain::<hdf5::types::$ty<$cap>, S>($ds, $config));
            }
        )+
        out.unwrap_or_else(|| {
            Err(ConvertError::UnsupportedDtype(format!(
                "dataset '{}' has fixed-length strings of {size} bytes, beyond \
                 the largest supported width",
                $ds.name()
            )))
        })
    }};
}

/// Read any 1-D HDF5 string dataset into `S`.
///
/// This is the single funnel: all four string flavours, one width ladder, one
/// pass over the data. Consumers pick their shape by picking a sink.
fn read_string_dataset_as<S: StringSink>(
    ds: &hdf5::Dataset,
    config: S::Config,
) -> Result<S, ConvertError> {
    let desc = ds.dtype()?.to_descriptor()?;
    match &desc {
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
            read_and_drain::<hdf5::types::VarLenUnicode, S>(ds, config)
        }
        TypeDescriptor::FixedAscii(size) => {
            fixed_width_ladder!(ds, config, FixedAscii, *size, 16, 32, 64, 128, 256, 1024, 4096)
        }
        TypeDescriptor::FixedUnicode(size) => {
            fixed_width_ladder!(
                ds,
                config,
                FixedUnicode,
                *size,
                16,
                32,
                64,
                128,
                256,
                1024,
                4096
            )
        }
        other => Err(ConvertError::UnsupportedDtype(format!(
            "dataset '{}' is not a string dataset: {other:?}",
            ds.name()
        ))),
    }
}

/// Read a 1-D HDF5 string dataset as an Arrow `Utf8` array, with no validity
/// buffer. The Arrow-side replacement for
/// `StringArray::from(read_string_dataset(ds)?.iter().map(…).collect::<Vec<_>>())`
/// — same bytes, same i32 offsets, no `Vec<String>` and no `Vec<&str>`.
pub(crate) fn read_string_array(ds: &hdf5::Dataset) -> Result<StringArray, ConvertError> {
    Ok(read_string_dataset_as::<Utf8Sink>(ds, ())?.finish())
}

/// Read a 1-D HDF5 string dataset as an Arrow `Utf8` array carrying validity
/// bits from `mask` (`mask[i] == true` ⇔ null).
///
/// The length agreement is checked against the dataset **shape**, before the
/// read, so a mismatched pair does not pay for a full decode first. `name` is
/// the column name, for the error message.
pub(crate) fn read_masked_string_array(
    ds: &hdf5::Dataset,
    mask: &[bool],
    name: &str,
) -> Result<StringArray, ConvertError> {
    let shape = ds.shape();
    let values_len = shape.first().copied().unwrap_or(0);
    if values_len != mask.len() {
        return Err(ConvertError::Other(format!(
            "nullable-string-array '{name}': values len {} != mask len {}",
            values_len,
            mask.len()
        )));
    }
    Ok(read_string_dataset_as::<MaskedUtf8Sink>(ds, mask)?.finish())
}

/// Read any 1-D HDF5 string dataset as owned `String`s.
///
/// Kept because five call sites genuinely need owned strings — CellBender's
/// barcode / feature join keys and two `serde_json::Value::String` builders —
/// and would gain nothing from an Arrow builder. Arrow consumers must use
/// [`read_string_array`] or [`read_masked_string_array`] instead; going
/// through here and re-collecting is the four-copy chain this module removes.
pub(crate) fn read_string_dataset(ds: &hdf5::Dataset) -> Result<Vec<String>, ConvertError> {
    Ok(read_string_dataset_as::<OwnedSink>(ds, ())?.into_vec())
}
