//! One traversal for every **1-D** HDF5 string *dataset* on the ingest side.
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
//! unrepresentable *on this shape*.
//!
//! # What is NOT on this traversal
//!
//! Only 1-D dataset reads are. Three ingest paths still read strings directly
//! and are unchanged by this module:
//!
//! * **rank-0 (scalar) dataset reads** — `read.rs`'s `uns` scalar arm and
//!   `cellbender.rs`'s metadata scalars both do `read_scalar::<VarLenUnicode>()`.
//!   They cannot share this code: the const-generic dispatch here is over
//!   `read_1d`, and a scalar needs `read_scalar`. **The same fixed-width drift
//!   lives there** — `uns`'s `FixedUnicode(_) | FixedAscii(_)` scalar arm reads
//!   the body as `VarLenUnicode`, exactly as the two dataset readers did before
//!   this change. A scalar companion to this dispatcher would close it.
//! * **1-D `uns` string arrays**, which route their var-length arm through
//!   [`read_string_dataset`] but have no fixed-width arm at all, so a
//!   fixed-width `uns` array is still rejected rather than read.
//! * **attribute** reads — `encoding-type`, `column-order`, the legacy
//!   `categories` attribute form. Attributes are not datasets and anndata only
//!   ever writes var-length there.
//!
//! # How it works
//!
//! The traversal reads each element **by reference** and appends it to a
//! [`StringSink`]. Nothing clones the HDF5 element type: `VarLenUnicode` is a
//! bare `{ ptr }` with a deep-copying `Clone`, so the `read_1d()?.to_vec()`
//! this replaces paid a `malloc` + `memcpy` per row before `to_string()` paid
//! a second one. A sink that appends into an Arrow buffer copies the payload
//! **once**.
//!
//! There is **one HDF5 read and one copy of the payload** into the
//! destination — not one traversal of the resident elements. A sink that wants
//! its buffer pre-sized needs the exact byte total, and that is its own walk
//! over the (already decoded) array; [`StringSink::WANTS_BYTE_COUNT`] is how a
//! sink that cannot use it declines to pay for it.
//!
//! Sinks are constructed by the traversal, not by the caller, because the item
//! count and the byte total are only known after the read — and passing both
//! is what lets the Arrow builders pre-size their value buffer instead of
//! growing it by doubling the way `StringArray::from_iter_values` does.
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

/// Where a string traversal puts what it reads.
///
/// `Config` is whatever the sink needs that the traversal cannot know — the
/// validity mask, for [`MaskedUtf8Sink`]. It is `Copy` so the width ladder can
/// hand it to whichever arm fires without an `Option` dance. The alternative,
/// dropping the associated type and passing `Option<&[bool]>` to every sink,
/// puts a parameter two of the three impls must accept and ignore;
/// `Config = ()` says "needs nothing" once, in the impl where it is true.
///
/// The method names deliberately avoid `Vec`'s and `GenericByteBuilder`'s
/// inherent ones: both of those *are* sinks here, and a name collision would
/// be resolved silently in favour of the inherent method.
trait StringSink: Sized {
    type Config: Copy;

    /// Whether this sink uses the payload byte count. Summing it costs a walk
    /// over the resident elements, and the owned path has nothing to spend it
    /// on — a `Vec<String>` allocates per element regardless.
    const WANTS_BYTE_COUNT: bool = true;

    /// Called exactly once, with the exact element count and — when
    /// [`StringSink::WANTS_BYTE_COUNT`] — the exact payload size in bytes,
    /// before any [`StringSink::append`].
    fn with_reservation(config: Self::Config, items: usize, bytes: usize) -> Self;

    fn append(&mut self, s: &str);
}

/// Owned `String`s — the shape the five non-Arrow consumers need (barcode-join
/// hashmap keys, `serde_json::Value::String`).
impl StringSink for Vec<String> {
    type Config = ();
    const WANTS_BYTE_COUNT: bool = false;

    fn with_reservation(_config: (), items: usize, _bytes: usize) -> Self {
        Vec::with_capacity(items)
    }

    fn append(&mut self, s: &str) {
        self.push(s.to_owned());
    }
}

/// An Arrow `Utf8` array with **no validity buffer**.
///
/// Byte-identity with the `StringArray::from(Vec<&str>)` this replaces rests
/// on `NullBufferBuilder` allocating lazily: `append_value` records a `true`
/// bit but the buffer materialises only once a `false` arrives, so `finish()`
/// yields `nulls: None` exactly as `from_iter_values` does. **Nothing on this
/// path may call `append_null` / `append_option(None)`** — an all-valid null
/// buffer is not the same array. That is why the masked form below is a
/// separate sink rather than a flag on this one.
impl StringSink for GenericStringBuilder<i32> {
    type Config = ();

    fn with_reservation(_config: (), items: usize, bytes: usize) -> Self {
        GenericStringBuilder::with_capacity(items, bytes)
    }

    fn append(&mut self, s: &str) {
        self.append_value(s);
    }
}

/// An Arrow `Utf8` array carrying validity bits from an external mask, for
/// anndata's `nullable-string-array` group form (`mask[i] == true` ⇔ null,
/// null positions filled with `""` on disk).
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

    fn with_reservation(mask: &'a [bool], items: usize, bytes: usize) -> Self {
        Self {
            builder: GenericStringBuilder::with_capacity(items, bytes),
            mask,
            next: 0,
        }
    }

    fn append(&mut self, s: &str) {
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
///
/// `AsRef<str>` is implemented for all four `hdf5::types` string types, so no
/// local adapter trait is needed.
fn read_and_drain<T, S>(ds: &hdf5::Dataset, config: S::Config) -> Result<S, ConvertError>
where
    T: hdf5::H5Type + AsRef<str>,
    S: StringSink,
{
    let array = ds.read_1d::<T>()?;
    let bytes: usize = if S::WANTS_BYTE_COUNT {
        array.iter().map(|s| s.as_ref().len()).sum()
    } else {
        0
    };
    let mut sink = S::with_reservation(config, array.len(), bytes);
    for s in array.iter() {
        sink.append(s.as_ref());
    }
    Ok(sink)
}

/// Fixed-length HDF5 strings are read through a const-generic type, so the
/// width must be known at compile time. Dispatch the runtime width up to the
/// next size in this ladder: HDF5 performs the string-size conversion, so
/// reading an `N`-byte dataset as `M >= N` is lossless.
///
/// The ladder is keyed on **bytes**, which is why a multi-byte UTF-8 fixture
/// belongs in its tests: a `FixedUnicode(24)` column holding eight three-byte
/// characters is a 24-byte dataset, not an 8-byte one.
///
/// Expanded only inside [`read_string_dataset_as`], and it reads that
/// function's `S` and `ConvertError` from the expansion site rather than
/// taking them as macro arguments — which is why it is defined here and not
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
/// The one funnel for that shape: all four string flavours, one width ladder,
/// one HDF5 read. Consumers pick their output by picking a sink. See the
/// module docs for the reads this does **not** cover.
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
    Ok(read_string_dataset_as::<GenericStringBuilder<i32>>(ds, ())?.finish())
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
    read_string_dataset_as::<Vec<String>>(ds, ())
}
