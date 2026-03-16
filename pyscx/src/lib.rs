mod anndata;
mod experiment;

use pyo3::prelude::*;

#[pymodule]
fn pyscx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m;
    Ok(())
}
