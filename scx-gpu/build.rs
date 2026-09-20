// CUDA kernel compilation to PTX via nvcc.
//
// Compiles kernels/*.cu into PTX files in OUT_DIR for runtime loading
// via cudarc's CudaContext::load_module(Ptx::from_src(...)).
//
// Target compute capability: compute_70 (Volta+, compatible with H100).
// PTX is forward-compatible, so compute_70 PTX runs on sm_90 (H100).
//
// If nvcc is not available, compilation is skipped gracefully and a
// fallback empty PTX marker is written so include_str!() doesn't fail.

/// Kernels built **without** `--use_fast_math`.
///
/// `--use_fast_math` remaps `logf`→`__logf` and `expf`→`__expf` and implies
/// `--prec-div=false --ftz=true`. Those are single-precision transformations,
/// so the cost of the flag is a property of what a kernel computes in `f32`.
///
/// - `harmony` — `harmony_update_r_kernel` folds `theta * logf(ratio)` into the
///   softmax logit, where `ratio = (2E+1)/(O+E+1)` sits near 1 in a well-mixed
///   cluster and so `logf(ratio)` is often ~1e-3. `__logf`'s documented maximum
///   *absolute* error is 2^-21.41, which at that magnitude is ~0.04 % relative
///   — injected into every penalty term and then exponentiated. The objective
///   and cross-entropy kernels take `logf` too, and `-dist/sigma` is an `f32`
///   division that `--prec-div=false` degrades. The file is memory-bound
///   reductions; there was nothing to buy (review §8.16). (`rsqrtf` at
///   `harmony.cu`'s column normalise is approximate whatever this flag says —
///   exempting the file does not change it.)
/// - `nb_glm` — a precision-sensitive `f64` fitter (Cox–Reid dispersion,
///   digamma/trigamma, IRLS divisions), and the reason its ~1.5e-4 CPU↔GPU
///   agreement floor is what it is. Its source holds **no `float` at all**, so
///   the obvious reading is that these flags cannot reach it — but a PTX diff
///   says otherwise: `--ftz=false` flips `abs.ftz.f32` / `setp.lt.ftz.f32` to
///   their non-ftz forms inside `nb_glm_fit_kernel`, which is libdevice's
///   double `lgamma` doing its range checks in `f32`. The exemption is
///   load-bearing through a transitive call, not through the source's own
///   arithmetic.
///
/// Every other kernel is an `f32` throughput path with no transcendentals — the
/// one borderline case is `normalize_log1p.cu`, whose `log1pf` nvcc does *not*
/// remap but whose `f32` divisions `--prec-div=false` does touch; exempting it
/// would move every GPU-normalized result, so it is left alone deliberately.
///
/// What the flag costs `harmony`, measured by PTX diff at `-arch=compute_70`:
/// 13 `lg2.approx.ftz.f32` and 19 `div.approx.ftz.f32` become 0 and
/// 15 `div.rn.f32` + 4 `rcp.rn.f32`. The `expf` half of the review's claim does
/// **not** reproduce — `ex2.approx.ftz.f32` appears 10 times either way, since
/// that is how nvcc implements precise `expf` too; only the `.ftz` qualifier
/// moves. Likewise `rsqrtf` stays `rsqrt.approx` and merely loses `.ftz`.
const FAST_MATH_EXEMPT: &[&str] = &["nb_glm", "harmony"];

fn main() {
    let kernel_dir = std::path::Path::new("kernels");
    if !kernel_dir.exists() {
        return;
    }

    let cu_files: Vec<std::path::PathBuf> = std::fs::read_dir(kernel_dir)
        .expect("failed to read kernels/ directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "cu") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if cu_files.is_empty() {
        return;
    }

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = std::path::Path::new(&out_dir);

    // Whether nvcc exists is NOT part of this build script's fingerprint —
    // cargo reruns it on `.cu` changes, not on a toolchain appearing. So a
    // target directory that once saw a CPU-only build keeps replaying the stub
    // branch below, *and its warning*, on a machine that has nvcc. That is not
    // hypothetical: the Phase 7 gate job burned a GPU allocation on it, built a
    // wheel from 41-byte PTX files, and died at
    // `CUDA_ERROR_INVALID_IMAGE: device kernel image is invalid` — a runtime
    // symptom four steps from its cause.
    //
    // `SCX_GPU_REQUIRE_NVCC=1` turns that into a build failure instead. Set it
    // in any job that intends to *run* GPU kernels; the stub branch stays the
    // default so a CPU-only `cargo check` still works.
    println!("cargo:rerun-if-env-changed=SCX_GPU_REQUIRE_NVCC");
    // Discovery below is `Command::new("nvcc")`, so **PATH alone** decides
    // whether nvcc resolves — this script never reads CUDA_HOME. PATH is not in
    // the fingerprint by default, which is the *mechanism* behind the trap
    // above, not merely an aggravator: declaring it means moving from a CPU node
    // to a GPU node re-runs this script instead of replaying the stub branch
    // from cache. PATH changes often, so this costs the occasional needless
    // kernel rebuild; that is a better trade than a wheel full of 41-byte PTX.
    //
    // CUDA_HOME is declared too, but as belt-and-braces only: nothing here
    // consults it, so on its own it cannot change whether nvcc is found. It is
    // listed because a toolchain move that edits CUDA_HOME usually edits PATH as
    // well, and a spurious rerun is cheap next to a silent stub.
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    let require_nvcc = std::env::var("SCX_GPU_REQUIRE_NVCC").is_ok_and(|v| v != "0");

    // Check if nvcc is available
    let nvcc_available = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok();

    if !nvcc_available && require_nvcc {
        panic!(
            "SCX_GPU_REQUIRE_NVCC is set but `nvcc` is not on PATH, so every \
             GPU kernel would be an empty PTX stub and fail at load time with \
             CUDA_ERROR_INVALID_IMAGE. Put the CUDA toolkit on PATH (a conda \
             env with `cuda-nvcc`, or /usr/local/cuda/bin), or unset \
             SCX_GPU_REQUIRE_NVCC to accept stubs."
        );
    }

    if !nvcc_available {
        println!(
            "cargo:warning=nvcc not found — writing empty PTX stubs. \
             GPU kernels will not be available until built on a machine with the CUDA toolkit."
        );
        // Write empty marker files so include_str!() doesn't fail at compile time
        for cu_file in &cu_files {
            let stem = cu_file.file_stem().unwrap().to_str().unwrap();
            let ptx_path = out_path.join(format!("{stem}.ptx"));
            // The first line is a sentinel `GpuDevice::load_module_cached`
            // checks, so a stub is refused at load with an actionable message
            // instead of surfacing four steps later as an opaque
            // `CUDA_ERROR_INVALID_IMAGE` from whichever kernel ran first.
            // Keep it byte-identical to `scx_gpu::device::PTX_STUB_MARKER`;
            // `device.rs`'s `the_stub_marker_matches_the_one_build_rs_writes`
            // reads this file and asserts they agree.
            std::fs::write(
                &ptx_path,
                "// SCX_PTX_IS_STUB — nvcc not available at build time\n",
            )
            .expect("failed to write PTX stub");
        }
    } else {
        // Compile each .cu file to PTX
        for cu_file in &cu_files {
            let stem = cu_file.file_stem().unwrap().to_str().unwrap();
            let ptx_path = out_path.join(format!("{stem}.ptx"));

            let mut cmd = std::process::Command::new("nvcc");
            cmd.arg("--ptx").arg("-O3");
            if FAST_MATH_EXEMPT.contains(&stem) {
                cmd.arg("--prec-div=true")
                    .arg("--prec-sqrt=true")
                    .arg("--ftz=false");
            } else {
                cmd.arg("--use_fast_math");
            }
            let status = cmd
                .arg("-arch=compute_70") // PTX is forward-compatible (runs on sm_90)
                .arg("-o")
                .arg(&ptx_path)
                .arg(cu_file)
                .status()
                .expect("failed to run nvcc");

            if !status.success() {
                panic!(
                    "nvcc failed to compile {} to PTX (exit code: {:?})",
                    cu_file.display(),
                    status.code()
                );
            }
        }
    }

    // Re-run build if any kernel file changes — and watch the directory itself
    // so newly added `.cu` files trigger a rebuild (per-file watches alone
    // never fire for a file that didn't exist on the previous run).
    println!("cargo:rerun-if-changed=kernels");
    for cu_file in &cu_files {
        println!("cargo:rerun-if-changed={}", cu_file.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
}
