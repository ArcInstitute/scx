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
            std::fs::write(&ptx_path, "// nvcc not available — empty PTX stub\n")
                .expect("failed to write PTX stub");
        }
    } else {
        // Compile each .cu file to PTX
        for cu_file in &cu_files {
            let stem = cu_file.file_stem().unwrap().to_str().unwrap();
            let ptx_path = out_path.join(format!("{stem}.ptx"));

            let mut cmd = std::process::Command::new("nvcc");
            cmd.arg("--ptx").arg("-O3");
            // `nb_glm` is a precision-sensitive `f64` fitter (Cox–Reid dispersion,
            // digamma/trigamma, IRLS divisions). `--use_fast_math` implies
            // `--prec-div=false --ftz=true`, which degrades exactly those f64
            // divisions and is the mechanism behind its ~1.5e-4 CPU↔GPU agreement
            // floor — so build it with precise division/sqrt and no flush-to-zero.
            // The other kernels are `f32` throughput paths and keep fast-math.
            if stem == "nb_glm" {
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
