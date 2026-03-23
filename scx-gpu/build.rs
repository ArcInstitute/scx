// CUDA kernel compilation via `cc` crate.
//
// Compiles kernels/*.cu into a static library linked via Rust FFI.
// Target compute capabilities:
//   - sm_90 (H100, primary target)
//   - sm_70 (Volta/Ampere fallback)
//
// The cc crate's Build::cuda(true) method uses nvcc under the hood.
// If nvcc is not available, compilation is skipped gracefully.

fn main() {
    // Collect all .cu files in the kernels/ directory
    let kernel_dir = std::path::Path::new("kernels");
    if !kernel_dir.exists() {
        return;
    }

    let cu_files: Vec<std::path::PathBuf> = std::fs::read_dir(kernel_dir)
        .expect("failed to read kernels/ directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "cu") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if cu_files.is_empty() {
        return;
    }

    // Check if nvcc is available before attempting CUDA compilation
    let nvcc_check = std::process::Command::new("nvcc")
        .arg("--version")
        .output();

    if nvcc_check.is_err() {
        println!(
            "cargo:warning=nvcc not found — skipping CUDA kernel compilation. \
             GPU kernels will not be available until built on a machine with the CUDA toolkit."
        );
        return;
    }

    let mut build = cc::Build::new();
    build.cuda(true);

    // Optimization flags
    build.flag("-O3");

    // Target compute capabilities: sm_70 (Volta+) and sm_90 (H100)
    build.flag("-gencode=arch=compute_70,code=sm_70");
    build.flag("-gencode=arch=compute_90,code=sm_90");

    // Enable fast math optimizations
    build.flag("--use_fast_math");

    for cu_file in &cu_files {
        build.file(cu_file);
    }

    build.compile("scx_gpu_kernels");

    // Re-run build if any kernel file changes
    for cu_file in &cu_files {
        println!("cargo:rerun-if-changed={}", cu_file.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
}
