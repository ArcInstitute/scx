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

    // Check if nvcc is available
    let nvcc_available = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok();

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

            let status = std::process::Command::new("nvcc")
                .arg("--ptx")
                .arg("-O3")
                .arg("--use_fast_math")
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
