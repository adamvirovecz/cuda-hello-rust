use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/vector_add.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let kernel_src = PathBuf::from("kernels/vector_add.cu");
    let ptx_file = out_dir.join("vector_add.ptx");
    let arch = "compute_89";
    let code = "sm_89";

    let compiler = cc::Build::new().get_compiler();
    let host_compiler = compiler.path();

    let nvcc_status = Command::new("nvcc")
        .arg("-ptx")
        .arg("-o")
        .arg(&ptx_file)
        .arg(&kernel_src)
        .arg(format!("-arch={}", arch))
        .arg(format!("-code={}", code))
        .arg("-ccbin")
        .arg(host_compiler)
        .status()
        .unwrap();

    assert!(nvcc_status.success(), "nvcc kernel compilation failed!");

    // Link CUDA runtime (static)
    if let Some(cuda_path) = env::var_os("CUDA_PATH") {
        let cuda_path = PathBuf::from(cuda_path);
        println!("cargo:rustc-link-search=native={}", cuda_path.display());
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_path.join("lib").join("x64").display()
        );
    }

    println!("cargo:rustc-link-lib=static:+whole-archive=cudart_static");
}
