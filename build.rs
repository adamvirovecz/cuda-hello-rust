use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Build script for MSVC / CUDA on Windows 11
    // Build kernel with nvcc
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_obj = out_dir.join("vector_add.o");
    let kernel_lib = out_dir.join("vector_add.lib");

    // Locate cl.exe via vswhere and inject into PATH for nvcc
    let cl_dir = {
        let vswhere = std::path::Path::new(
            r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
        );
        let output = Command::new(vswhere)
            .args(["-latest", "-find", r"VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe"])
            .output()
            .expect("vswhere not found");
        let cl_path = String::from_utf8(output.stdout).unwrap();
        PathBuf::from(cl_path.trim()).parent().unwrap().to_path_buf()
    };

    let path = env::var("PATH").unwrap_or_default();
    let new_path = format!("{};{}", cl_dir.display(), path);

    let nvcc = Command::new("nvcc")
        .env("PATH", &new_path)
        .args([
            "-c", "kernels/vector_add.cu",
            "-O2",
            "--gpu-architecture=compute_89", "--gpu-code=sm_89",
            "-o",
        ])
        .arg(&kernel_obj)
        .status()
        .expect("nvcc not found");
    assert!(nvcc.success(), "nvcc compilation failed");

    // create static library from object with MSVC lib.exe
    let lib_status = Command::new("lib")
        .env("PATH", &new_path)
        .arg("/nologo")
        .arg(format!("/OUT:{}", kernel_lib.display()))
        .arg(&kernel_obj)
        .status()
        .expect("lib.exe not found");
    assert!(lib_status.success(), "lib.exe failed to create static library");

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

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=vector_add");

    println!("cargo:rerun-if-changed=kernels/vector_add.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
