fn main() {
    println!("cargo:rerun-if-changed=kernels/vector_add.cu");

    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .flag("-gencode").flag("arch=compute_89,code=sm_89")
        .flag("-O2")
        .file("kernels/vector_add.cu")
        .compile("vector_add");
}
