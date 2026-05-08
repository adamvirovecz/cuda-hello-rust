mod cuda;

use cuda::*;
use std::ffi::c_void;

fn main() {
    let count = 16usize;
    let bytes = count * size_of::<f32>();

    let h_a: Vec<f32> = (0..count).map(|i| i as f32).collect();
    let h_b: Vec<f32> = (10..(count + 10)).map(|i| i as f32).collect();
    let mut h_dest: Vec<f32> = vec![0.0; count];
    println!("A vector ({} elements): {:?}", h_a.len(), h_a);
    println!("B vector ({} elements): {:?}", h_b.len(), h_b);

    let mut d_a: *mut c_void = std::ptr::null_mut();
    let mut d_b: *mut c_void = std::ptr::null_mut();
    let mut d_c: *mut c_void = std::ptr::null_mut();

    unsafe {
        cuda_check(cudaMalloc(&mut d_a, bytes), "cudaMalloc d_a");
        cuda_check(cudaMalloc(&mut d_b, bytes), "cudaMalloc d_b");
        cuda_check(cudaMalloc(&mut d_c, bytes), "cudaMalloc d_c");

        cuda_check(
            cudaMemcpy(
                d_a,
                h_a.as_ptr() as *const c_void,
                bytes,
                cudaMemcpyHostToDevice,
            ),
            "cudaMemcpy h_a to d_a",
        );
        cuda_check(
            cudaMemcpy(
                d_b,
                h_b.as_ptr() as *const c_void,
                bytes,
                cudaMemcpyHostToDevice,
            ),
            "cudaMemcpy h_b to d_b",
        );

        cuda_check(
            launch_vector_add(
                d_a as *const f32,
                d_b as *const f32,
                d_c as *mut f32,
                count as u32,
            ),
            "launch_vector_add",
        );

        cuda_check(
            cudaMemcpy(
                h_dest.as_mut_ptr() as *mut c_void,
                d_c,
                bytes,
                cudaMemcpyDeviceToHost,
            ),
            "cudaMemcpy d_c to h_dest",
        );

        cuda_check(cudaFree(d_a), "cudaFree");
        cuda_check(cudaFree(d_b), "cudaFree");
        cuda_check(cudaFree(d_c), "cudaFree");
    }

    println!("Hello from CUDA & RUST: A + B: {:?}", h_dest);
}
