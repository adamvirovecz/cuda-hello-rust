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

    const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/vector_add.ptx"));

    unsafe {
        let mut kernel_lib: cudaLibrary_t = std::ptr::null_mut();
        let mut kernel: cudaKernel_t = std::ptr::null_mut();

        let ptx_cstring = std::ffi::CString::new(PTX).unwrap();
        cuda_check(
            cudaLibraryLoadData(
                &mut kernel_lib,
                ptx_cstring.as_ptr() as *const c_void,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            ),
            "loading kernel library",
        );
        cuda_check(
            cudaLibraryGetKernel(&mut kernel, kernel_lib, c"vector_add_kernel".as_ptr()),
            "loading kernel",
        );

        cuda_check(cudaMalloc(&mut d_a, bytes), "cudaMalloc d_a");
        cuda_check(cudaMalloc(&mut d_b, bytes), "cudaMalloc d_b");
        cuda_check(cudaMalloc(&mut d_c, bytes), "cudaMalloc d_c");

        cuda_check(
            cudaMemcpy(
                d_a,
                h_a.as_ptr() as *const c_void,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            ),
            "cudaMemcpy h_a to d_a",
        );
        cuda_check(
            cudaMemcpy(
                d_b,
                h_b.as_ptr() as *const c_void,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            ),
            "cudaMemcpy h_b to d_b",
        );

        let tb: u32 = 32;
        let grid = (count as u32).div_ceil(tb);
        let mut count_arg = count as u32;
        let mut args: [*mut c_void; 4] = [
            &mut d_a as *mut *mut c_void as *mut c_void,
            &mut d_b as *mut *mut c_void as *mut c_void,
            &mut d_c as *mut *mut c_void as *mut c_void,
            &mut count_arg as *mut u32 as *mut c_void,
        ];

        cuda_check(
            cudaLaunchKernel(
                kernel as *const c_void,
                dim3 {
                    x: grid,
                    y: 1,
                    z: 1,
                },
                dim3 { x: tb, y: 1, z: 1 },
                args.as_mut_ptr(),
                0usize,
                std::ptr::null_mut(),
            ),
            "launching kernel",
        );

        cuda_check(
            cudaMemcpy(
                h_dest.as_mut_ptr() as *mut c_void,
                d_c,
                bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            ),
            "cudaMemcpy d_c to h_dest",
        );

        cuda_check(cudaFree(d_a), "cudaFree");
        cuda_check(cudaFree(d_b), "cudaFree");
        cuda_check(cudaFree(d_c), "cudaFree");
    }

    println!("Hello from CUDA & RUST: A + B: {:?}", h_dest);
}
