#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

use std::ffi::c_void;

pub type cudaError_t = i32;
pub const cudaSuccess: cudaError_t = 0;

pub const cudaMemcpyHostToDevice: i32 = 1;
pub const cudaMemcpyDeviceToHost: i32 = 2;

unsafe extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> cudaError_t;
    pub fn cudaGetErrorString(error: cudaError_t) -> *const i8;

    pub fn launch_vector_add(
        d_a: *const f32,
        d_b: *const f32,
        d_c: *mut f32,
        n: u32,
    ) -> cudaError_t;
}

pub fn cuda_check(err: cudaError_t, context: &str) {
    if err != cudaSuccess {
        let msg = unsafe {
            let ptr = cudaGetErrorString(err);
            std::ffi::CStr::from_ptr(ptr).to_string_lossy()
        };
        panic!("CUDA error in {context}: {msg} (code {err})")
    }
}
