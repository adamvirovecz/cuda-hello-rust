#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

use std::ffi::{c_char, c_uint, c_void};

pub type cudaError_t = i32;
pub const cudaSuccess: cudaError_t = 0;

#[allow(dead_code)]
#[repr(u32)]
#[derive(Debug, Copy, Clone)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUlib_st {
    _unused: [u8; 0],
}
pub type cudaLibrary_t = *mut CUlib_st;

#[allow(dead_code)]
#[repr(u32)]
#[derive(Debug, Copy, Clone)]
pub enum cudaJitOption {
    cudaJitMaxRegisters = 0,
    cudaJitThreadsPerBlock = 1,
    cudaJitWallTime = 2,
    cudaJitInfoLogBuffer = 3,
    cudaJitInfoLogBufferSizeBytes = 4,
    cudaJitErrorLogBuffer = 5,
    cudaJitErrorLogBufferSizeBytes = 6,
    cudaJitOptimizationLevel = 7,
    cudaJitFallbackStrategy = 10,
    cudaJitGenerateDebugInfo = 11,
    cudaJitLogVerbose = 12,
    cudaJitGenerateLineInfo = 13,
    cudaJitCacheMode = 14,
    cudaJitPositionIndependentCode = 30,
    cudaJitMinCtaPerSm = 31,
    cudaJitMaxThreadsPerBlock = 32,
    cudaJitOverrideDirectiveValues = 33,
}

#[allow(dead_code)]
#[repr(u32)]
#[derive(Debug, Copy, Clone)]
pub enum cudaLibraryOption {
    cudaLibraryHostUniversalFunctionAndDataTable = 0,
    cudaLibraryBinaryIsPreserved = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUkern_st {
    _unused: [u8; 0],
}
pub type cudaKernel_t = *mut CUkern_st;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct dim3 {
    pub x: c_uint,
    pub y: c_uint,
    pub z: c_uint,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}

pub type cudaStream_t = *mut CUstream_st;

unsafe extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaGetErrorString(error: cudaError_t) -> *const i8;

    pub fn cudaLibraryLoadData(
        library: *mut cudaLibrary_t,
        code: *const c_void,
        jitOptions: *mut cudaJitOption,
        jitOptionsValues: *mut *mut c_void,
        numJitOptions: c_uint,
        libraryOptions: *mut cudaLibraryOption,
        libraryOptionValues: *mut *mut c_void,
        numLibraryOptions: c_uint,
    ) -> cudaError_t;

    pub fn cudaLibraryGetKernel(
        pKernel: *mut cudaKernel_t,
        library: cudaLibrary_t,
        name: *const c_char,
    ) -> cudaError_t;

    pub fn cudaLaunchKernel(
        func: *const c_void,
        gridDim: dim3,
        blockDim: dim3,
        args: *mut *mut c_void,
        sharedMem: usize,
        stream: cudaStream_t,
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
