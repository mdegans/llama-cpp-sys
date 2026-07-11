//! Link/bindgen smoke tests for the `mtmd` feature. No model required:
//! these prove libmtmd compiled, linked, and the bindings resolve at
//! runtime. Anything touching an mmproj lives downstream.
#![cfg(feature = "mtmd")]

use std::ffi::CStr;

use llama_cpp_sys_3::*;

#[test]
fn default_marker_is_nonempty() {
    let marker = unsafe { mtmd_default_marker() };
    assert!(!marker.is_null());
    let marker = unsafe { CStr::from_ptr(marker) }
        .to_str()
        .expect("marker is valid UTF-8");
    assert!(!marker.is_empty());
    println!("mtmd_default_marker: {marker}");
}

#[test]
fn context_params_default_is_sane() {
    let params = unsafe { mtmd_context_params_default() };
    assert!(!params.media_marker.is_null());
    assert!(params.n_threads > 0);
    assert!(params.batch_max_tokens > 0);
}
