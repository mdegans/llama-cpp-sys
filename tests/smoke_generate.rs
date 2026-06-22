//! Minimal generation smoke test.
//!
//! This is intentionally tiny: it loads a model, decodes a short prompt, and
//! samples a handful of tokens with a greedy sampler. It does *not* check
//! coherence -- it only proves the build links correctly and that the
//! load -> tokenize -> decode -> sample loop is not completely broken.
//! Coherence is covered downstream in the higher-level wrapper crate.
//!
//! The model path is taken from the `LLAMA_TEST_MODEL` environment variable
//! (CI downloads a tiny GGUF and points this at it). If the variable is unset
//! and the default path is missing, the test skips rather than fails so that a
//! plain `cargo test` without a model does not error out.

use core::slice;
use llama_cpp_sys_3::*;
use std::{
    ffi::{CStr, CString},
    path::Path,
};

const BATCH_TOKENS: usize = 512;
/// Number of tokens to generate. Kept tiny so this runs in milliseconds on CPU.
const N_GEN: usize = 3;

#[test]
pub fn smoke_generate() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("LLAMA_TEST_MODEL").unwrap_or_else(|_| "models/model.gguf".to_string());
    if !Path::new(&model_path).exists() {
        eprintln!("skipping smoke_generate: model not found at `{model_path}` (set LLAMA_TEST_MODEL)");
        return Ok(());
    }

    // Global init
    unsafe {
        llama_backend_init();
        llama_numa_init(ggml_numa_strategy_GGML_NUMA_STRATEGY_DISABLED);
    }

    // Load model
    let model_params = unsafe { llama_model_default_params() };
    let model_name = CString::new(model_path.as_str())?;
    let model = unsafe { llama_model_load_from_file(model_name.as_ptr(), model_params) };
    assert!(!model.is_null(), "unable to load model: {model_path}");

    let vocab = unsafe { llama_model_get_vocab(model) };

    // Context
    let mut ctx_params = unsafe { llama_context_default_params() };
    ctx_params.n_ctx = 256;
    ctx_params.n_threads = 2;
    ctx_params.n_threads_batch = 2;
    let ctx = unsafe { llama_new_context_with_model(model, ctx_params) };
    assert!(!ctx.is_null(), "unable to create context");

    // Greedy sampler
    let sparams = unsafe { llama_sampler_chain_default_params() };
    let smpl = unsafe { llama_sampler_chain_init(sparams) };
    unsafe { llama_sampler_chain_add(smpl, llama_sampler_init_greedy()) };

    // Tokenize a short prompt
    let tokens_list = tokenize(vocab, "Once upon a time", true);
    assert!(!tokens_list.is_empty(), "prompt tokenized to zero tokens");

    // Evaluate the prompt
    let mut batch = unsafe { llama_batch_init(BATCH_TOKENS as i32, 0, 1) };
    for (pos, &token) in tokens_list.iter().enumerate() {
        llama_batch_add(&mut batch, token, pos, false);
    }
    // Request logits for the last prompt token only.
    {
        let logits = unsafe { slice::from_raw_parts_mut(batch.logits, BATCH_TOKENS) };
        logits[usize::try_from(batch.n_tokens - 1)?] = 1;
    }
    assert_eq!(unsafe { llama_decode(ctx, batch) }, 0, "initial llama_decode failed");

    // Generate N_GEN tokens.
    let mut n_cur = batch.n_tokens;
    let mut produced = 0usize;
    let eos = unsafe { llama_vocab_eos(vocab) };
    for _ in 0..N_GEN {
        let new_token_id = unsafe { llama_sampler_sample(smpl, ctx, batch.n_tokens - 1) };
        if new_token_id == eos {
            break;
        }
        // Sanity check: every sampled token should round-trip through the vocab.
        let _piece = token_to_piece(new_token_id, vocab);
        produced += 1;

        batch.n_tokens = 0;
        llama_batch_add(&mut batch, new_token_id, n_cur as usize, true);
        n_cur += 1;
        assert_eq!(unsafe { llama_decode(ctx, batch) }, 0, "llama_decode failed during generation");
    }

    assert!(produced > 0, "model produced no tokens before EOS");

    // Cleanup
    unsafe {
        llama_sampler_free(smpl);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
    }

    Ok(())
}

/// Adapted from `llama.cpp/common/common.cpp`.
fn token_to_piece(token: llama_token, vocab: *const llama_vocab) -> String {
    let mut buf = [0u8; 64];
    let n = unsafe { llama_token_to_piece(vocab, token, buf.as_mut_ptr() as *mut i8, buf.len() as i32, 0, false) };
    assert!(n >= 0, "token `{token}` piece longer than 64 chars");
    CStr::from_bytes_until_nul(&buf).unwrap().to_string_lossy().into_owned()
}

/// Adapted from `llama.cpp/common/common.cpp`.
fn llama_batch_add(batch: &mut llama_batch, token: llama_token, pos: usize, logits: bool) {
    assert!(batch.n_tokens <= BATCH_TOKENS as i32);
    let n: usize = batch.n_tokens as usize;
    unsafe {
        slice::from_raw_parts_mut(batch.token, BATCH_TOKENS)[n] = token;
        slice::from_raw_parts_mut(batch.pos, BATCH_TOKENS)[n] = pos as i32;
        slice::from_raw_parts_mut(batch.n_seq_id, BATCH_TOKENS)[n] = 1;
        let ids = slice::from_raw_parts_mut(batch.seq_id, BATCH_TOKENS)[n];
        slice::from_raw_parts_mut(ids, 1)[0] = 0;
        slice::from_raw_parts_mut(batch.logits, BATCH_TOKENS)[n] = logits as i8;
    }
    batch.n_tokens += 1;
}

/// Adapted from `llama.cpp/common/common.cpp`.
fn tokenize(vocab: *const llama_vocab, text: &str, add_bos: bool) -> Vec<llama_token> {
    let mut n_tokens: i32 = (text.len() + if add_bos { 1 } else { 0 }) as i32;
    let mut result = vec![0; n_tokens as usize];
    n_tokens = unsafe {
        llama_tokenize(
            vocab,
            text.as_ptr() as *const i8,
            text.len() as i32,
            result.as_mut_ptr(),
            result.len() as i32,
            add_bos,
            false,
        )
    };
    if n_tokens < 0 {
        result.resize((-n_tokens) as usize, 0);
        let check = unsafe {
            llama_tokenize(
                vocab,
                text.as_ptr() as *const i8,
                text.len() as i32,
                result.as_mut_ptr(),
                result.len() as i32,
                add_bos,
                false,
            )
        };
        assert_eq!(check, -n_tokens);
    } else {
        result.resize(n_tokens as usize, 0);
    }
    result
}
