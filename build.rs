use cmake::Config;
use std::env;
use std::path::PathBuf;

fn main() {
    // Configure
    let mut config = Config::new("external/llama.cpp");
    config
        .build_target("install")
        .generator("Ninja")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        // Upstream added a unified `app` binary that is not gated behind
        // LLAMA_BUILD_COMMON; disable it so we only build the libraries.
        .define("LLAMA_BUILD_APP", "OFF");

    #[cfg(target_os = "macos")]
    {
        config
            .define("GGML_METAL", "ON")
            .define("GGML_ACCELERATE", "ON")
            .define("GGML_METAL_EMBED_LIBRARY", "ON");
    }

    #[cfg(feature = "cuda")]
    config.define("GGML_CUDA", "ON");

    #[cfg(feature = "cuda_f16")]
    config.define("GGML_CUDA_FP16", "ON");

    #[cfg(feature = "native")]
    config.define("GGML_NATIVE", "ON");

    // Decide how to handle OpenMP on Linux. ggml-cpu links an OpenMP runtime
    // (libgomp for GCC, libomp for Clang), and because we link the static
    // archives directly we have to locate and link that runtime ourselves.
    // Keep this in lockstep with the cmake build: if we can find a runtime we
    // enable OpenMP and link it; if we can't, we disable OpenMP (matching
    // upstream's graceful fallback) and warn, rather than fail the link.
    #[cfg(target_os = "linux")]
    let linux_openmp = {
        println!("cargo:rerun-if-env-changed=LLAMA_CPP_SYS_OMP_PATH");
        println!("cargo:rerun-if-env-changed=CC");
        let runtime = linux_openmp_runtime();
        match &runtime {
            Some((lib, _)) => {
                config.define("GGML_OPENMP", "ON");
                eprintln!("llama-cpp-sys: building with OpenMP (lib{lib})");
            }
            None => {
                config.define("GGML_OPENMP", "OFF");
                println!(
                    "cargo:warning=OpenMP runtime not found; building ggml without OpenMP. \
                     Set LLAMA_CPP_SYS_OMP_PATH to the directory containing the runtime \
                     (libomp.so/libgomp.so) to enable it."
                );
            }
        }
        runtime
    };

    // Build
    let dst = config.very_verbose(true).build();

    // Link search paths - cmake install puts libs in lib/
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    // Some cmake configs put libs in build/ directly
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!(
        "cargo:rustc-link-search=native={}/build/src",
        dst.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/build/ggml/src",
        dst.display()
    );

    // Link llama and ggml libraries (order matters - dependents first)
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    // C++ standard library
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=dylib=c++");
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        if let Some((lib, search_dir)) = linux_openmp {
            if let Some(dir) = search_dir {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
    #[cfg(all(target_os = "windows", debug_assertions))]
    println!("cargo:rustc-link-lib=dylib=msvcrtd");
    #[cfg(all(target_os = "windows", not(debug_assertions)))]
    println!("cargo:rustc-link-lib=dylib=msvcrt");

    // macOS frameworks
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=static=ggml-metal");
        println!("cargo:rustc-link-lib=static=ggml-blas");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    }

    // CUDA libraries
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    // Rerun triggers
    println!("cargo:rerun-if-changed=external/llama.cpp/include/llama.h");
    println!("cargo:rerun-if-changed=external/llama.cpp/ggml/include/ggml.h");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("external/llama.cpp/include/llama.h")
        .clang_arg("-Iexternal/llama.cpp/ggml/include")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("gguf_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

/// Resolve the OpenMP runtime to link on Linux.
///
/// Returns `Some((link_name, search_dir))` where `link_name` is the library to
/// pass to `-l` and `search_dir` an optional directory to add to the linker
/// search path, or `None` if no runtime could be located (OpenMP is disabled).
///
/// In every case we verify that the runtime `.so` actually exists before
/// committing to link it; if it can't be found we return `None` so the caller
/// warns and falls back to ggml's own threadpool rather than failing the link.
/// A directory given via `LLAMA_CPP_SYS_OMP_PATH` takes precedence, but is
/// still checked - a wrong override disables OpenMP rather than auto-detecting
/// something else.
#[cfg(target_os = "linux")]
fn linux_openmp_runtime() -> Option<(&'static str, Option<PathBuf>)> {
    let compiler = cc::Build::new().get_compiler();
    // GCC links libgomp, Clang/LLVM links libomp.
    let (lib, soname) = if compiler.is_like_clang() {
        ("omp", "libomp.so")
    } else {
        ("gomp", "libgomp.so")
    };

    // 1. Explicit override wins, but must actually contain the runtime.
    if let Some(dir) = env::var_os("LLAMA_CPP_SYS_OMP_PATH").map(PathBuf::from) {
        return dir.join(soname).exists().then_some((lib, Some(dir)));
    }

    // 2. Ask the compiler where the runtime is. GCC resolves libgomp to an
    //    absolute path this way; Clang cannot locate libomp like this, so it
    //    falls through to the resource-dir derivation below.
    if let Some(path) = compiler_lib_path(&compiler, soname) {
        return Some((lib, path.parent().map(|p| p.to_path_buf())));
    }

    // 3. Clang fallback: libomp lives in clang's private lib dir (e.g.
    //    /usr/lib/llvm-NN/lib), which is not on the default linker path.
    if compiler.is_like_clang() {
        if let Some(dir) = clang_openmp_dir(&compiler) {
            if dir.join(soname).exists() {
                return Some((lib, Some(dir)));
            }
        }
    }

    // 4. Nothing found - caller disables OpenMP and warns.
    None
}

/// Ask the compiler for the absolute path of a library via `-print-file-name`.
///
/// Returns `Some(path)` only when the compiler resolves it to a real, absolute
/// file; a bare soname (the "not found" response) yields `None`.
#[cfg(target_os = "linux")]
fn compiler_lib_path(compiler: &cc::Tool, soname: &str) -> Option<PathBuf> {
    let out = std::process::Command::new(compiler.path())
        .arg(format!("-print-file-name={soname}"))
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let path = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    (path.is_absolute() && path.exists()).then_some(path)
}

/// Derive the directory containing Clang's `libomp.so` from its resource dir.
///
/// `clang -print-resource-dir` yields `<llvm>/lib/clang/<ver>`, and the runtime
/// lives at `<llvm>/lib`, i.e. two levels up.
#[cfg(target_os = "linux")]
fn clang_openmp_dir(compiler: &cc::Tool) -> Option<PathBuf> {
    let out = std::process::Command::new(compiler.path())
        .arg("-print-resource-dir")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let resource_dir = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    Some(resource_dir.parent()?.parent()?.to_path_buf())
}
