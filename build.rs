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
        // ggml-cpu is built with OpenMP enabled; since we link the static
        // archives directly, the GNU OpenMP runtime (libgomp) must be linked
        // explicitly to resolve the GOMP_*/omp_* symbols it references.
        println!("cargo:rustc-link-lib=dylib=gomp");
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
