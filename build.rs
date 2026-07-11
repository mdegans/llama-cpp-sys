use cmake::Config;
use std::env;
use std::path::PathBuf;

fn main() {
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
                eprintln!("llama-cpp-sys: building with OpenMP (lib{lib})");
            }
            None => {
                println!(
                    "cargo:warning=OpenMP runtime not found; building ggml without OpenMP. \
                     Set LLAMA_CPP_SYS_OMP_PATH to the directory containing the runtime \
                     (libomp.so/libgomp.so) to enable it."
                );
            }
        }
        runtime
    };

    // Everything both cmake passes agree on. The passes share one build tree,
    // so any define left out here would flip-flop the cmake cache between
    // reconfigures.
    let base_config = || {
        let mut config = Config::new("external/llama.cpp");
        config
            .generator("Ninja")
            .define("BUILD_SHARED_LIBS", "OFF")
            .define("LLAMA_BUILD_EXAMPLES", "OFF")
            .define("LLAMA_BUILD_SERVER", "OFF")
            .define("LLAMA_BUILD_TESTS", "OFF")
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

        #[cfg(target_os = "linux")]
        config.define(
            "GGML_OPENMP",
            if linux_openmp.is_some() { "ON" } else { "OFF" },
        );

        config
    };

    // Build the core libraries via the `install` target. Tools stay OFF here:
    // `install` depends on `all`, which would otherwise build every tool
    // executable.
    let mut config = base_config();
    config
        .build_target("install")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF");
    let dst = config.very_verbose(true).build();

    // mtmd: reconfigure the same build tree with tools enabled, but build
    // ONLY the `mtmd` library target — it links just ggml+llama (upstream
    // FATAL_ERRORs if it ever links llama-common), and no tool executables
    // get built this way.
    #[cfg(feature = "mtmd")]
    {
        let mut config = base_config();
        config
            .build_target("mtmd")
            .define("LLAMA_BUILD_COMMON", "ON")
            .define("LLAMA_BUILD_TOOLS", "ON")
            .define("MTMD_VIDEO", "OFF");
        config.very_verbose(true).build();
    }

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
    // mtmd is never installed; it stays in the build tree.
    #[cfg(feature = "mtmd")]
    {
        println!(
            "cargo:rustc-link-search=native={}/build/tools/mtmd",
            dst.display()
        );
        println!("cargo:rustc-link-lib=static=mtmd");
    }
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
    // ggml-cpu reads the registry (HKLM\...\CentralProcessor) to get the CPU
    // name on Windows, pulling in the Reg* APIs from advapi32. CMake links this
    // automatically for its own targets, but since we consume the static
    // archive directly we have to link the system lib ourselves.
    #[cfg(target_os = "windows")]
    println!("cargo:rustc-link-lib=dylib=advapi32");

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
        // cublas/cudart and the CUDA driver API live in the toolkit, which is
        // not on the linker's default search path. Derive the lib dirs from
        // CUDA_PATH (set by standard installs and CI toolkit actions), falling
        // back to CUDA_HOME and the conventional install location.
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        let cuda_root = PathBuf::from(
            env::var("CUDA_PATH")
                .or_else(|_| env::var("CUDA_HOME"))
                .unwrap_or_else(|_| "/usr/local/cuda".to_string()),
        );

        #[cfg(target_os = "windows")]
        println!("cargo:rustc-link-search=native={}", cuda_root.join("lib").join("x64").display());

        #[cfg(not(target_os = "windows"))]
        {
            // Cover both the lib64 layout and the targets/ layout used by some
            // distributions. The stubs dir provides the link-time libcuda so a
            // real GPU driver isn't required to build.
            for lib in [
                cuda_root.join("lib64"),
                cuda_root.join("lib64").join("stubs"),
                cuda_root.join("targets").join("x86_64-linux").join("lib"),
                cuda_root.join("targets").join("x86_64-linux").join("lib").join("stubs"),
            ] {
                println!("cargo:rustc-link-search=native={}", lib.display());
            }
        }

        println!("cargo:rustc-link-lib=static=ggml-cuda");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    // Rerun triggers
    println!("cargo:rerun-if-changed=external/llama.cpp/include/llama.h");
    println!("cargo:rerun-if-changed=external/llama.cpp/ggml/include/ggml.h");
    #[cfg(feature = "mtmd")]
    {
        println!("cargo:rerun-if-changed=external/llama.cpp/tools/mtmd/mtmd.h");
        println!("cargo:rerun-if-changed=external/llama.cpp/tools/mtmd/mtmd-helper.h");
    }

    // Generate bindings
    #[allow(unused_mut)]
    let mut builder = bindgen::Builder::default()
        .header("external/llama.cpp/include/llama.h")
        .clang_arg("-Iexternal/llama.cpp/ggml/include")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("gguf_.*");

    // The helper functions share the mtmd_ prefix, and we want them: the safe
    // layer differential-tests its own eval loop against them.
    #[cfg(feature = "mtmd")]
    {
        builder = builder
            .header("external/llama.cpp/tools/mtmd/mtmd.h")
            .header("external/llama.cpp/tools/mtmd/mtmd-helper.h")
            .clang_arg("-Iexternal/llama.cpp/include")
            .allowlist_function("mtmd_.*")
            .allowlist_type("mtmd_.*");
    }

    let bindings = builder
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
