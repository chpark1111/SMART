use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(smart_no_manifold_bridge)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/manifold_bridge.cpp");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("rust/smart-core must live under the SMART repo");
    let manifold_root = repo_root.join("smart/vendor/manifold");
    let manifold_lib_dir = manifold_root.join("build/src/manifold");
    let manifold_lib = manifold_lib_dir.join("libmanifold.a");
    if !manifold_lib.exists() {
        println!("cargo:rustc-cfg=smart_no_manifold_bridge");
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj_path = out_dir.join("smart_manifold_bridge.o");
    let compiler = env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let source = manifest_dir.join("src/manifold_bridge.cpp");

    let status = Command::new(compiler)
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-DNDEBUG")
        .arg("-fPIC")
        .arg("-c")
        .arg(&source)
        .arg("-I")
        .arg(manifold_root.join("src/manifold/include"))
        .arg("-I")
        .arg(manifold_root.join("src/utilities/include"))
        .arg("-I")
        .arg(manifold_root.join("src/polygon/include"))
        .arg("-I")
        .arg(manifold_root.join("src/collider/include"))
        .arg("-I")
        .arg(manifold_root.join("src/sdf/include"))
        .arg("-I")
        .arg(manifold_root.join("src/third_party/glm"))
        .arg("-o")
        .arg(&obj_path)
        .status();

    match status {
        Ok(status) if status.success() => {
            println!("cargo:rustc-link-arg={}", obj_path.display());
            println!(
                "cargo:rustc-link-search=native={}",
                manifold_lib_dir.display()
            );
            println!("cargo:rustc-link-lib=static=manifold");
            if cfg!(target_os = "macos") {
                println!("cargo:rustc-link-lib=c++");
            } else {
                println!("cargo:rustc-link-lib=stdc++");
            }
        }
        _ => {
            println!("cargo:rustc-cfg=smart_no_manifold_bridge");
        }
    }
}
