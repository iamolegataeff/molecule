// build.rs — link BLAS when --features blas is enabled
fn main() {
    if std::env::var("CARGO_FEATURE_BLAS").is_ok() {
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        } else {
            println!("cargo:rustc-link-lib=openblas");
        }
    }
}
