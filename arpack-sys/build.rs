// Locate ARPACK-NG via pkg-config and emit link directives.
//
// Bindings are pre-generated and committed at `src/bindings.rs`, so this
// build script only handles linkage. To regenerate bindings, run the
// `regen-bindings.sh` script at the workspace root.

fn main() {
    pkg_config::Config::new()
        .atleast_version("3.8.0")
        .probe("arpack")
        .expect("pkg-config could not locate ARPACK-NG (>= 3.8.0). Install it (e.g. `brew install arpack`) and ensure its .pc file is on PKG_CONFIG_PATH.");
}
