// Locate ARPACK-NG via pkg-config, verify ABI compatibility, and emit
// link directives.
//
// Bindings are pre-generated and committed at `src/bindings.rs`, so this
// build script only handles linkage and an integer-ABI sanity check.
// To regenerate bindings, run `regen-bindings.sh` at the workspace root.

use std::fs;
use std::path::PathBuf;

fn main() {
    // Reject non-64-bit-pointer-width targets up front so the user
    // sees the supported-targets message before pkg-config / linker
    // diagnostics. The mirrored `compile_error!` in lib.rs catches
    // configurations where this build script is skipped (e.g. docs
    // builds).
    let pointer_width = std::env::var("CARGO_CFG_TARGET_POINTER_WIDTH")
        .expect("CARGO_CFG_TARGET_POINTER_WIDTH must be set by cargo");
    if pointer_width != "64" {
        panic!(
            "arpack-sys is supported only on 64-bit-pointer-width targets \
             (got target_pointer_width = {pointer_width})"
        );
    }

    // Track DOCS_RS regardless of which branch this run takes, so a
    // target directory shared between docs.rs-style (`DOCS_RS=1`,
    // skip-link) and normal (probe + link) builds invalidates the
    // cached build-script output when the env var flips.
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    // docs.rs builds documentation in a sandbox that does not have
    // ARPACK-NG installed and never links the resulting artifact, so
    // the pkg-config probe and the ABI check it feeds would only abort
    // the build for no benefit. Bindings are pre-generated and
    // committed, so skipping here still lets rustdoc render the public
    // surface.
    if std::env::var_os("DOCS_RS").is_some() {
        return;
    }

    let lib = pkg_config::Config::new()
        .atleast_version("3.8.0")
        .probe("arpack")
        .expect("pkg-config could not locate ARPACK-NG (>= 3.8.0). Install it (e.g. `brew install arpack`) and ensure its .pc file is on PKG_CONFIG_PATH.");

    verify_default_integer_abi(&lib.include_paths);
}

// The committed bindings target ARPACK-NG's default 32-bit integer ABI
// (`a_int = int`). A library built with `INTERFACE64=1` exposes the same
// symbol names but with 64-bit `a_int`, which would corrupt memory when
// the wrapper passes 32-bit ints / pointers. Read `arpackdef.h` and bail
// out at build time rather than letting the mismatch surface at runtime.
fn verify_default_integer_abi(include_paths: &[PathBuf]) {
    let mut header = None;
    for dir in include_paths {
        let candidate = dir.join("arpackdef.h");
        if candidate.exists() {
            header = Some(candidate);
            break;
        }
    }
    let Some(header) = header else {
        panic!(
            "could not find arpackdef.h in pkg-config include paths ({:?}); cannot verify ARPACK integer ABI",
            include_paths
        );
    };
    println!("cargo:rerun-if-changed={}", header.display());

    let content = fs::read_to_string(&header)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", header.display()));
    // Walk every `#define` line — the first one in the header is the
    // include guard, not the macro we want. Tolerate whitespace
    // between `#` and `define` and strip trailing `//` or `/* */`
    // comments and surrounding parentheses from the macro value so
    // distro-patched headers with different formatting still parse.
    let raw_value: Option<String> = content
        .lines()
        .map(str::trim)
        .filter_map(|l| l.strip_prefix('#').map(str::trim_start))
        .filter_map(|l| l.strip_prefix("define"))
        .find_map(|rest| {
            let mut tokens = rest.split_whitespace();
            match tokens.next() {
                Some("INTERFACE64") => tokens.next().map(str::to_owned),
                _ => None,
            }
        });
    let value = raw_value.as_deref().map(strip_macro_decoration);

    match value {
        Some("0") => {}
        Some("1") => panic!(
            "ARPACK-NG at {} was built with INTERFACE64=1 (64-bit a_int). \
             The committed arpack-sys bindings target the default 32-bit ABI, \
             so using a 64-bit-ABI build would silently corrupt memory. \
             Rebuild ARPACK-NG with INTERFACE64=0 (the upstream default) \
             or open an issue if 64-bit-ABI support is needed.",
            header.display()
        ),
        Some(other) => panic!(
            "unrecognized INTERFACE64 value '{other}' in {}; expected 0 or 1",
            header.display()
        ),
        None => panic!(
            "could not locate `#define INTERFACE64` in {}; the installed ARPACK-NG \
             header may be too old or non-standard",
            header.display()
        ),
    }
}

/// Strip wrapping parentheses, trailing `//...` or `/* ... */`
/// comments, and surrounding whitespace from a `#define` value token.
/// Conservative — only handles forms that occur in real headers; it is
/// not a full C preprocessor.
fn strip_macro_decoration(raw: &str) -> &str {
    let mut s = raw.trim();
    if let Some(pos) = s.find("//") {
        s = s[..pos].trim();
    }
    if let Some(pos) = s.find("/*") {
        s = s[..pos].trim();
    }
    while let (Some(stripped_start), true) = (s.strip_prefix('('), s.ends_with(')')) {
        s = stripped_start
            .strip_suffix(')')
            .unwrap_or(stripped_start)
            .trim();
    }
    s
}
