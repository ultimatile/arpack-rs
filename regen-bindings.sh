#!/usr/bin/env bash
# Regenerate arpack-sys/src/bindings.rs from the locally installed ARPACK-NG.
#
# Requirements:
#   - bindgen CLI on PATH (`cargo install bindgen-cli`)
#   - pkg-config can find arpack (`pkg-config --modversion arpack`)
#
# The generated file is committed; downstream users do NOT need bindgen.

set -euo pipefail

cd "$(dirname "$0")"

CFLAGS=$(pkg-config --cflags arpack)

bindgen arpack-sys/wrapper.h \
    -o arpack-sys/src/bindings.rs \
    --allowlist-function '[sdcz](na|ne|sa|se)upd_c' \
    --allowlist-type 'a_(int|uint|fcomplex|dcomplex)' \
    --merge-extern-blocks \
    --formatter rustfmt \
    -- $CFLAGS

echo "regenerated: arpack-sys/src/bindings.rs"
