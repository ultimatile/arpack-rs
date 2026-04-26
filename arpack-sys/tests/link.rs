// Confirms that the bindings actually resolve at link time. We do not
// drive a real eigenvalue problem here; just take the address of one
// of the C-callable wrappers so the linker has to find the symbol.
// Reading the address through `core::hint::black_box` keeps the
// optimizer from eliding the reference entirely.

use arpack_sys::dsaupd_c;
use core::hint::black_box;

#[test]
fn dsaupd_c_symbol_is_linkable() {
    let addr = dsaupd_c as *const () as usize;
    let _ = black_box(addr);
}
