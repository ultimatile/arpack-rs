// Confirms that the bindings actually resolve at link time. We do not
// drive a real eigenvalue problem here; just take the address of one
// of the C-callable wrappers so the linker has to find the symbol.

use arpack_sys::dsaupd_c;

#[test]
fn dsaupd_c_symbol_is_linkable() {
    let f: unsafe extern "C" fn(_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) =
        dsaupd_c;
    assert!(!(f as *const ()).is_null());
}
