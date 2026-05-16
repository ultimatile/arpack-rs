//! `Which` selector for ARPACK driver entry points.
//!
//! ARPACK's `*aupd_c` family takes a 2-character `which` parameter
//! that specifies which Ritz values to retain (e.g. smallest
//! algebraic, largest magnitude). The accepted set differs per
//! driver family — symmetric Lanczos accepts `{SA, LA, SM, LM, BE}`,
//! complex Arnoldi accepts `{SR, LR, SI, LI, SM, LM}` — so passing
//! an unsupported variant to the wrong driver makes ARPACK return
//! `info = -5`.
//!
//! The wrapper rejects the incompatible combinations up front via
//! [`Which::accepted_by_symmetric`] / [`Which::accepted_by_complex_arnoldi`]
//! so callers see a self-describing [`crate::Error::InvalidParam`]
//! rather than a late `-5` from inside the reverse-communication
//! loop.
//!
//! `BE` (both ends, symmetric-only, requires `nev >= 2`) is not
//! modelled here — adding it would need an enum design that knows
//! about driver family and `nev` jointly. Deferred until a concrete
//! need shows up.

use std::ffi::CStr;

/// Ritz value selector passed to ARPACK's `*aupd_c` `which`
/// parameter.
///
/// Each variant maps to a 2-character ASCII tag in the ARPACK
/// Users' Guide §3.3. Not every variant is accepted by every
/// driver family:
///
/// | Variant              | Tag  | Symmetric (`*saupd`) | Complex (`*naupd`) |
/// |----------------------|------|----------------------|--------------------|
/// | `SmallestAlgebraic`  | `SA` | accepted             | rejected           |
/// | `LargestAlgebraic`   | `LA` | accepted             | rejected           |
/// | `SmallestRealPart`   | `SR` | rejected             | accepted           |
/// | `LargestRealPart`    | `LR` | rejected             | accepted           |
/// | `SmallestImagPart`   | `SI` | rejected             | accepted           |
/// | `LargestImagPart`    | `LI` | rejected             | accepted           |
/// | `SmallestMagnitude`  | `SM` | accepted             | accepted           |
/// | `LargestMagnitude`   | `LM` | accepted             | accepted           |
///
/// The wrapper rejects mismatched combinations up front, so callers
/// see [`crate::Error::InvalidParam`] instead of a late ARPACK
/// `info = -5`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Which {
    SmallestAlgebraic,
    LargestAlgebraic,
    SmallestRealPart,
    LargestRealPart,
    SmallestImagPart,
    LargestImagPart,
    SmallestMagnitude,
    LargestMagnitude,
}

impl Which {
    /// ARPACK's 2-character C string for this selector.
    pub(crate) fn as_c_str(self) -> &'static CStr {
        match self {
            Which::SmallestAlgebraic => c"SA",
            Which::LargestAlgebraic => c"LA",
            Which::SmallestRealPart => c"SR",
            Which::LargestRealPart => c"LR",
            Which::SmallestImagPart => c"SI",
            Which::LargestImagPart => c"LI",
            Which::SmallestMagnitude => c"SM",
            Which::LargestMagnitude => c"LM",
        }
    }

    /// True if the symmetric Lanczos drivers (`{s,d}{sa,se}upd_c`)
    /// accept this selector.
    pub(crate) fn accepted_by_symmetric(self) -> bool {
        matches!(
            self,
            Which::SmallestAlgebraic
                | Which::LargestAlgebraic
                | Which::SmallestMagnitude
                | Which::LargestMagnitude
        )
    }

    /// True if the complex Arnoldi drivers (`{c,z}{na,ne}upd_c`)
    /// accept this selector.
    pub(crate) fn accepted_by_complex_arnoldi(self) -> bool {
        matches!(
            self,
            Which::SmallestRealPart
                | Which::LargestRealPart
                | Which::SmallestImagPart
                | Which::LargestImagPart
                | Which::SmallestMagnitude
                | Which::LargestMagnitude
        )
    }
}
