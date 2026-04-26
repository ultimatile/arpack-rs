/* bindgen entry point. ARPACK-NG installs its public C API at
 * `<arpack/arpack.h>` (and friends). We pull only the eigenvalue-driver
 * surface; debug/stat helpers can be added if a downstream caller
 * needs them. */

#include <stdint.h>
#include <arpack.h>
