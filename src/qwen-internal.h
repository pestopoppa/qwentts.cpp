#pragma once
// Private entry for the bundled server; the public qt_init ABI stays unchanged.
#include "qwen.h"

#ifdef __cplusplus
extern "C" {
#endif
struct qt_context * qt_init_with_threads(const struct qt_init_params * params, int n_threads);
#ifdef __cplusplus
}
#endif
