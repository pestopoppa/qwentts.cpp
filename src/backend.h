#pragma once
// backend.h: GGML backend initialization
//
// All modules use the same pattern: load all backends, pick best GPU,
// keep CPU as fallback. Each backend_init call returns a fresh backend
// pair with its own device context and memory pool, so independent
// qt_contexts never share allocator state and can run concurrently.
// Sharing within one pipeline (talker, predictor, codec) is done by
// passing the same BackendPair to each module.

#include "ggml-backend.h"
#include "qt-error.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>

struct BackendPair {
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    bool           has_gpu;
};

// Physical core count heuristic (logical / 2 for HT/SMT).
// Used for GGML CPU thread count: GEMM shares SIMD units across hyperthreads,
// so one thread per physical core is optimal.
//
// Override: QT_CPU_THREADS=<n> (or qwen-tts --threads <n>) sets the count explicitly. Unset or
// invalid keeps the heuristic above, so the default is unchanged.
static int backend_cpu_n_threads(void) {
    const char * env = std::getenv("QT_CPU_THREADS");
    if (env && *env) {
        int v = std::atoi(env);
        if (v > 0) {
            return v;
        }
    }
    int n = (int) std::thread::hardware_concurrency() / 2;
    return n > 0 ? n : 1;
}

// Persistent CPU threadpool, one per CPU backend.
//
// Stock ggml builds and frees a disposable threadpool (n-1 pthread_create + join) on EVERY
// ggml_backend_graph_compute when the backend has none attached, and qwentts computes thousands
// of small graphs per second of audio. We create ONE pool at backend init, attach it with
// ggml_backend_cpu_set_threadpool, reuse it for every compute and free it in backend_release.
//
//   QT_THREADPOOL=0        disable (stock behaviour, disposable pool per compute)
//   QT_THREADPOOL_POLL=<p> ggml_threadpool_params.poll. Idle workers busy-poll for work for
//                          1024*128*p _mm_pause rounds before sleeping on the condvar. Default 0:
//                          workers sleep immediately between computes, so an idle pool costs no
//                          CPU and does not fight co-running work for cores. Other fields:
//                          paused=false (workers start immediately), strict_cpu=false and an
//                          all-zero cpumask (no pinning; the caller's affinity/taskset is
//                          inherited), prio=normal.
//
// The pool is not thread-safe: one compute at a time per backend. That already holds (the
// qt_context serialises compute under gpu_mu and every context owns its own backends).
inline std::mutex & backend_pool_mu() {
    static std::mutex m;
    return m;
}
inline std::map<ggml_backend_t, ggml_threadpool_t> & backend_pools() {
    static std::map<ggml_backend_t, ggml_threadpool_t> m;
    return m;
}

static void backend_cpu_attach_threadpool(ggml_backend_t cpu, ggml_backend_reg_t reg, int n_threads) {
    const char * on = std::getenv("QT_THREADPOOL");
    if ((on && std::strcmp(on, "0") == 0) || n_threads < 2 || !reg) {
        return;
    }
    using pool_new_t  = ggml_threadpool_t (*)(struct ggml_threadpool_params *);
    using pool_free_t = void (*)(ggml_threadpool_t);
    using pool_set_t  = void (*)(ggml_backend_t, ggml_threadpool_t);
    auto new_fn  = (pool_new_t) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto free_fn = (pool_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
    auto set_fn  = (pool_set_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
    if (!new_fn || !free_fn || !set_fn) {
        return;  // older/other CPU backend: keep the disposable pool
    }
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    tpp.poll = 0;
    if (const char * p = std::getenv("QT_THREADPOOL_POLL")) {
        int v = std::atoi(p);
        if (v >= 0 && v <= 100) {
            tpp.poll = (uint32_t) v;
        }
    }
    ggml_threadpool_t tp = new_fn(&tpp);
    if (!tp) {
        return;
    }
    set_fn(cpu, tp);
    qt_log(QT_LOG_INFO, "[Load] CPU threadpool: persistent, %d threads, poll=%u", n_threads, (unsigned) tpp.poll);
    std::lock_guard<std::mutex> lk(backend_pool_mu());
    backend_pools()[cpu] = tp;
}

// Free the pool attached by backend_cpu_attach_threadpool, if any. Call AFTER ggml_backend_free
// of the CPU backend (the backend never owns or frees the pool).
static void backend_cpu_free_threadpool(ggml_backend_t cpu) {
    ggml_threadpool_t tp = NULL;
    {
        std::lock_guard<std::mutex> lk(backend_pool_mu());
        auto it = backend_pools().find(cpu);
        if (it == backend_pools().end()) {
            return;
        }
        tp = it->second;
        backend_pools().erase(it);
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;
    auto free_fn = reg ? (void (*)(ggml_threadpool_t)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free") : nullptr;
    if (free_fn) {
        free_fn(tp);
    }
}

// Standalone CPU backend via Registry API (DL-safe, no ggml-cpu.h needed).
// Sets thread count via proc address since ggml_backend_cpu_device_init_backend
// ignores its params string and always defaults to GGML_DEFAULT_N_THREADS (4).
// Returns NULL on failure.
static ggml_backend_t cpu_backend_new(int n_threads) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t     cpu     = NULL;
    if (cpu_dev) {
        cpu = ggml_backend_dev_init(cpu_dev, NULL);
    }
    if (!cpu) {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
    if (!cpu) {
        return NULL;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;
    if (reg) {
        auto set_fn =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_fn) {
            set_fn(cpu, n_threads);
        }
        backend_cpu_attach_threadpool(cpu, reg, n_threads);
    }
    return cpu;
}

// Initialize backends: load all available (CUDA, Metal, Vulkan...),
// pick the best one, keep CPU as fallback.
// label: log prefix, e.g. "DiT", "VAE", "LM"
// Each call returns a fresh backend pair with its own memory pool.
// Returns a BackendPair with .backend == NULL when initialisation fails;
// the caller must check this before passing it to any pipeline_*_load.
// Collapse exact consecutive duplicate ggml log lines and report the total
// count when the run ends (tames the CUDA graph capture "reused" flood).
static void qt_ggml_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    static char last[256] = { 0 };
    static int  count     = 0;

    if (count > 0 && strcmp(text, last) == 0) {
        count++;
        return;
    }

    if (count > 1) {
        fprintf(stderr, "[Dedup] Previous line repeated %d times total\n", count);
    }

    fputs(text, stderr);
    strncpy(last, text, sizeof(last) - 1);
    last[sizeof(last) - 1] = 0;
    count                  = 1;
    fflush(stderr);
}

static BackendPair backend_init(const char * label) {
    // Magic static: log callback install and dynamic backend loading
    // happen exactly once, safe under concurrent qt_init calls.
    static const bool loaded = [] {
        ggml_log_set(qt_ggml_log, nullptr);
        ggml_backend_load_all();
        return true;
    }();
    (void) loaded;

    BackendPair bp = {};

    // GGML_BACKEND env var: force a specific device instead of auto-best.
    // Device names: CUDA0, Vulkan0, CPU, BLAS (see ggml_backend_dev_name).
    const char * force_backend = std::getenv("GGML_BACKEND");
    if (force_backend) {
        bp.backend = ggml_backend_init_by_name(force_backend, nullptr);
        if (!bp.backend) {
            // Assemble the device list inline so the log callback gets one
            // self-contained line instead of three. The available list can
            // grow with each backend that registers, so a std::string here
            // keeps the formatting allocation-free for the common case.
            std::string msg = "[Load] GGML_BACKEND=";
            msg += force_backend;
            msg += " not found. Available:";
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                msg += ' ';
                msg += ggml_backend_dev_name(ggml_backend_dev_get(i));
            }
            qt_log(QT_LOG_ERROR, "%s", msg.c_str());
            return BackendPair{};
        }
    } else {
        bp.backend = ggml_backend_init_best();
    }
    if (!bp.backend) {
        qt_log(QT_LOG_ERROR, "[Load] no backend available");
        return BackendPair{};
    }
    bool best_is_cpu = (strcmp(ggml_backend_name(bp.backend), "CPU") == 0);
    int  n_threads   = backend_cpu_n_threads();
    if (best_is_cpu) {
        ggml_backend_free(bp.backend);
        bp.backend     = cpu_backend_new(n_threads);
        bp.cpu_backend = bp.backend;
    } else {
        bp.cpu_backend = cpu_backend_new(n_threads);
    }
    if (!bp.cpu_backend) {
        qt_log(QT_LOG_ERROR, "[Load] failed to init CPU backend");
        if (bp.backend && bp.backend != bp.cpu_backend) {
            ggml_backend_free(bp.backend);
        }
        return BackendPair{};
    }
    bp.has_gpu = !best_is_cpu;
    qt_log(QT_LOG_INFO, "[Load] %s backend: %s (CPU threads: %d)", label, ggml_backend_name(bp.backend), n_threads);
    return bp;
}

// Free a backend pair returned by backend_init.
static void backend_release(ggml_backend_t backend, ggml_backend_t cpu_backend) {
    if (backend && backend != cpu_backend) {
        ggml_backend_free(backend);
    }
    if (cpu_backend) {
        ggml_backend_free(cpu_backend);
        backend_cpu_free_threadpool(cpu_backend);
    }
}

// Create a scheduler from a backend pair.
// max_nodes: graph size hint (4096 for small models, 8192 for large)
// When a GPU is present, use its host buffer type for the CPU backend.
// Pinned memory lets the scheduler keep more ops on GPU instead of
// falling back to CPU with plain malloc.
static ggml_backend_sched_t backend_sched_new(BackendPair bp, int max_nodes) {
    ggml_backend_t             backends[2] = { bp.backend, bp.cpu_backend };
    ggml_backend_buffer_type_t bufts[2]    = { NULL, NULL };
    int                        n           = (bp.backend == bp.cpu_backend) ? 1 : 2;

    bufts[0] = ggml_backend_get_default_buffer_type(bp.backend);
    if (n == 2) {
        ggml_backend_dev_t         gpu_dev   = ggml_backend_get_device(bp.backend);
        ggml_backend_buffer_type_t host_buft = gpu_dev ? ggml_backend_dev_host_buffer_type(gpu_dev) : NULL;
        bufts[1] = host_buft ? host_buft : ggml_backend_get_default_buffer_type(bp.cpu_backend);
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, n, max_nodes, false, true);
    if (!sched) {
        qt_log(QT_LOG_ERROR, "[Load] failed to create scheduler");
        return nullptr;
    }
    return sched;
}
