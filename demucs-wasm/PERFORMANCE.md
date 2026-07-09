# WASM Performance Notes

The WebAssembly build is inherently slower than the native CLI, but the gap
has several distinct causes with very different fixes. This document records
what was investigated, what is implemented, and what is intentionally left on
the table.

## Why native is faster

1. **Shader compilation target.** The native CLI uses Burn's `metal` /
   `vulkan` features, which compile CubeCL kernels straight to MSL or SPIR-V
   (`wgpu<msl>` / `wgpu<spirv>`). In the browser, kernels go through WGSL and
   the browser's own shader compiler (Tint/Naga), which supports fewer
   optimizations (e.g. no passthrough subgroup matrix ops, line size capped
   at 4 vs 8 on MSL). This is structural — nothing to fix on our side.

2. **CPU code generation.** Native builds use `-C target-cpu=native`
   (SSE/AVX/NEON in rustfft, autovectorized DSP loops). The wasm build
   previously compiled scalar-only.

3. **Autotune behavior.** CubeCL's autotuner cannot block on GPU readbacks in
   the browser, so tuning runs as detached async tasks and every kernel
   executes with the *default* variant until its tune result lands
   (`cubecl-runtime`'s `TuneCacheResult::Miss → index 0` path). Tune results
   are also never persisted on wasm (the cache is `std_io`-gated), so every
   page load re-tunes. The existing `warmup_model` pass mitigates this: it
   runs a full forward with the exact shapes real inference uses, which
   kicks off tuning while the user is still picking a file.

4. **Single-threaded CPU work.** GPU→CPU readback, iSTFT, and overlap-add all
   share one worker thread; there is no rayon on wasm without
   cross-origin-isolation headers.

## Implemented optimizations

### wasm SIMD (`+simd128`)

`.cargo/config.toml` now passes `-C target-feature=+simd128` for
`wasm32-unknown-unknown`, and `demucs-core` enables rustfft's `wasm_simd`
feature on wasm. Every browser with WebGPU also supports wasm SIMD, so this
is a safe baseline. It vectorizes:

- the forward STFT (2×) and inverse STFT (2 × n_stems per chunk) — ~3,400
  4096-point FFTs per chunk,
- polyphase resampling for non-44.1 kHz inputs (a 48 kHz 4-minute track costs
  billions of scalar MACs — this is one of the largest CPU-side wins),
- windowed overlap-add accumulation (LLVM autovectorization).

### GPU/CPU pipelining for chunked inference

For the single-model variants (`htdemucs`, `htdemucs_6s`), the chunk loop in
`demucs-core` now enqueues chunk *i+1*'s forward pass on the GPU immediately
after chunk *i*'s outputs are read back, then runs chunk *i*'s CPU
post-processing (CaC→complex, iSTFT, overlap-add) while the GPU crunches the
next chunk. Previously the GPU idled during all per-chunk CPU work
(~50–60 MB of readback post-processing per chunk).

The fine-tuned variant stays sequential: it runs up to four model forwards
per chunk, and pipelining would hold several full sets of forward outputs on
the GPU at once.

### Model instance caching (`demucs-wasm`)

Loading a model = parsing ~84 MB of safetensors + uploading every tensor to
the GPU — several seconds on wasm. Previously `warmup_model` loaded the model
and dropped it, then `separate` loaded it again, and every subsequent
separation re-loaded it. The wasm layer now caches the most recent
`Demucs` instance (keyed by model id, weight byte length, and stem
selection), so warmup → separate loads once, and repeat separations skip
loading entirely. `clear_model_cache()` is exported for JS to release the
GPU memory if needed.

### wasm-opt tuning

`wasm-pack` defaults to `wasm-opt -O`. The release profile now runs `-O3`
with the feature flags (`--enable-simd`, `--enable-bulk-memory`, …) needed to
process the instructions rustc emits.

### Experimental f16 inference (`--features f16`)

`make wasm-release-f16` builds the module with Burn's float element set to
`f16`, halving GPU memory traffic — most WGSL kernels in this workload are
bandwidth-bound, so this is a potential ~1.5–2× GPU-side win. It requires the
WebGPU `shader-f16` feature (recent Chrome on most GPUs; the module fails at
kernel compile time without it) and slightly changes output numerics, so it
is off by default and not wired into the shipped web app. Readbacks convert
back to f32 on the CPU (`TensorData::convert`, a no-op on the f32 build).

## Not implemented (and why)

- **wasm threads + rayon for CPU DSP** — needs COOP/COEP headers
  (cross-origin isolation), which GitHub Pages cannot set without a service
  worker shim, plus an atomics-enabled std rebuild. The pipelining change
  hides most of the CPU work behind GPU compute instead.
- **Persisting autotune results across page loads** — would need an upstream
  CubeCL change (e.g. an IndexedDB-backed tune cache on wasm).
- **Concurrent freq/time readbacks** — the two per-chunk readbacks happen
  back-to-back; joining them saves only a few ms once the pipeline overlaps
  everything else.
- **Raising `tasks_max`** — already tuned to 128 (vs the default 32) in
  `ensure_wgpu`.
