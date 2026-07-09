use std::borrow::Cow;

use crate::dsp::cac::{cac_data_to_complex, stft_to_cac};
use crate::dsp::resample::resample_channel;
use crate::dsp::stft::Stft;
use crate::listener::{ForwardEvent, ForwardListener, NoOpListener};
use crate::model::{
    htdemucs::HTDemucs,
    metadata::{ModelInfo, StemId, HTDEMUCS, HTDEMUCS_6S, HTDEMUCS_FT},
};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
pub mod dsp;
pub mod error;
pub mod listener;
pub mod model;
pub mod provider;
pub mod weights;

pub use error::{DemucsError, Result};

pub struct Demucs<B: Backend> {
    opts: ModelOptions,
    models: Vec<HTDemucs<B>>,
    device: B::Device,
}

impl<B: Backend> Demucs<B> {
    pub fn from_bytes(opts: ModelOptions, bytes: &[u8], device: B::Device) -> Result<Self> {
        let info = opts.model_info();
        let models = weights::load::load_model(bytes, info, &device)?;
        Ok(Self {
            opts,
            models,
            device,
        })
    }

    /// Run the full inference pipeline with dummy audio to pre-compile all
    /// GPU shaders and resolve CubeCL autotune for every matmul shape.
    ///
    /// Uses a synthetic sine wave (not zeros) so autotune sees representative
    /// value distributions and selects the same kernel variants as real audio.
    pub async fn warmup(&self) {
        let dummy: Vec<f32> = (0..TRAINING_LENGTH)
            .map(|i| (i as f32 * 0.1).sin() * 0.5)
            .collect();
        let info = self.opts.model_info();
        let _ = self
            .separate_single_segment(&dummy, &dummy, TRAINING_LENGTH, info, &mut NoOpListener)
            .await;
    }

    pub async fn separate(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<Stem>> {
        self.separate_with_listener(left_channel, right_channel, sample_rate, &mut NoOpListener)
            .await
    }

    pub async fn separate_with_listener(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        sample_rate: u32,
        listener: &mut impl ForwardListener,
    ) -> Result<Vec<Stem>> {
        let info = self.opts.model_info();

        // ── 0. Resample to 44100 Hz if needed ────────────────────────────────
        let needs_resample = sample_rate != SAMPLE_RATE as u32;
        let (left_in, right_in): (Cow<[f32]>, Cow<[f32]>) = if needs_resample {
            let l = resample_channel(left_channel, sample_rate, SAMPLE_RATE as u32)
                .map_err(DemucsError::Dsp)?;
            let r = resample_channel(right_channel, sample_rate, SAMPLE_RATE as u32)
                .map_err(DemucsError::Dsp)?;
            (Cow::Owned(l), Cow::Owned(r))
        } else {
            (Cow::Borrowed(left_channel), Cow::Borrowed(right_channel))
        };
        let left_channel = &*left_in;
        let right_channel = &*right_in;
        let n_samples = left_channel.len();

        // ── 1. Short audio fast path (≤ TRAINING_LENGTH) ────────────────────
        let mut stems = if n_samples <= TRAINING_LENGTH {
            self.separate_single_segment(left_channel, right_channel, n_samples, info, listener)
                .await?
        } else {
            // ── 2. Chunked inference for long audio ─────────────────────────
            match &self.opts {
                ModelOptions::FourStem | ModelOptions::SixStem => {
                    self.separate_chunked_pipelined(left_channel, right_channel, info, listener)
                        .await?
                }
                ModelOptions::FineTuned(_) => {
                    self.separate_chunked_sequential(left_channel, right_channel, info, listener)
                        .await?
                }
            }
        };

        // ── 3. Resample outputs back to original rate if needed ──────────────
        if needs_resample {
            for stem in &mut stems {
                stem.left = resample_channel(&stem.left, SAMPLE_RATE as u32, sample_rate)
                    .map_err(DemucsError::Dsp)?;
                stem.right = resample_channel(&stem.right, SAMPLE_RATE as u32, sample_rate)
                    .map_err(DemucsError::Dsp)?;
            }
        }

        Ok(stems)
    }

    /// Chunked inference for single-model variants with GPU/CPU pipelining.
    ///
    /// The forward pass for chunk i+1 is enqueued on the GPU right after chunk
    /// i's outputs are read back, so the GPU crunches the next chunk while the
    /// CPU runs the iSTFTs and overlap-add for the current one.
    async fn separate_chunked_pipelined(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        info: &'static ModelInfo,
        listener: &mut impl ForwardListener,
    ) -> Result<Vec<Stem>> {
        let n_samples = left_channel.len();
        let segment = TRAINING_LENGTH;
        let stride = segment * 3 / 4; // 75% of segment = 25% overlap
        let num_chunks = n_samples.saturating_sub(segment).div_ceil(stride) + 1;
        let n_stems = info.stems.len();

        let bounds = |idx: usize| {
            let start = idx * stride;
            let end = (start + segment).min(n_samples);
            (start, end)
        };

        let mut acc = ChunkAccumulator::new(n_stems, n_samples);

        listener.on_event(ForwardEvent::ChunkStarted {
            index: 0,
            total: num_chunks,
        });
        let (s0, e0) = bounds(0);
        let first = self.forward_segment(
            0,
            &left_channel[s0..e0],
            &right_channel[s0..e0],
            e0 - s0,
            listener,
        )?;
        let mut pending = Some((first, 0usize));

        while let Some((seg, chunk_idx)) = pending.take() {
            let (start, end) = bounds(chunk_idx);
            let chunk_len = end - start;

            // Wait for this chunk's GPU results (2 bulk readbacks)
            let data = Self::read_segment(seg, n_stems).await?;

            // Enqueue the next chunk's forward pass before doing CPU work so
            // the GPU stays busy during the iSTFT + overlap-add below.
            let next = if chunk_idx + 1 < num_chunks {
                let (s, e) = bounds(chunk_idx + 1);
                let seg = self.forward_segment(
                    0,
                    &left_channel[s..e],
                    &right_channel[s..e],
                    e - s,
                    listener,
                )?;
                Some((seg, chunk_idx + 1))
            } else {
                None
            };

            let chunk_stems = Self::stems_from_segment(&data, info)?;
            for i in 0..chunk_stems.len() {
                listener.on_event(ForwardEvent::StemDone {
                    index: i,
                    total: n_stems,
                });
            }
            acc.add(info, &chunk_stems, start, chunk_len);

            listener.on_event(ForwardEvent::ChunkDone {
                index: chunk_idx,
                total: num_chunks,
            });
            if listener.is_cancelled() {
                return Err(DemucsError::Cancelled);
            }
            if next.is_some() {
                listener.on_event(ForwardEvent::ChunkStarted {
                    index: chunk_idx + 1,
                    total: num_chunks,
                });
            }
            pending = next;
        }

        Ok(acc.finalize(info))
    }

    /// Chunked inference for the fine-tuned variant, one segment at a time.
    ///
    /// Sequential on purpose: this variant runs up to four model forwards per
    /// chunk, so pipelining the next chunk behind the current one would hold
    /// several full sets of forward outputs on the GPU at once.
    async fn separate_chunked_sequential(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        info: &'static ModelInfo,
        listener: &mut impl ForwardListener,
    ) -> Result<Vec<Stem>> {
        let n_samples = left_channel.len();
        let segment = TRAINING_LENGTH;
        let stride = segment * 3 / 4; // 75% of segment = 25% overlap
        let num_chunks = n_samples.saturating_sub(segment).div_ceil(stride) + 1;

        let mut acc = ChunkAccumulator::new(info.stems.len(), n_samples);

        for chunk_idx in 0..num_chunks {
            listener.on_event(ForwardEvent::ChunkStarted {
                index: chunk_idx,
                total: num_chunks,
            });

            let start = chunk_idx * stride;
            let end = (start + segment).min(n_samples);
            let chunk_len = end - start;

            let chunk_stems = self
                .separate_single_segment(
                    &left_channel[start..end],
                    &right_channel[start..end],
                    chunk_len,
                    info,
                    listener,
                )
                .await?;

            acc.add(info, &chunk_stems, start, chunk_len);

            listener.on_event(ForwardEvent::ChunkDone {
                index: chunk_idx,
                total: num_chunks,
            });

            if listener.is_cancelled() {
                return Err(DemucsError::Cancelled);
            }
        }

        Ok(acc.finalize(info))
    }

    /// Run one model's forward pass for a segment: pad, STFT, build input
    /// tensors, and enqueue the GPU forward. Returns without waiting on the
    /// GPU — outputs stay on-device until [`Self::read_segment`] is awaited.
    fn forward_segment(
        &self,
        model_idx: usize,
        left_channel: &[f32],
        right_channel: &[f32],
        n_samples: usize,
        listener: &mut impl ForwardListener,
    ) -> Result<SegmentForward<B>> {
        let device = &self.device;

        // Pad to TRAINING_LENGTH
        let padded_len = TRAINING_LENGTH;
        let mut left_padded = vec![0.0f32; padded_len];
        let mut right_padded = vec![0.0f32; padded_len];
        left_padded[..n_samples].copy_from_slice(left_channel);
        right_padded[..n_samples].copy_from_slice(right_channel);

        let mut stft = Stft::new(N_FFT, HOP_LENGTH);

        // STFT both channels
        let left_spec = stft.forward(&left_padded)?;
        let right_spec = stft.forward(&right_padded)?;
        let bins = N_FFT / 2;
        let n_frames = left_spec.len() / bins;

        // CaC: each [2, F, T], stack to [4, F, T], add batch → [1, 4, F, T]
        let left_cac = stft_to_cac::<B>(&left_spec, N_FFT, device);
        let right_cac = stft_to_cac::<B>(&right_spec, N_FFT, device);
        let freq = Tensor::cat(vec![left_cac, right_cac], 0).unsqueeze_dim::<4>(0);

        let time = build_time_tensor::<B>(&left_padded, &right_padded, padded_len, device);

        let (freq_out, time_out) =
            self.models[model_idx].forward_with_listener(freq, time, listener)?;

        Ok(SegmentForward {
            freq_out,
            time_out,
            n_frames,
            n_samples,
        })
    }

    /// Read a segment's forward outputs back from the CPU in 2 bulk transfers
    /// (instead of 3 per stem), consuming the on-device tensors.
    async fn read_segment(seg: SegmentForward<B>, n_sources: usize) -> Result<SegmentData> {
        let freq_bins = N_FFT / 2;
        let SegmentForward {
            freq_out,
            time_out,
            n_frames,
            n_samples,
        } = seg;

        // GPU: trim time dimensions (no sync, stays on GPU)
        let freq_trimmed = freq_out
            .squeeze_dim::<3>(0) // [n_sources*4, F, padded_T]
            .narrow(2, 0, n_frames); // [n_sources*4, F, n_frames]

        let time_trimmed = time_out
            .squeeze_dim::<2>(0) // [n_sources*2, padded_T]
            .narrow(1, 0, n_samples); // [n_sources*2, n_samples]

        // convert::<f32> is a no-op on f32 backends; on reduced-precision
        // backends (e.g. the wasm f16 build) it widens back to f32.
        let freq_data: Vec<f32> = freq_trimmed
            .reshape([n_sources * 4 * freq_bins * n_frames])
            .to_data_async()
            .await
            .map_err(|e| DemucsError::Tensor(format!("freq bulk read failed: {e}")))?
            .convert::<f32>()
            .to_vec()
            .map_err(|e| DemucsError::Tensor(format!("freq extraction failed: {e}")))?;

        let time_data: Vec<f32> = time_trimmed
            .reshape([n_sources * 2 * n_samples])
            .to_data_async()
            .await
            .map_err(|e| DemucsError::Tensor(format!("time bulk read failed: {e}")))?
            .convert::<f32>()
            .to_vec()
            .map_err(|e| DemucsError::Tensor(format!("time extraction failed: {e}")))?;

        Ok(SegmentData {
            freq_data,
            time_data,
            n_frames,
            n_samples,
        })
    }

    /// CPU post-processing for a segment: split the bulk buffers per stem,
    /// CaC → complex → iSTFT, and combine the freq + time branches.
    fn stems_from_segment(data: &SegmentData, info: &ModelInfo) -> Result<Vec<Stem>> {
        let freq_bins = N_FFT / 2;
        let padded_len = TRAINING_LENGTH;
        let n_sources = info.stems.len();
        let n_frames = data.n_frames;
        let n_samples = data.n_samples;

        let mut stft = Stft::new(N_FFT, HOP_LENGTH);

        let cac_stride = 4 * freq_bins * n_frames; // floats per stem in freq
        let ch_stride = 2 * freq_bins * n_frames; // floats per stereo CaC pair
        let time_stride = 2 * n_samples; // floats per stem in time

        let mut stems = Vec::with_capacity(n_sources);
        for (i, &stem_id) in info.stems.iter().enumerate() {
            let freq_offset = i * cac_stride;
            let left_spec = cac_data_to_complex(
                &data.freq_data[freq_offset..freq_offset + ch_stride],
                freq_bins,
                n_frames,
            );
            let right_spec = cac_data_to_complex(
                &data.freq_data[freq_offset + ch_stride..freq_offset + cac_stride],
                freq_bins,
                n_frames,
            );

            let left_freq_wav = stft.inverse(&left_spec, padded_len)?;
            let right_freq_wav = stft.inverse(&right_spec, padded_len)?;

            let time_offset = i * time_stride;
            let left_time = &data.time_data[time_offset..time_offset + n_samples];
            let right_time = &data.time_data[time_offset + n_samples..time_offset + time_stride];

            let left: Vec<f32> = left_freq_wav[..n_samples]
                .iter()
                .zip(left_time)
                .map(|(f, t)| f + t)
                .collect();
            let right: Vec<f32> = right_freq_wav[..n_samples]
                .iter()
                .zip(right_time)
                .map(|(f, t)| f + t)
                .collect();

            stems.push(Stem {
                id: stem_id,
                left,
                right,
            });
        }

        Ok(stems)
    }

    /// Process a single segment (≤ TRAINING_LENGTH) through the full pipeline.
    async fn separate_single_segment(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        n_samples: usize,
        info: &'static ModelInfo,
        listener: &mut impl ForwardListener,
    ) -> Result<Vec<Stem>> {
        let total_stems = info.stems.len();
        let stems = match &self.opts {
            ModelOptions::FourStem | ModelOptions::SixStem => {
                let seg =
                    self.forward_segment(0, left_channel, right_channel, n_samples, listener)?;
                let data = Self::read_segment(seg, total_stems).await?;
                let stems = Self::stems_from_segment(&data, info)?;
                for i in 0..stems.len() {
                    listener.on_event(ForwardEvent::StemDone {
                        index: i,
                        total: total_stems,
                    });
                }
                stems
            }
            ModelOptions::FineTuned(selected) => {
                let device = &self.device;

                // Pad to TRAINING_LENGTH
                let padded_len = TRAINING_LENGTH;
                let mut left_padded = vec![0.0f32; padded_len];
                let mut right_padded = vec![0.0f32; padded_len];
                left_padded[..n_samples].copy_from_slice(left_channel);
                right_padded[..n_samples].copy_from_slice(right_channel);

                let mut stft = Stft::new(N_FFT, HOP_LENGTH);

                // STFT both channels
                let left_spec = stft.forward(&left_padded)?;
                let right_spec = stft.forward(&right_padded)?;
                let bins = N_FFT / 2;
                let n_frames = left_spec.len() / bins;

                // CaC: each [2, F, T], stack to [4, F, T], add batch → [1, 4, F, T]
                let left_cac = stft_to_cac::<B>(&left_spec, N_FFT, device);
                let right_cac = stft_to_cac::<B>(&right_spec, N_FFT, device);
                let freq = Tensor::cat(vec![left_cac, right_cac], 0).unsqueeze_dim::<4>(0);

                let time = build_time_tensor::<B>(&left_padded, &right_padded, padded_len, device);

                let mut stems = Vec::new();
                for (i, &stem_id) in info.stems.iter().enumerate() {
                    if !selected.contains(&stem_id) {
                        continue;
                    }
                    let (freq_out, time_out) = self.models[i].forward_with_listener(
                        freq.clone(),
                        time.clone(),
                        listener,
                    )?;
                    let stem = extract_single_stem::<B>(
                        &freq_out, &time_out, i, stem_id, n_frames, padded_len, n_samples,
                        &mut stft,
                    )
                    .await?;
                    stems.push(stem);
                    listener.on_event(ForwardEvent::StemDone {
                        index: i,
                        total: total_stems,
                    });
                }
                stems
            }
        };

        Ok(stems)
    }
}

pub enum ModelOptions {
    FourStem,
    SixStem,
    FineTuned(Vec<StemId>),
}

impl ModelOptions {
    pub fn model_info(&self) -> &'static ModelInfo {
        match self {
            ModelOptions::FourStem => &HTDEMUCS,
            ModelOptions::SixStem => &HTDEMUCS_6S,
            ModelOptions::FineTuned(_) => &HTDEMUCS_FT,
        }
    }
}

pub struct Stem {
    pub id: StemId,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// On-device outputs of one segment forward pass, plus segment geometry.
/// Holding this keeps the output tensors alive on the GPU; the work may still
/// be in flight until [`Demucs::read_segment`] awaits it.
struct SegmentForward<B: Backend> {
    freq_out: Tensor<B, 4>, // [1, n_sources * 4, F, padded_T]
    time_out: Tensor<B, 3>, // [1, n_sources * 2, padded_T]
    n_frames: usize,
    n_samples: usize,
}

/// A segment's forward outputs read back to host memory.
struct SegmentData {
    freq_data: Vec<f32>,
    time_data: Vec<f32>,
    n_frames: usize,
    n_samples: usize,
}

/// Accumulates triangular-windowed chunk outputs into full-length stems.
struct ChunkAccumulator {
    out_left: Vec<Vec<f32>>,
    out_right: Vec<Vec<f32>>,
    sum_weight: Vec<f32>,
}

impl ChunkAccumulator {
    fn new(n_stems: usize, n_samples: usize) -> Self {
        Self {
            out_left: vec![vec![0.0f32; n_samples]; n_stems],
            out_right: vec![vec![0.0f32; n_samples]; n_stems],
            sum_weight: vec![0.0f32; n_samples],
        }
    }

    /// Apply a triangular window to `stems` and add them in at `start`.
    fn add(&mut self, info: &ModelInfo, stems: &[Stem], start: usize, chunk_len: usize) {
        let window = triangular_window(chunk_len);
        for stem in stems {
            let s = info
                .stems
                .iter()
                .position(|&id| id == stem.id)
                .expect("stem id missing from model info");
            for (i, &w) in window.iter().enumerate() {
                self.out_left[s][start + i] += w * stem.left[i];
                self.out_right[s][start + i] += w * stem.right[i];
            }
        }
        for (i, &w) in window.iter().enumerate() {
            self.sum_weight[start + i] += w;
        }
    }

    /// Normalize by accumulated weight and produce the final stems.
    fn finalize(mut self, info: &ModelInfo) -> Vec<Stem> {
        let n_samples = self.sum_weight.len();
        let mut stems = Vec::with_capacity(self.out_left.len());
        for (s, &stem_id) in info.stems.iter().enumerate() {
            for i in 0..n_samples {
                let w = self.sum_weight[i];
                if w > 0.0 {
                    self.out_left[s][i] /= w;
                    self.out_right[s][i] /= w;
                }
            }
            stems.push(Stem {
                id: stem_id,
                left: std::mem::take(&mut self.out_left[s]),
                right: std::mem::take(&mut self.out_right[s]),
            });
        }
        stems
    }
}

/// Build a triangular (Bartlett) window of the given length.
/// Ramps linearly from 0 at the edges to 1 at the center.
fn triangular_window(length: usize) -> Vec<f32> {
    if length <= 1 {
        return vec![1.0; length];
    }
    let denom = (length - 1) as f32;
    (0..length)
        .map(|i| 1.0 - (2.0 * i as f32 / denom - 1.0).abs())
        .collect()
}

/// Build the time-domain input tensor [1, 2, padded_time_t] from stereo audio.
fn build_time_tensor<B: Backend>(
    left: &[f32],
    right: &[f32],
    padded_len: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut data = vec![0.0f32; 2 * padded_len];
    data[..left.len()].copy_from_slice(left);
    data[padded_len..padded_len + right.len()].copy_from_slice(right);
    Tensor::from_data(TensorData::new(data, [1, 2, padded_len]), device)
}

/// Extract one stem from model output using bulk GPU readback.
///
/// Reads the stem's freq and time data in 2 GPU→CPU transfers (instead of 3),
/// then does CaC→complex→iSTFT on CPU.
#[allow(clippy::too_many_arguments)]
async fn extract_single_stem<B: Backend>(
    freq_out: &Tensor<B, 4>, // [1, n_sources * 4, F, padded_T]
    time_out: &Tensor<B, 3>, // [1, n_sources * 2, padded_T]
    stem_idx: usize,
    stem_id: StemId,
    n_frames: usize,
    padded_len: usize,
    n_samples: usize,
    stft: &mut Stft,
) -> Result<Stem> {
    let freq_bins = N_FFT / 2;

    // GPU: narrow to this stem's channels, trim time dim (no sync)
    let freq_stem = freq_out
        .clone()
        .narrow(1, stem_idx * 4, 4) // [1, 4, F, padded_T]
        .narrow(3, 0, n_frames) // [1, 4, F, n_frames]
        .squeeze_dim::<3>(0); // [4, F, n_frames]

    let time_stem = time_out
        .clone()
        .narrow(1, stem_idx * 2, 2) // [1, 2, padded_T]
        .narrow(2, 0, n_samples) // [1, 2, n_samples]
        .squeeze_dim::<2>(0); // [2, n_samples]

    // Bulk GPU→CPU readback (2 syncs instead of 3)
    let freq_data: Vec<f32> = freq_stem
        .reshape([4 * freq_bins * n_frames])
        .to_data_async()
        .await
        .map_err(|e| DemucsError::Tensor(format!("freq read failed: {e}")))?
        .convert::<f32>()
        .to_vec()
        .map_err(|e| DemucsError::Tensor(format!("freq extraction failed: {e}")))?;

    let time_data: Vec<f32> = time_stem
        .reshape([2 * n_samples])
        .to_data_async()
        .await
        .map_err(|e| DemucsError::Tensor(format!("time read failed: {e}")))?
        .convert::<f32>()
        .to_vec()
        .map_err(|e| DemucsError::Tensor(format!("time extraction failed: {e}")))?;

    // CPU: CaC → complex → iSTFT → combine
    let ch_stride = 2 * freq_bins * n_frames;
    let left_spec = cac_data_to_complex(&freq_data[..ch_stride], freq_bins, n_frames);
    let right_spec = cac_data_to_complex(&freq_data[ch_stride..], freq_bins, n_frames);

    let left_freq_wav = stft.inverse(&left_spec, padded_len)?;
    let right_freq_wav = stft.inverse(&right_spec, padded_len)?;

    let (left_time, right_time) = time_data.split_at(n_samples);

    let left: Vec<f32> = left_freq_wav[..n_samples]
        .iter()
        .zip(left_time)
        .map(|(f, t)| f + t)
        .collect();
    let right: Vec<f32> = right_freq_wav[..n_samples]
        .iter()
        .zip(right_time)
        .map(|(f, t)| f + t)
        .collect();

    Ok(Stem {
        id: stem_id,
        left,
        right,
    })
}

/// Compute the number of chunks needed for a given sample count.
pub fn num_chunks(n_samples: usize) -> usize {
    if n_samples <= TRAINING_LENGTH {
        return 1;
    }
    let segment = TRAINING_LENGTH;
    let stride = segment * 3 / 4;
    n_samples.saturating_sub(segment).div_ceil(stride) + 1
}

pub(crate) const AUDIO_CHANNELS: usize = 2;

pub(crate) const N_FFT: usize = 4096;
pub(crate) const HOP_LENGTH: usize = 1024;

pub(crate) const CHANNELS: usize = 48;
pub(crate) const GROWTH: usize = 2;
pub(crate) const DEPTH: u32 = 4;
pub(crate) const KERNEL_SIZE: usize = 8;
pub(crate) const STRIDE: usize = 4;
pub(crate) const T_LAYERS: usize = 5;
pub(crate) const T_HEADS: usize = 8;
pub(crate) const T_HIDDEN_SCALE: f32 = 4.0;
pub(crate) const DCONV_COMP: usize = 8;
pub(crate) const DCONV_DEPTH: usize = 2;
pub(crate) const SAMPLE_RATE: usize = 44100;

/// Training segment length in samples. All HTDemucs variants were trained with
/// segment = 39/5 seconds → int(39/5 * 44100) = 343980. The model always pads
/// its input to this length during inference (via `use_train_segment`).
pub const TRAINING_LENGTH: usize = 343980;
