---
license: apache-2.0
library_name: transformers
tags:
  - audio
  - audio-tokenizer
  - neural-codec
  - moss-tts-family
  - MOSS Audio Tokenizer Nano
  - speech-tokenizer
  - trust-remote-code
---

# MOSS-Audio-Tokenizer-Nano

This repository contains the Hugging Face remote-code implementation and weights for **MOSS-Audio-Tokenizer-Nano**, the lightweight audio tokenizer used by **MOSS-TTS-Nano**.

MOSS-Audio-Tokenizer-Nano is a compact discrete audio tokenizer based on the **Cat** (**C**ausal **A**udio **T**okenizer with **T**ransformer) architecture from [MOSS-Audio-Tokenizer: Scaling Audio Tokenizers for Future Audio Foundation Models](https://arxiv.org/abs/2602.10934). The checkpoint in this repository has **21,969,664 parameters** (approximately **22M**), making it much smaller than the full-size MOSS-Audio-Tokenizer while preserving the 48 kHz stereo tokenizer interface used by the MOSS-TTS family.

## Key Features

- **Small model size**: approximately **22M parameters**, including about 10.45M encoder parameters, 10.45M decoder parameters, and 1.07M quantizer parameters.
- **Native high-resolution audio**: supports **48 kHz** input and output with **2-channel stereo** audio, helping reduce compression loss and improve listening quality.
- **Low-frame-rate discrete codes**: compresses 48 kHz stereo audio into a **12.5 Hz** token stream with a downsample rate of 7,680 samples.
- **Variable bitrate reconstruction**: uses a residual quantizer stack with **16 codebooks** and 1,024 entries per codebook. Each codebook contributes about **0.125 kbps**, for an inference range from **0.125 kbps to 2 kbps**.
- **Transformer-based tokenizer**: uses causal Transformer blocks and supports low-latency streaming encode/decode.
- **MOSS-TTS family interface**: designed as the audio tokenizer backbone for MOSS-TTS-Nano and compatible MOSS-TTS-family workflows.

**Summary:**
By combining a compact causal Transformer tokenizer with native 48 kHz stereo modeling, MOSS-Audio-Tokenizer-Nano reduces the deployment cost of the MOSS audio tokenizer interface while keeping high-fidelity reconstruction for speech, general audio, and music. It provides a lightweight, low-frame-rate, and streaming-friendly discrete audio representation for MOSS-TTS-Nano and other real-time speech generation workflows.

This repository contains a lightweight remote-code implementation that mirrors the current Hugging Face Transformers `transformers.models.moss_audio_tokenizer` module. Load it with `trust_remote_code=True` when needed.

## Evaluation Metrics

The table below compares the reconstruction quality of MOSS-Audio-Tokenizer-Nano with open-source audio tokenizers with **no more than 120M parameters** on speech, audio, and music data. MOSS-Audio-Tokenizer-Nano keeps one of the smallest model sizes in the comparison while supporting **48 kHz stereo** reconstruction.

- Speech metrics are evaluated on LibriSpeech test-clean (English) and AISHELL-2 (Chinese), reported as EN/ZH.
- Audio metrics are evaluated on the AudioSet evaluation subset, while music metrics are evaluated on MUSDB, reported as audio/music.
- STFT-Dist. denotes the STFT distance.
- Higher is better for speech metrics, while lower is better for audio/music metrics (Mel-Loss, STFT-Dist.).
- Ch. denotes the number of input/output channels supported by the audio tokenizer: `ch=1` means mono audio, and `ch=2` means stereo audio.
- Nvq denotes the number of quantizers.

| Model | Params (M) | Sample rate | Ch. | bps | Nvq | Speech: SIM ↑ (EN/ZH) | Speech: STOI ↑ (EN/ZH) | Speech: PESQ-NB ↑ (EN/ZH) | Speech: PESQ-WB ↑ (EN/ZH) | Audio/Music: Mel-Loss ↓ | Audio/Music: STFT-Dist. ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mimi VAE** | 28 | 24k | 1 | -- | -- | 0.75 / 0.54 | 0.91 / 0.83 | 2.92 / 2.20 | 2.30 / 1.73 | 1.35 / 1.31 | 2.70 / 2.59 |
| **DAC** | 77 | 44.1k | 1 | 861 | 1 | 0.30 / 0.20 | 0.76 / 0.68 | 1.55 / 1.36 | 1.24 / 1.15 | 1.25 / 1.18 | 2.71 / 2.54 |
| **SpeechTokenizer** | 120 | 16k | 1 | 1000 | 2 | 0.36 / 0.25 | 0.77 / 0.68 | 1.59 / 1.38 | 1.25 / 1.17 | -- / -- | -- / -- |
| **Mimi** | 96 | 24k | 1 | 1100 | 8 | 0.74 / 0.59 | 0.91 / 0.85 | 2.80 / 2.24 | 2.25 / 1.78 | 1.24 / 1.19 | 2.62 / 2.49 |
| **MOSS-Audio-Tokenizer-Nano** | 22 | 48k | 2 | 750 | 6 | 0.64 / 0.61 | 0.90 / 0.85 | 2.65 / 2.28 | 2.11 / 1.87 | 1.04 / 1.01 | 2.42 / 2.27 |
| **MOSS-Audio-Tokenizer-Nano** | 22 | 48k | 2 | 1000 | 8 | **0.75 / 0.69** | **0.92 / 0.87** | **2.92 / 2.48** | **2.36 / 2.04** | **1.00 / 0.97** | **2.37 / 2.22** |
| **EnCodec** | 19 | 48k | 2 | 1500 | 1 | 0.35 / 0.30 | 0.76 / 0.75 | 1.54 / 1.60 | 1.25 / 1.32 | 1.25 / 1.05 | 2.73 / 2.30 |
| **SpeechTokenizer** | 120 | 16k | 1 | 1500 | 3 | 0.52 / 0.38 | 0.84 / 0.75 | 2.00 / 1.60 | 1.57 / 1.33 | -- / -- | -- / -- |
| **Mimi** | 96 | 24k | 1 | 1512.5 | 11 | 0.82 / 0.67 | 0.92 / 0.88 | 3.10 / 2.50 | 2.54 / 2.00 | 1.19 / 1.14 | 2.55 / 2.42 |
| **DAC** | 77 | 44.1k | 1 | 1723 | 2 | 0.57 / 0.47 | 0.86 / 0.80 | 2.21 / 1.85 | 1.74 / 1.49 | 1.03 / 0.99 | 2.43 / 2.26 |
| **SpeechTokenizer** | 120 | 16k | 1 | 2000 | 4 | 0.66 / 0.50 | 0.88 / 0.80 | 2.38 / 1.79 | 1.92 / 1.49 | -- / -- | -- / -- |
| **Mimi** | 96 | 24k | 1 | 2062.5 | 15 | 0.87 / 0.73 | 0.94 / 0.90 | 3.36 / 2.76 | 2.81 / 2.22 | 1.14 / 1.09 | 2.49 / 2.36 |
| **MOSS-Audio-Tokenizer-Nano** | 22 | 48k | 2 | 1500 | 12 | 0.84 / 0.77 | 0.94 / 0.90 | 3.25 / 2.77 | 2.71 / 2.31 | 0.95 / 0.91 | 2.31 / 2.14 |
| **MOSS-Audio-Tokenizer-Nano** | 22 | 48k | 2 | 2000 | 16 | **0.88 / 0.81** | **0.95 / 0.91** | **3.40 / 2.93** | **2.89 / 2.47** | **0.93 / 0.89** | **2.28 / 2.11** |

## Usage

### Quickstart

```python
import torchaudio
from transformers import AutoModel

repo_id = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()

wav, sr = torchaudio.load("demo/demo_gt.wav")
if sr != model.sampling_rate:
    wav = torchaudio.functional.resample(wav, sr, model.sampling_rate)

# The public waveform interface expects stereo audio.
if wav.shape[0] == 1:
    wav = wav.repeat(model.config.number_channels, 1)
else:
    wav = wav[: model.config.number_channels]

wav = wav.unsqueeze(0)
enc = model.encode(wav, return_dict=True)
print(f"enc.audio_codes.shape: {enc.audio_codes.shape}")

dec = model.decode(enc.audio_codes, return_dict=True)
print(f"dec.audio.shape: {dec.audio.shape}")

wav = dec.audio.squeeze(0)
torchaudio.save("demo/demo_rec.wav", wav, sample_rate=model.sampling_rate)

# Decode with the first 8 codebooks, roughly 1 kbps.
dec_rvq8 = model.decode(enc.audio_codes[:8], return_dict=True)
wav_rvq8 = dec_rvq8.audio.squeeze(0)
torchaudio.save("demo/demo_rec_rvq8.wav", wav_rvq8, sample_rate=model.sampling_rate)
```

### Attention Backend And Compute Dtype

`config.attention_implementation` controls whether Transformer layers prefer `sdpa` or `flash_attention_2`.
`config.compute_dtype` controls the non-quantizer autocast dtype and supports `fp32`, `bf16`, and `fp16`.

```python
model.set_attention_implementation("flash_attention_2")
model.set_compute_dtype("fp16")
```

The quantizer always runs in fp32.

### Streaming

`MossAudioTokenizerModel.encode`, `decode`, `batch_encode`, and `batch_decode` all support streaming through a `chunk_duration` argument.

- `chunk_duration` is expressed in seconds.
- `chunk_duration * MossAudioTokenizerConfig.sampling_rate` must be divisible by `MossAudioTokenizerConfig.downsample_rate`.
- Streaming batch inference is supported.
- The public waveform interface expects stereo inputs shaped `(2, T)` or batched stereo inputs shaped `(B, 2, T)`.

```python
import torch
from transformers import AutoModel

repo_id = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()
audio = torch.randn(2, 48000 * 6)  # dummy stereo waveform

# 6.0s @ 48kHz = 288000 samples, divisible by downsample_rate=3840
enc = model.encode(audio.unsqueeze(0), return_dict=True, chunk_duration=0.08)
dec = model.decode(enc.audio_codes, return_dict=True, chunk_duration=0.08)

batch_enc = model.batch_encode([audio, audio[:, : 48000 * 3]], chunk_duration=0.08)
codes_list = [
    batch_enc.audio_codes[:, i, : batch_enc.audio_codes_lengths[i]]
    for i in range(batch_enc.audio_codes.shape[1])
]
batch_dec = model.batch_decode(codes_list, chunk_duration=0.08)
```

#### Continuous Batch Streaming Decode

For decoder-side continuous batching, prefer `batch_decode(..., streaming=True, ...)`.

- The first streaming call may pass `max_batch_size=...`. If it is omitted, the first batch size reserves the fixed-slot decoder budget for that public stream.
- Same-size calls continue the existing logical rows in order.
- If a later call is larger, the new rows are admitted by tail append.
- `finalize_indices` means "decode these rows one last time, then evict them". The indices are interpreted against the pre-call logical order.
- After a finalize call returns, the next streaming call may use the smaller survivor batch.
- `reset_stream=True` discards the hidden public streaming state and starts a fresh stream.

Milestone 1 boundaries:

- decode-only continuous batching
- one active streaming decode state per model instance
- fixed-slot decoder reservation from `max_batch_size`
- no encode-side continuous batching
- no physical compaction of surviving decode slots
- no multi-session concurrency on one model instance

```python
import torch
from transformers import AutoModel

repo_id = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()
num_quantizers = model.config.quantizer_kwargs["num_quantizers"]
codebook_size = model.config.quantizer_kwargs["codebook_size"]

codes_a0 = torch.randint(0, codebook_size, (num_quantizers, 2))
codes_b0 = torch.randint(0, codebook_size, (num_quantizers, 3))
codes_a1 = torch.randint(0, codebook_size, (num_quantizers, 2))
codes_b1 = torch.randint(0, codebook_size, (num_quantizers, 2))
codes_c0 = torch.randint(0, codebook_size, (num_quantizers, 1))
codes_a2 = torch.randint(0, codebook_size, (num_quantizers, 1))
codes_b2 = torch.randint(0, codebook_size, (num_quantizers, 2))
codes_c1 = torch.randint(0, codebook_size, (num_quantizers, 2))
codes_b3 = torch.randint(0, codebook_size, (num_quantizers, 1))
codes_c2 = torch.randint(0, codebook_size, (num_quantizers, 1))

# First call reserves 3 fixed decoder slots for A and B.
out_ab0 = model.batch_decode(
    [codes_a0, codes_b0],
    streaming=True,
    max_batch_size=3,
    reset_stream=True,
)

# Same logical rows continue in order; C is a tail append.
out_abc1 = model.batch_decode(
    [codes_a1, codes_b1, codes_c0],
    streaming=True,
)

# Finalize A against the pre-call logical order. A still decodes in this call,
# then is evicted immediately afterward.
out_abc2 = model.batch_decode(
    [codes_a2, codes_b2, codes_c1],
    streaming=True,
    finalize_indices=[0],
)

# The next call can shrink to the surviving logical rows only.
out_bc3 = model.batch_decode(
    [codes_b3, codes_c2],
    streaming=True,
)
```

## Repository Layout

- `configuration_moss_audio_tokenizer.py`
- `modeling_moss_audio_tokenizer.py`
- `__init__.py`
- `config.json`
- model weights

## Citation

If you use this model or code in your work, please cite:

```bibtex
@misc{gong2026mossttstechnicalreport,
  title={MOSS-TTS Technical Report},
  author={Yitian Gong and Botian Jiang and Yiwei Zhao and Yucheng Yuan and Kuangwei Chen and Yaozhou Jiang and Cheng Chang and Dong Hong and Mingshu Chen and Ruixiao Li and Yiyang Zhang and Yang Gao and Hanfu Chen and Ke Chen and Songlin Wang and Xiaogui Yang and Yuqian Zhang and Kexin Huang and ZhengYuan Lin and Kang Yu and Ziqi Chen and Jin Wang and Zhaoye Fei and Qinyuan Cheng and Shimin Li and Xipeng Qiu},
  year={2026},
  eprint={2603.18090},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2603.18090}
}
```

```bibtex
@misc{gong2026mossaudiotokenizerscalingaudiotokenizers,
  title={MOSS-Audio-Tokenizer: Scaling Audio Tokenizers for Future Audio Foundation Models},
  author={Yitian Gong and Kuangwei Chen and Zhaoye Fei and Xiaogui Yang and Ke Chen and Yang Wang and Kexin Huang and Mingshu Chen and Ruixiao Li and Qingyuan Cheng and Shimin Li and Xipeng Qiu},
  year={2026},
  eprint={2602.10934},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2602.10934}
}
```
