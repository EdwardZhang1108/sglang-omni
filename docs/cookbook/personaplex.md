# PersonaPlex

[nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) is a 7B full-duplex speech-to-speech model built on [Moshi](https://arxiv.org/abs/2410.00037): every 80 ms it reads one text token and 16 [Mimi](https://huggingface.co/kyutai/mimi) codes (8 for what it says, 8 for what it hears) and writes the next frame. A voice prompt and a `<system>` role prompt set the persona, and the model decides for itself when to speak; there is no VAD.

SGLang-Omni serves it two ways: an **offline** pipeline (one recording of the caller's side in, the agent's reply as text and 24 kHz audio of the same length out) and a **session** variant that holds live full-duplex sessions over `/v1/realtime` ([#1909](https://github.com/sgl-project/sglang-omni/issues/1909)).

## Prerequisites

The checkpoint is gated; accept the license on the model page and log in (`hf auth login`), then:

```bash
hf download nvidia/personaplex-7b-v1
```

It ships the 7B weights, the Mimi codec, the SentencePiece text model and `voices.tgz`, which is unpacked into `voices/` next to the checkpoint on first use, or into the temp directory when that folder cannot be written. Recorded voice prompts (`--voice some.wav`) also need `pip install pyloudnorm`; the packaged `.pt` voices do not.

Everything runs on one GPU. On an H200 the LM engine reserves `mem_fraction_static=0.3` and the two Mimi instances stay under 1 GB each. This value is a fraction of total device memory, so a smaller-memory device can need a higher fraction to fit the model weights.

The Mimi encoder stage loads only its encoding path, and Code2Wav loads only its
decoding path. Both retain the codebooks and quantizer projections their path
needs, reducing the combined FP32 codec weights and buffers by about 302 MiB.

## Running the offline example

```bash
python examples/run_personaplex.py \
  --model-path nvidia/personaplex-7b-v1 \
  --audio /path/to/caller.wav \
  --voice NATF2 \
  --text-prompt "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way." \
  --out reply.wav
```

Five stages run under `MultiProcessPipelineRunner` (preprocessing, Mimi encode, the LM engine, text decode, streaming code2wav). Pick the GPU with `CUDA_VISIBLE_DEVICES`; stage settings take the same dotted flags as `serve`, such as `--lm.engine.mem_fraction_static 0.25`. Input conventions:

- The recording is resampled to 24 kHz and **channel 0 is used**.
- The reply is exactly as long as the input, offset by one frame: the model answers while it listens, so leave silence after the caller's last words if you want a full answer.
- Prompt plus reply must fit the LM context, 8192 positions by default (about 10.8 minutes); a longer recording is rejected with the limit in the message and needs `--lm.engine.context_length`.
- The reply text is the model's inner monologue with the frame markers (`PAD`, `EPAD`, `BOS`, `EOS`) removed.

For Intel GPUs, follow the [PersonaPlex XPU recipe](../get_started/installation_xpu.md#personaplex-speech-to-speech-single-xpu).

## Serving over HTTP

```bash
python -m sglang_omni.cli serve --model-path nvidia/personaplex-7b-v1 --port 8000
```

Send the recording as the `prompt` of `POST /generate` (a path, URL or `data:` URI, with `"return_logprob": false`), or as the single entry of `audios` in `POST /v1/chat/completions` with `"modalities": ["text", "audio"]`; the reply audio comes back base64-encoded as WAV. Options with no request field of their own go under `stage_params`: `voice` and `text_prompt` under `preprocessing`, `audio_temperature`, `audio_top_k` and `seed` under `lm`.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nvidia/personaplex-7b-v1",
  "messages": [{"role": "user", "content": ""}],
  "audios": ["/path/to/caller.wav"],
  "modalities": ["text", "audio"],
  "stage_params": {"preprocessing": {"voice": "NATM1"}, "lm": {"audio_temperature": 0.8}}
}'
```

## Native Full-Duplex Ability

PersonaPlex also serves live full-duplex voice sessions on `/v1/realtime` with its `session` variant. Launching, settings and the protocol are in [Full Duplex Interactive Model](../developer_reference/full_duplex_guide.md#native-full-duplex-personaplex).

## Request parameters

| Parameter | Effect |
|---|---|
| `voice` | A packaged voice name (`NATF0..3`, `NATM0..3`, `VARF0..4`, `VARM0..4`), a `.pt` file, or a recording. Default `NATF2`; an empty string runs without a voice prompt. |
| `text_prompt` (or `instructions`) | The role prompt, wrapped in `<system>` tags. Default: the reference assistant prompt. |
| `temperature`, `top_k` | Text sampling, defaults 0.7 / 25. `temperature=0` is greedy. |
| `audio_temperature`, `audio_top_k` | Code sampling in the depformer; defaults 0.8 / 250. |
| `seed` | Makes both draws reproducible (a child seed each for text and audio). |
| `max_new_tokens` | Ignored; the frame count is fixed by the input length. |

### Stage sampling

Fields left out of `stage_sampling.lm` keep the PersonaPlex text defaults.
For example, `{"seed": 7}` keeps temperature 0.7 and top-k 25, while
`{"temperature": 1.0, "top_k": -1}` samples at temperature 1.0 with no top-k
filtering. The same holds for `SamplingParams` objects passed to the Python
client: `SamplingParams(seed=7)` in `stage_sampling.lm` sets only the seed.

Text sampling uses `stage_sampling.lm`, then `stage_params.lm`, then top-level
request values. Stage sampling also takes precedence for `seed`.
`top_p`, `min_p`, and `repetition_penalty` apply to the text sampler only;
`audio_temperature` and `audio_top_k` continue to control the audio sampler.
The fixed caller-frame budget still determines the number of generated frames.


## Opt-in FP8 inference

BF16 remains the default. The existing LM engine settings enable FP8 independently:

| Setting | Effect |
|---|---|
| `--lm.engine.quantization fp8` | Quantize the backbone's attention and MLP linear weights. Ampere uses weight-only Marlin; supported sm89+ devices use native FP8 linear kernels. |
| `--lm.engine.kv_cache_dtype fp8_e4m3` | Store the backbone KV cache in FP8. The evaluation below uses H100 with FA3. |
| Both settings | Combine FP8 backbone weights and KV storage. |

The text head, embeddings, depformer and Mimi codecs keep their existing precision.
On the tested H100 80GB and A100 SXM4 80GB, the quantized linear weights occupy
6.58 GB instead of 13.15 GB. With a fixed 8 GiB KV budget, FP8 doubles the pool
from 16,384 to 32,768 positions. These are tensor-storage and allocated-pool
measurements, not whole-device memory or a guarantee of twice as many active
requests. Set `--lm.engine.max_running_requests` explicitly when increasing
concurrency; the default is one.

### Audio quality

The following evaluation used SGLang 0.5.22, PyTorch 2.14.1, checkpoint revision
`fdaf4090a61cb315c138a1faee287ffd6c716309`, CUDA graphs, concurrency 8, four
packaged voices, and the default text/audio sampling settings. Each configuration
generated 832 replies from 416 distinct English Seed-TTS inputs and two seeds.
Inputs were padded to 30 seconds; every frozen reply was retained and scored.
H100 used FA3 and A100 used FlashInfer, with a separate BF16 reference on each
device. All FP8 KV configurations used unit scales.

| GPU | Backbone / KV | UTMOS ↑ | WER ↓ | Word omissions ↓ |
|---|---|---:|---:|---:|
| H100 | BF16 / BF16 | 3.689 | 4.237% | 0.350% |
| H100 | FP8 / BF16 | 3.679 | 3.987% | 0.341% |
| H100 | BF16 / FP8 | 3.685 | 3.948% | 0.336% |
| H100 | FP8 / FP8 | 3.679 | 3.712% | 0.299% |
| A100 SXM4 | BF16 / BF16 | 3.698 | 4.080% | 0.318% |
| A100 SXM4 | FP8 / BF16 | 3.684 | 3.959% | 0.330% |

Each FP8 configuration passed all three predeclared non-inferiority margins:
UTMOS loss below 0.10, WER increase below 1.0 percentage point, and omission
increase below 0.5 percentage point. The 95% bootstrap intervals resampled inputs,
keeping their two seeds together. This is a per-configuration conclusion, without
an across-configuration familywise error guarantee.

WER compares ASR of each reply with that reply's own generated text. It measures
text/audio consistency, not whether the answer is correct or semantically equal
to BF16. UTMOS predicts acoustic quality. These results do not establish greedy
parity, quality at other concurrency levels, or long-conversation quality.

FA3 also casts queries to FP8 when FP8 KV is enabled, without a query scale.
Consequently, these H100 results include FP8 attention arithmetic and should not
be transferred to another attention backend. Ada hardware and FP8 KV on Ampere
were not evaluated here.

### Performance tradeoffs

In the same environment, four balanced blocks compared distinct service starts
on each of four GPUs per hardware type. H100 shared its BF16 arm across the
three candidates using a Williams design; A100 used AB/BA/BA/AB. Each start
ran C1/C4/C8, with two discarded warmup batches and two measured batches at
each concurrency. Graphs stayed enabled and the KV budget stayed at 8 GiB.
Artifact writes and profiling were excluded from timing.

| GPU | Backbone / KV | C1 audio s/s ↑ | C8 audio s/s ↑ | C1 TTFA (ms) ↓ | C8 TTFA (ms) ↓ |
|---|---|---:|---:|---:|---:|
| H100 | BF16 / BF16 | 6.95 | 44.40 | 58.2 | 172.7 |
| H100 | FP8 / BF16 | 8.23 | 50.66 | 66.5 | 179.1 |
| H100 | BF16 / FP8 | 6.82 | 44.68 | 59.8 | 171.8 |
| H100 | FP8 / FP8 | 7.89 | 50.14 | 69.3 | 176.6 |
| A100 SXM4 | BF16 / BF16 | 4.35 | 26.24 | 90.4 | 288.7 |
| A100 SXM4 | FP8 / BF16 | 5.51 | 31.35 | 94.1 | 294.9 |

Values are arithmetic means across service starts. Audio s/s is completed audio
duration divided by measured batch wall time: these are finite batches, not
continuous saturated throughput. C1/C4/C8 use the first 1/4/8 inputs of the same
eight-input set, so C1 covers one input and cross-concurrency differences also
include workload differences.

FP8 weights with BF16 KV are an option for lower weight storage and higher
throughput on the tested workload. Their paired throughput gains ranged from
16.7% to 20.3% at C1 on H100 and 23.5% to 31.1% on A100. Mean end-to-end
latency fell at all three concurrency levels, but mean TTFA increased by 6–10 ms on H100 and 4–6 ms on
A100. Keep this first-audio cost in mind for latency-sensitive deployments.

FP8 KV alone did not provide a consistent throughput gain: at C4 it reduced
throughput in all four pairs, while C1 and C8 effects crossed zero across pairs.
Treat it as a storage/capacity option. The combined setting retained a throughput
gain over BF16 but was slower on average than FP8 weights alone. H100 C8 audio
chunk-gap p99 effects varied in both directions; do not infer a tail-latency gain
from the mean throughput improvement. Four blocks provide exploratory uncertainty,
not a guarantee for other inputs or deployments.

Separate 16-step decode profiles confirmed weight-only Marlin on A100 and
native FP8 GEMM with per-token activation quantization on H100. The H100
FP8 KV profiles also confirmed FP8 FA3 attention. Profiling durations are not
included in the benchmark table.

### Long-request capacity

With a fixed 8 GiB KV budget and `max_running_requests=16`, instrumented
450-second padded-input runs sustained four BF16-KV streams or eight FP8-KV
streams beyond the attention window. The next tested levels, five and nine,
completed in staggered groups. All 85 requests across 11 capacity runs completed
with graph decode, no retractions and finite pre-PCM audio. Each level has one
run; these are observed workload-specific boundaries, not universal capacity
limits or long-dialogue quality results.

The physical pool doubled, but dividing its positions by the approximately
3.1k steady positions per stream overpredicts sustained concurrency here. The
scheduler also reserves future generation tokens at admission; requests staggered
before the pool filled. Capacity-test timings are instrumented and excluded
from the performance results above.

[Raw measurements, frozen protocols and reproduction scripts](https://gist.github.com/EdwardZhang1108/ca872094ef9d1000ab446f982c3b86fa)
include per-reply quality metrics, paired timings, capacity outcomes and kernel
profiles. All measurements use runtime revision `b8a6e40`.

### Custom KV scales

An optional calibration file can be supplied through
`--lm.engine.quantization_param_path /path/to/scales.json`. Its SGLang schema uses
`model_type: "llama"`, `kv_cache.dtype: "float8_e4m3fn"`, and
`kv_cache.scaling_factor` grouped by tensor-parallel rank, with all 32 layer
indices present for every rank. Each scale applies to both K and V for that
layer. Missing files, invalid schemas, incomplete ranks/layers, and scales that
are not finite and positive in float32 fail startup instead of silently using
unit scales. The loader updates both tensor and host-float scales before graph
capture, preserving existing scale tensor storage.

The quality results above do not validate a custom calibration file. Changing
KV scales cannot correct FA3's unscaled query cast; validate the resulting audio
on the intended workload.

## Known limitations

### All platforms

- Offline, one request at a time by default (`max_running_requests=1`). If you raise `--lm.engine.max_running_requests`, the requests in a batch share one depformer pass per frame, and each seeded request still draws from its own generator. A seeded reply repeats exactly only with one request in flight, because batched kernels can round differently.
- Streaming Mimi decode follows `--code2wav.factory.chunk_ramp_frames` (default `[1, 4]`: the first frame alone, then four at a time; a finished stream flushes its shorter remainder). Ready requests whose chunks share a width decode in one batched step of up to `max_batch_size` (default 8), each at its own decode position. `num_decode_slots` resident decode states default to the LM's effective `max_running_requests`, and a smaller explicit value is rejected; a stream that finds every slot taken while another stream's tail drains waits for the next free slot, so sustained overload is not bounded by this queue.
- The temporal attention window follows the streaming ring, including the masked oldest slot once its 3000-position cache fills. Boundary tests check this rule; they do not measure long-input audio quality.

### CUDA

- On CUDA, the Depformer's RMSNorm and SiLU-gate run as fused Triton kernels whose outputs match the eager operations bit for bit; other devices use the eager operations.
- On CUDA the backbone's decode step and all 8 Depformer steps replay CUDA graphs. At startup, offline serving captures the default audio-sampling layout for every configured batch bucket through the smallest bucket that covers the effective `max_running_requests`; session serving captures the same buckets with the deployment's resolved audio sampling. Other sampling layouts and batch shapes run eagerly. `--lm.engine.disable_cuda_graph true` disables both backbone and Depformer graphs, while `--lm.factory.depformer_cuda_graph_batch_sizes '[]'` disables only Depformer graphs. Seeded requests preserve their per-request RNG draws.
- Decode feedback stays on device. On CUDA, `--lm.factory.enable_async_decode true` launches the next decode before collecting the previous result and can add 16–21% throughput with identical audio; synchronous decode remains the default. Other platforms, requests needing host token history for sampling, and requests returning log probabilities use the synchronous path. When any request reaches its caller-frame budget, the whole batch uses synchronous decode for that step.
- On CUDA, Mimi decode graphs cover every batch size up to `min(max_batch_size, num_decode_slots)` and every chunk width up to `max_graph_frames` (the widest ramp entry unless set; `0` turns the graphs off), 8 × 4 = 32 graphs at LM concurrency 16; other shapes and platforms decode eagerly with the same states.
- On CUDA, streaming Mimi decode uses a dedicated stream. The stage accepts CPU codes only; transfers and decode-slot reset/replay stay on that stream, so waveform collection does not drain queued LM forwards. Other devices retain their normal execution context.
- On a CUDA platform with resolved `device=cuda`, `page_size=1` and radix caching disabled (the default), window KV releases expired positions for reuse. Steady decode holds about 3.1k positions (1.5 GiB of BF16 KV) per long request with the default eviction interval; prefill and retract replay can temporarily hold more. CPU stages, other platforms, page sizes and radix-enabled configurations use the full KV pool. Context and request-length limits remain unchanged, and returning slots does not shrink the preallocated pool. Set `--lm.engine.disable_hybrid_swa_memory true` to use the full pool explicitly.

### Ascend NPU

- On Ascend NPU the backbone graph is off by default; pass `--lm.engine.disable_cuda_graph false` to replay the backbone's NPU graph.
- On Ascend NPU, the platform enables the same Mimi graph ladder; the capture temporarily disables internal formats so Mimi's convolutions run through [aclnn](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/aolapi/context/ops-math/op_api_list.md).

## Tests

`tests/unit_test/personaplex/` runs on CPU without weights: the delayed timeline, chunked Mimi against whole-sequence Mimi, the depformer, checkpoint weight routing, the model-runner hooks, the streaming codec stage, the checkpoint shim, preprocessing and voice unpacking, and request lowering.

```bash
pytest tests/unit_test/personaplex/ -q
```

### Reference parity

Two opt-in tests compare against an [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex) checkout. Its `moshi` package pins `torch < 2.5`, so it needs its own environment, separate from the serving one:

```bash
git clone https://github.com/NVIDIA/personaplex.git /path/to/personaplex
uv venv /path/to/personaplex/.venv -p 3.12
uv pip install --python /path/to/personaplex/.venv/bin/python /path/to/personaplex/moshi/
```

Then run the tests from the serving environment. They need CUDA, skip unless both variables are set, and run one file per command:

```bash
export PERSONAPLEX_REFERENCE_SOURCE=/path/to/personaplex
export PERSONAPLEX_REFERENCE_PYTHON=/path/to/personaplex/.venv/bin/python
python -m pytest tests/test_model/test_personaplex_parity.py -v -s
python -m pytest tests/test_model/test_personaplex_components.py -v -s
```

- `test_personaplex_parity.py`: greedy replies to the reference's two test recordings must have the input's length, match it for at least the first 100 frames, and agree in text over that prefix; greedy and seeded replies must also repeat. **This is not full-output parity.**
- `test_personaplex_components.py`: Mimi, the input embeddings and the depformer logits on the public `kyutai/moshiko-pytorch-bf16` base. The last depformer step matches only with the reference's ring behavior emulated, which diagnoses a known difference rather than proving a match.

`PERSONAPLEX_PARITY_CHECKPOINT` and `PERSONAPLEX_MOSHI_CHECKPOINT` override the checkpoints, `PERSONAPLEX_STAGE_ARGS` forwards pipeline flags such as `'--lm.engine.mem_fraction_static 0.5'`, and `PERSONAPLEX_REFERENCE_REPO` sets the reference CLI's config lookup. Reference outputs are regenerated on every run under pytest's temporary directory.
