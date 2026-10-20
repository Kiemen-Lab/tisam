# Package and checkpoint contracts

The project root defines one installable distribution, `tisam`, implemented
under `src/tisam`, with tests under `tests`. A single `pyproject.toml` owns package
metadata, extras, build configuration and development tools, alongside `uv.lock`.
There is no uv workspace. The library does not depend on its repository layout.

## Dependencies and ownership

- `model` contains SAM3 and pathology encoder adapters, pixel/query decoders,
  positional embeddings and predictors. Only the dual-encoder architecture is
  constructible. Internal parameter names retain checkpoint compatibility.
- `config` owns architecture-only `ModelConfig`, raw-label/data descriptions,
  and `TrainConfig`. Dataset settings are not needed to instantiate the model.
- `checkpointing` reads native and historical weights without importing Mammoth.
- `data` owns TIFF resource lifetime, patch geometry, raw-label semantics,
  augmentation and class-count caching. `TileDataset` keeps separate supervision
  and metric masks before remapping raw labels into canonical class IDs.
- Model, data, and inference imports support Python 3.9+. Dataset class-count
  caches use a same-directory temporary file and atomic replacement without
  importing the training runtime. Python 3.9–3.10 pair legacy tifffile with
  Zarr 2; Python 3.11+ pair modern tifffile with Zarr 3.
- `training` loads its public trainer on demand and supports Python 3.9+;
  loss modules can be imported independently. It supplies steps, metrics and
  checkpoint semantics to Mammoth. Mammoth owns optimization-loop, AMP, accumulation, scheduler updates,
  callback lifecycle and atomic checkpoint publication. There is no distributed
  launcher or topology configuration.
- `inference` owns RGB prediction, tile validation/test, and WSI assembly. It
  has no training-runtime import. WSI scratch uses chunked Zarr arrays, a
  completion bitmap, source identity and a model fingerprint. The nonblocking portalocker
  writer lock is released by process exit, allowing recovery after termination.
- `cli` registers only train, eval validate, eval test and eval segment.
  Optional modules load when invoked; base imports and CLI help do not require
  training/WSI extras.

The public foundation choices are UNI2-h, Virchow2 and UNI2-SEAL. SAM3 source and
base weights have immutable identities. Foundation adapters retain the existing
library-native Hugging Face/timm loading and the pinned SEAL adaptation.
An explicitly supplied SAM3 checkpoint is validated against the same canonical
weight digest as the cache download; arbitrary replacement frozen weights are
not accepted. No upstream weights are vendored or uploaded by this project.

## Model I/O

The model consumes normalized BCHW RGB tensors whose spatial shape must match
`ModelConfig.input_hw`. A mismatched input height or width is rejected before
encoding. Both `input_hw` and `output_hw` default to `(1024, 1024)`; training
tiles are resized to `input_hw`, and prediction and WSI helpers resize arbitrary
inputs to `input_hw` before the forward pass. The model returns BCHW semantic
logits with canonical void at channel 0. `num_classes` excludes void;
`total_queries` defaults to the resulting total class count. SAM3 handles its
input resizing internally. The configurable `output_hw` determines the dense
logits; image/WSI helpers resize logits to image/patch coordinates before
argmax.

`return_probs=True` returns normalized probabilities. `return_intermediates=True`
returns the existing semantic-mask and decoder-output mapping. Frozen encoders
stay in evaluation mode while trainable decoder modules train.

## Weights

Native metadata uses `tisam_schema=1`, `tisam_model_config`, `class_names`,
`mean` and `std`. Best weights contain trainable parameters and persistent
buffers, with reconstructible complex RoPE buffers omitted for safetensors.
Frozen encoder parameters are reconstructed from their pretrained identities.
Missing required trainable weights, unknown weights, incompatible shapes and
conflicting shared-parameter aliases are errors.

Historical import translates the dual-encoder architecture fields from
`tisam_config_payload`. Files without metadata require an explicit model config.
The obsolete positional-projection keys receive the existing narrow legacy
exception; no general architecture fallback exists. An explicit config must
agree with saved architecture metadata. Model-only loading returns eval mode.

## Training state

Native resumable `.pt` files include `training_schema=1`, model weights,
architecture/preprocessing metadata, resolved training configuration, optimizer,
scheduler, early-stopping callback, scaler, epoch and step counters, plus Python,
NumPy and Torch random-generator states. This format is loadable with
`torch.load(weights_only=True)`. Only historical trusted pickle import permits
unrestricted loading.

Resume checks the model, classes, data, loss, optimizer and scheduler horizon.
Output location/name, device, loader worker settings and compile choice are
runtime settings. A new training objective or dataset uses explicit weights-only
initialization and a new run name. Cross-project optimizer or run-provenance
migration is intentionally unsupported. Random-generator restoration does not
claim bitwise replay across worker counts, hardware, or library versions.

Mammoth owns latest/all retention, atomic publication, and best-checkpoint
selection. Its committed publication receipts are retained in the attempt's
JSONL stream, including SHA-256, size, role, epoch, and retired paths. A receipt
describes the bytes at publication time; later retention or replacement may
remove those bytes. Training does not overwrite checkpoints for a fresh restart.

TiSAM claims Mammoth's logical-run lease before checkpoint selection and composes
an `ExecutionSession` from the established execution context and logging bundle.
Mammoth owns cross-platform lease handling, exclusive text logs, phase/process
terminal events, and log cleanup. TiSAM supplies configuration and checkpoint
receipts, training metric fields, TensorBoard output, and Python logger routing.
The lease remains held until checkpoint and log flushes finish. Training uses
Mammoth 0.9.0's single-process Windows support and compressed event streams;
model loading and data imports remain independent of Mammoth. Inference uses
Mammoth core pipelines without its training runtime or TensorBoard extra.

Older TiSAM versions used a separate `.training.lock`. Stop older training
processes before upgrading a run: old and new versions do not share the same
ownership lock. Existing checkpoints and execution records remain compatible;
a leftover `.training.lock` file does not prevent a new attempt.

Each invocation publishes immutable execution metadata and an atomic config
snapshot under `logs/executions/<execution-id>/`. The metadata records the prior
attempt, starting epoch/global step, and exact resume checkpoint digest. Both
resume preflight and restore use Mammoth descriptor-bound artifact sessions;
restore rejects bytes differing from the preflight receipt before changing model
state. Historical checkpoint and model-only loading remain independent of these
training-only imports. `config.json` at the run root remains a convenience copy.

Setup and training share one process lifecycle. TiSAM scopes its outer train
phase through the Mammoth session, which records phase and process outcomes;
Mammoth's Trainer owns epoch tasks and validation phases.
Checkpoints flush before successful phase completion. JSONL keeps progress,
heartbeats, all epoch metrics, config/resume receipts, and checkpoint publication
events. Training `throughput` is Mammoth's accumulation-window rate;
`batches_per_second` converts it using actual consumed batches, not the nominal
accumulation factor. Both rates are epoch averages, not instantaneous samples.
TensorBoard retains dense metrics in each attempt's `tensorboard/` directory.

## Local provenance

The extraction starts from research source snapshot
`641b4a6f783f58cf16c4c9c8d06b9b09f7bb9a77`; no source-repository history or data is
included. The decoder/loss math and supported dual-encoder options are preserved.
The source is hosted publicly at https://github.com/Kiemen-Lab/tisam. License
selection, package registry publication and hosted CI remain unconfigured.

## CLI segmentation and previews

All command configuration/checkpoint inputs use explicit named options. The
segment CLI delegates to `inference.batch.segment_sources`, which inventories
one file or a nonrecursive directory, checks stem collisions and palette
coverage, and loads one model lazily for unfinished masks. `inference.wsi`
retains the single-file integer TIFF and interrupted-scratch resume contract.
Completed mask reuse is explicit through default CLI resume; it does not infer
checkpoint identity from an arbitrary final TIFF. `--no-resume` rejects it.

`inference.overlays` consumes completed masks and `DataConfig.colormap`, ordered
by canonical `class_names`. Independent `--check-classify` and
`--mini-check-classify` options default false and publish normal and tenfold
smaller JPGs under `check_classification/` and its `mini/` directory. Neither
option implies the other. Rendering blends 30% palette color with 70% RGB;
mini uses area resizing after blending. Missing JPGs can be repaired without
model loading. JPG writes use temporary files and atomic replacement; existing
JPGs are retained. Rendering materializes one slide at a time, unlike the
bounded-memory mask writer. Configuration examples and CLI migration usage
are maintained in README.md.

## Bounded asynchronous inference

`inference.transfers.BatchPrefetcher` stages one batch ahead on a CUDA H2D
stream, using pinned DataLoader buffers. `PredictionHandoff` retains the integer
GPU cores, pinned host buffer and D2H completion event until the writer resolves
them. CPU runs use the same handoff without CUDA events. The sole Zarr writer is
Mammoth's `BoundedBackgroundPipeline(max_pending=1)`; it writes each core before
marking its completion bitmap. Its context drains accepted jobs before source
resources close or the exclusive scratch lock is released.

`inference.wsi.prepare_wsi` returns a `PreparedMask` only after all predictions
are committed. This CPU-only object owns the completed Zarr location and lock;
`publish` reads chunks into a tiled Zstd TIFF, publishes it atomically and removes
successful scratch. Failure releases the lock while preserving recoverable data.
`segment_wsi` remains the synchronous single-file API by publishing immediately.

`inference.batch` hands completed slides to a second Mammoth pipeline with one
active publication. It publishes the TIFF and requested previews while inference
prepares the next slide. Before handing off that next slide it waits for queue
capacity. Rejected handoffs release their lock; accepted work is drained even
on error or interruption. Final exit waits for publication and propagates worker
failures. Preview-only repair uses this same worker. The maximum is one publishing
slide plus one being inferred/prepared, not an unbounded collection of WSIs.
DataLoader uses spawn when workers are requested, avoiding fork after background
threads/CUDA initialization. Inference depends on Mammoth core via its optional
extra; model imports and the base wheel still do not require Mammoth.
