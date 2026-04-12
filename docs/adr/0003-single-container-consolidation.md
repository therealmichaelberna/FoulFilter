# Single container: absorb the GPU scanner sidecar

The original split (web-api container without GPU + gpu-scanner container with
ROCm) existed only because WhisperX could not use the GPU; the web tier had no
reason to load ROCm PyTorch. Once ADR-0001 moved all GPU work to Hugging Face
Transformers inside ordinary PyTorch, that reason vanished: the base image
(`rocm/pytorch`) already ships working ROCm PyTorch, so transcription runs
in-process in the API service. The two-image topology, the internal HTTP hop,
the scanner healthcheck gate, and the published port 5000 are removed; models
also stay loaded across Jobs instead of reloading per request.

## Consequences

- One image, one compose service, one published port (8000).
- A crash in model code can take down the API process; acceptable for
  single-operator use and offset by the ephemeral-job model (ADR-0002).
- Future dual-GPU parallelism would require reintroducing worker separation;
  not designed for today.
