# Ephemeral jobs, persistent transcripts

Job state (queue, progress, outputs) lives only in memory and scratch files:
if the container restarts or crashes, unfinished Jobs are lost and temp uploads
are wiped on next startup. This is deliberate — the operator downloads results
before restarting, and durable queue infrastructure was judged not worth its
complexity for a single-user homelab tool. The one exception is the Transcript:
it is persisted to `/data/transcripts` keyed by content hash and reused to skip
GPU transcription entirely on re-processing (Resume), because transcription is
by far the most expensive stage.
