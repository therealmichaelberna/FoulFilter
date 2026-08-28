# Censor rendering re-encodes at the source's bitrate, never stream-copies

Censor edits (silence, bleep, and remove) operate below the packet boundary:
muting, sine-mixing, and jump-cutting are all applied by audio filters
(`volume`, `atrim`/`concat`, `sine`), and any ffmpeg filter forces an audio
re-encode — there is no `-c:a copy` for mid-frame edits. We accept that
re-encode but pass the source's codec, sample rate, channels, and bitrate so the
output stays near-input in size and quality. Without this, ffmpeg's default
encoder settings collapsed a 10-hour audiobook from ~477 MB to ~286 MB purely
from bitrate, not from the minutes actually cut. Do not "optimize" this back to
`-c:a copy`: it only works for whole-packet cuts and would silently discard the
bitrate preservation. (Sizing a smaller output is a deliberate downstream
remux, out of scope here. Video tracks already use `-c:v copy`.)
