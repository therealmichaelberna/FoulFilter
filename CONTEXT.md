# FoulFilter

FoulFilter automatically locates profanity in audio/video files and edits it out
(silence, bleep, or cut), optimized for long-form media like audiobooks on AMD
ROCm GPUs.

## Language

### Pipeline artifacts

**Transcript**:
The complete transcription of one media file: ordered segments with text and
timestamps, later refined to word-level timestamps. Persisted to disk because it
is expensive to regenerate.
_Avoid_: subtitles, captions, analysis data

**Segment**:
A contiguous span of transcribed speech with a start, an end, and its text.
Produced by the GPU transcription stage; the unit that alignment windows are
built from.

**Candidate**:
A token position in a Segment whose normalized text matches an entry in the Bad
Words List. A Candidate has only approximate timestamps until Alignment refines
it.

**Alignment Window**:
A contiguous batch of Segments (plus padding) sent together to the aligner so
every word inside gets precise timestamps. Windows tile the whole Transcript
whenever any Candidate exists — alignment is never partial-by-flaggedness.
_Avoid_: crop, snippet

**Hit**:
A Candidate after Alignment (or estimate fallback) confirmed for editing, with
final padded start/end times. Overlapping Hits are merged into one.
_Avoid_: match, detection (ambiguous between candidate and hit)

### Editing

**Censor Method**:
How a Hit is rendered inaudible: `silence` (zero volume), `bleep` (1 kHz tone),
or `remove` (audio cut out; audio files only — video falls back to silence).
_Avoid_: edit mode, filter mode

**Smart Cut**:
An optional LLM refinement that widens, narrows, or rejects a Hit using
surrounding words (e.g. cutting the whole idiom "go to hell", skipping the tool
sense of "hoe"). Never creates new Hits. Widening is only permitted when the
Censor Method is `remove` on an audio file; silence and bleep (and all video)
always stay surgical because stretched dead air is more noticeable than the
word.
_Avoid_: AI pass, semantic removal

### Jobs

**Job**:
One file moving through the pipeline, with progress state exposed over SSE.
Jobs are ephemeral: they exist only while the container runs.
_Avoid_: task, queue item

**Batch**:
A set of files uploaded together that become independent Jobs processed strictly
one at a time (GPU serialization).

**Resume**:
Reusing a persisted Transcript (matched by file hash) to skip GPU transcription
and alignment for a file processed before.

**Rescan Pass**:
An optional second full detection run over the same file with every chunk
boundary shifted (silence-padded start). Catches swear words the first pass
missed — typically words garbled when they straddle a chunk boundary. Detection
only: it adds Segments before matching; alignment still happens once.
_Avoid_: double pass, offset pass, Scan Pass

### Configuration inputs

**Bad Words List**:
The plain-text file of words/phrases (one per line) that must not survive into
the output. Matching is case- and punctuation-insensitive; entries may be
multi-word phrases up to three words.
_Avoid_: blacklist, blocklist, swear list

**Scan Pass** (deprecated):
The removed first-generation architecture that transcribed the whole file twice
(HF Whisper on GPU for detection, then full WhisperX re-transcription on CPU).
Superseded by the hybrid design in ADR-0001.
