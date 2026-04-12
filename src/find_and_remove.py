"""find_and_remove.py - CLI entry point around the FoulFilter pipeline.

The web service calls pipeline.run_job directly; this wrapper exists for
one-off terminal runs. Transcripts are cached by file hash under
/data/transcripts and reused automatically.
"""

import argparse
import logging
import os

from analyze_file_type import detect_media_type
from pipeline import run_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FoulFilter")

parser = argparse.ArgumentParser(
    prog="FoulFilter",
    description="Finds and removes a list of swear words from an audio or video file.",
)
parser.add_argument("file_path", type=str, help="Path to the audio/video file")
parser.add_argument(
    "bad_words_list_path",
    type=str,
    help="Path to file containing words that should be removed",
)
parser.add_argument("--output", type=str, help="Explicit path to save the final file")
parser.add_argument(
    "--bleep",
    action="store_true",
    help="Bleep out words instead of replacing with silence",
)
parser.add_argument(
    "--delete",
    action="store_true",
    help="Cut words out entirely instead of silence (audio files only)",
)
parser.add_argument(
    "--censor_method",
    type=str,
    choices=["silence", "bleep", "remove"],
    help="Override CENSOR_METHOD env/flags with an explicit method",
)
parser.add_argument("--debug", action="store_true", help="Export transcript to text file")
parser.add_argument(
    "--rescan",
    action="store_true",
    help="Second detection pass with shifted chunk boundaries (finds boundary-garbled words)",
)
parser.add_argument(
    "--no_edit",
    action="store_true",
    help="Analyze only: report what would be cut, do not edit",
)

env_censor_method = os.getenv("CENSOR_METHOD", "silence").lower()
args = parser.parse_args()

if args.censor_method:
    censor_method = args.censor_method
elif args.bleep:
    censor_method = "bleep"
elif args.delete:
    censor_method = "remove"
else:
    # legacy env accepted 'delete' as a synonym for remove
    censor_method = "remove" if env_censor_method == "delete" else env_censor_method

media_type = detect_media_type(args.file_path)
if media_type == "unknown":
    print(f"Error: Unrecognized file type: {args.file_path}")
    raise SystemExit(1)

base = os.path.splitext(os.path.basename(args.file_path))[0]
output = args.output or os.path.join(
    os.path.dirname(args.file_path) or ".", f"censored_{base}{os.path.splitext(args.file_path)[1]}"
)

summary = run_job(
    input_path=args.file_path,
    output_path=output,
    bad_words_list_path=args.bad_words_list_path,
    transcript_dir=os.getenv("TRANSCRIPT_DIR") or "/data/transcripts",
    scratch_dir=os.path.join(os.path.dirname(output) or ".", f".{base}_scratch"),
    censor_method=censor_method,
    debug=args.debug,
    rescan=args.rescan,
    render=not args.no_edit,
)

if summary["hits"]:
    print(f"{len(summary['hits'])} hit(s):")
    for h in summary["hits"]:
        print(f"  {h['start']:8.2f} - {h['end']:8.2f}  {h['phrase']}")
else:
    print("No inappropriate words found.")
print(f"Done. Output: {output if not args.no_edit else '(not written)'}")
