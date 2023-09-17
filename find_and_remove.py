import sys
from word_finder import find_all_words_in_all_segments,find_bad_words
from word_remover import delete_audio_using_word_match_list
import argparse
from splitter import split_for_analysis
import pickle
import os
from video_edit import replace_audio_in_video
from analyze_file_type import detect_media_type
import ffmpeg
import shutil
import tempfile

parser = argparse.ArgumentParser(
                    prog='Swear Word Remover',
                    description='Finds and removes a list of swear words from an audio file.')
                    
parser.add_argument('file_path', type=str, help='Path to the audio file')
parser.add_argument('bad_words_list_path', type=str, help='Path to file containing words that should be removed')
parser.add_argument('--bleep', action='store_true', help='Bleep out words instead of replacing with silence (default is False)')
parser.add_argument('--analysis_chunk_size', type=int, default=300, help='How many seconds of audio to split your chunks into for analysis (defaults to 300 seconds [5 minutes]) less than 15 seconds may be inaccurate')
parser.add_argument('--edit_chunk_size', type=int, default=300, help='How many seconds of audio to split your chunks into for editing (defaults to 300 seconds [5 minutes])')
parser.add_argument('--no_edit', action='store_true', help="Analyze only don't edit")
parser.add_argument('--resume', action='store_true', help="Resume a previous run (input audio filename must remain same)")
parser.add_argument('--transcript', type=str, help='Path to transcript (optional, improves accuracy if transcript is correct)')

args = parser.parse_args()

file_path = args.file_path
bad_words_list_path = args.bad_words_list_path
bleep = args.bleep
analysis_chunk_size = args.analysis_chunk_size
edit_chunk_size = args.edit_chunk_size
no_edit = args.no_edit
transcript_path = args.transcript
#existing_analysis_file = args.existing_analysis
resume = args.resume

#print(f"existing analysis file: {existing_analysis_file}")
# to do:
#   implement LCS to match transcript
#   
# plan for large files
# 1. split file into xx minute pieces for analysis and convert to wav
# 2. analyze files
# 3. edit the split file
# 4. comnine split files
bad_words = []
discovered_words_list = []

media_type = detect_media_type(file_path)


if media_type == "unknown":
    print("Error: Unrecognized file type.")
    sys.exit(1) #stop further execution
else:
    sys_temp_dir = tempfile.gettempdir()
    tmp_dir = sys_temp_dir + "/" + os.path.splitext(os.path.basename(file_path))[0] + "_tmp"
    #print(f"using {tmp_dir} as temp directory")
    #tmp_dir = "./" + os.path.splitext(os.path.basename(file_path))[0] + "_tmp"
    os.makedirs(tmp_dir, exist_ok = True)#create our tmp directory



with open(bad_words_list_path, 'r') as file: # Open the file in read mode
    # Read the lines and strip newline characters
    bad_words = [line.strip() for line in file]
    
if len(bad_words) < 1:
    print("It seems that your bad words list is empty. Please add words to it that you would like to detect and remove.")
    sys.exit(1) #stop further execution

if resume:
    data_file = tmp_dir + "/data.pkl"
    if os.path.exists(data_file):
        with open(data_file, "rb") as pickle_file:
            discovered_words_list = pickle.load(pickle_file)
            if len(discovered_words_list) < 1:
                print("Error loading data. Please run again without --resume to re-generate")
                sys.exit(1) #stop further execution
            print(f"data successfully loaded from {data_file}")
    else:
        print("Error! Resume selected, but data not present. Please remove --resume flag and re-run to generate data.")
        sys.exit(1) #stop further execution 

else:
    if media_type == "audio":
        sound_file_path = file_path
    elif media_type == "video":
        sound_file_path = tmp_dir + "/" + "extracted_audio_track.mp3"
        ffmpeg.input(file_path).output(sound_file_path).run(overwrite_output=True)

    analysis_files = split_for_analysis(sound_file_path, tmp_dir, analysis_chunk_size)
    discovered_bad_words_list = []
    discovered_words_file_export_name = tmp_dir + "/data.pkl"

if len(discovered_words_list) == 0:# no existing data loaded
    print("no discovered_words_list found")
    discovered_words_list = find_all_words_in_all_segments(analysis_files, tmp_dir, transcript_path)

    with open(discovered_words_file_export_name, "wb") as pickle_file: #save our data so that we can resume later if we want to or something crashes.
        pickle.dump(discovered_words_list, pickle_file)
        print(f"saved data to pickl at {pickle_file}")

discovered_bad_words_list = find_bad_words(discovered_words_list, bad_words)

if len(discovered_bad_words_list) > 0:
    print(len(discovered_bad_words_list), "bad words discovered.")

    if no_edit == True:
        print(f"No edit was specified. Analysis is complete. Stopping.")
        sys.exit(1) #stop further execution
    
    #print(f"searching and removing prohibited words from audio portions in {tmp_dir}")
    print("Initiating removal function.")
    
    edited_audio = delete_audio_using_word_match_list(sound_file_path, discovered_bad_words_list, bleep)
    out_format = "mp3"
    if media_type == "audio":
        output_audio_file = os.path.splitext(os.path.basename(sound_file_path))[0] + "_CENSORED." + out_format
        print(f"Edited file succesfully saved to {output_audio_file}")
    if media_type == "video":
        output_audio_file = tmp_dir + "/" + os.path.splitext(os.path.basename(sound_file_path))[0] + "_CENSORED." + out_format
    edited_audio.export(output_audio_file, format=out_format)
    
    if media_type == "video": #replace original video audio with edited audio
        output_vid_path = os.path.splitext(os.path.basename(file_path))[0] + "_CENSORED" + os.path.splitext(os.path.basename(file_path))[1]
        replace_audio_in_video(file_path, output_audio_file, output_vid_path)
        print(f"Video edited succesfully. Saved to {output_vid_path}")
    
else:
    print("no inapproriate words found to censor. If this has false negatives, adding a transcript may improve accuracy. Note: small chunk sizes may lead to inaccurate results as well")