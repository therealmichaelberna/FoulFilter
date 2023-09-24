import os
import math
import time
import fnmatch
import ffmpeg
import soundfile as sf

def split_for_analysis(input_file, output_dir, segment_length_s):
    print(f"splitting audio file into smaller pieces of {segment_length_s} seconds each for easier analysis")
    #segment_length_ms = segment_length_s * 1000
    input_stream = ffmpeg.input(input_file)
    audio = input_stream.audio

    with sf.SoundFile(input_file) as audio_file:
        total_duration_s = len(audio_file) / audio_file.samplerate
    ##42021 seconds

    segment_file_paths = []

    print(f"total duration: {total_duration_s}")
    # Calculate the number of segments needed
    num_segments = math.ceil(total_duration_s / segment_length_s)
    #print(f"num_segments = {total_duration_s}/{segment_length_s}")

    print(f"audio file will be split into {num_segments} segments with a max length of {segment_length_s} seconds each")


    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    timer_start_time = time.time()# log start time so that we can estimate remaining time

    for i in range(num_segments):
        start_time = i * segment_length_s
        output_file = os.path.join(output_dir, f'segment_{i}.wav')
        #print(f"converting file {i+1} of {num_segments+1}")

        #last iteration. We remove t here so hopefully it just does whatever is left
        if i == num_segments -1:
            (
                ffmpeg
                .output(audio, output_file, ss=start_time, loglevel='error', format='wav', acodec='pcm_s16le', ar=16000, ac=1)
                .overwrite_output()
                .run()
            )

        else:
            (
                ffmpeg
                .output(audio, output_file, ss=start_time, t=segment_length_s, loglevel='error', format='wav', acodec='pcm_s16le', ar=16000, ac=1)
                .overwrite_output()
                .run()
            )

        current_time = time.time()
        elapsed_time = current_time - timer_start_time
        average_time_per_iteration = elapsed_time / (i + 1)
        # Calculate the estimated time remaining
        iterations_remaining = num_segments - (i + 1)
        #print(f"\n num_segments: {num_segments} average_time_per_iteration: {average_time_per_iteration} elapsed_time: {elapsed_time} timer_start_time: {timer_start_time} current_time {current_time}")
        estimated_time_remaining = iterations_remaining * average_time_per_iteration
        
        if estimated_time_remaining > 3600:
            hours_remaining = estimated_time_remaining/3600
            print(f"Iteration {i + 1}/{num_segments} - Elapsed Time: {elapsed_time:.2f} seconds - Estimated Time Remaining: {hours_remaining:.2f} hours", end='\r')
        elif estimated_time_remaining > 60:
            minutes_remaining = estimated_time_remaining/60
            print(f"Iteration {i + 1}/{num_segments} - Elapsed Time: {elapsed_time:.2f} seconds - Estimated Time Remaining: {minutes_remaining:.2f} minutes", end='\r')
        else:
            print(f"Iteration {i + 1}/{num_segments} - Elapsed Time: {elapsed_time:.2f} seconds - Estimated Time Remaining: {estimated_time_remaining:.2f} seconds", end='\r')

        segment_file_paths.append(output_file)

    end_time = time.time()
    total_elapsed_time = end_time - start_time
    print(f"\nTotal Time Taken to Split Files: {total_elapsed_time:.2f} seconds")

    return segment_file_paths

def find_split_files(search_directory):#search our temp directory and locate all segment files and place them in an array
    pattern = "segment_*.wav"
    matching_files = []
    all_files = os.listdir(search_directory)
    matching_files = fnmatch.filter(all_files, pattern)
    matching_files.sort()
    matching_files_full_path = [search_directory + "/" + item for item in matching_files]
    return matching_files_full_path