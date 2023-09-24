import sys
import os
import subprocess
import ffmpeg
#import time
#from pydub.generators import Sine
#from pydub import AudioSegment
#from pydub.silence import split_on_silence

#def generate_replacement_sound(duration_ms, bleep):
#    # Generate a beep sound with the specified duration in milliseconds
#    if bleep is True:
#        frequency = 1000  # 1000 Hz
#        tone = Sine(frequency)
#        replacement_sound = tone.to_audio_segment(duration=duration_ms)
#    else:
#        replacement_sound = AudioSegment.silent(duration=duration_ms)
#
#    return replacement_sound

#def censor_audio(audio_in, start_time, stop_time, bleep):
#    # Calculate the start and stop time in milliseconds
#    start_ms = start_time * 1000
#    stop_ms = stop_time * 1000
#
#    replacement_duration_ms = stop_ms - start_ms
#    replacement_sound = generate_replacement_sound(replacement_duration_ms, bleep)
#
#    part1 = audio_in[:start_ms]
#    part2 = replacement_sound
#    part3 = audio_in[stop_ms:]
#    # Split the audio into three parts: before, censor beep, and after
#
#    # Concatenate the parts together
#    edited_audio = part1 + part2 + part3
#    return edited_audio
#
#    # Export the edited audio to a new MP3 file
#    #edited_audio.export(output_file, format="mp3")

def delete_audio_using_word_match_list(audio_file_path, match_list, bleep=False, delete=False):

    probe = ffmpeg.probe(audio_file_path)
    audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)

    if audio_stream:
        sample_rate = int(audio_stream['sample_rate'])
        print(f"Sample rate: {sample_rate} Hz")
    else:
        print(f"Error, no audio stream found in the input file {audio_file_path}.")
        sys.exit(1)

    print(f"Deleting bad words from {audio_file_path}")
    #print(f"Loading The Audio File")
    #audio = AudioSegment.from_mp3(audio_file_path)
    #print(f"Audio File Loaded Successfully")
    #total_iterations = len(match_list)

    if bleep:
        out_format = "mp3"
    else:
        # use original format
        out_format = os.path.splitext(os.path.basename(audio_file_path))[1]

    out_file = os.path.splitext(os.path.basename(audio_file_path))[0] + "_CENSORED" + out_format

    #begin_time = time.time()# log start time so that we can estimate remaining time

    if bleep:
        # Define an array of sine filters with different frequencies and durations
        sine_filters = []

        for index,match in enumerate(match_list):
            start_time = match['start']
            end_time = match['end']
            duration = end_time - start_time
            sine_filters.append(f"frequency: 1000, duration: {duration}, start_time: {start_time}")
        #sine_filters = [
        #    {"frequency": 1000, "duration": 0.5, "start_time": 2.0},
        #    {"frequency": 2000, "duration": 0.3, "start_time": 5.0},
        #    {"frequency": 500, "duration": 0.2, "start_time": 8.0},
        #]

        # Create the filtergraph string with adelay and sine filters
        filtergraph = []
        for index, sine in enumerate(sine_filters):
            adelay = f"adelay={int(sine['start_time'] * 1000)}|{int(sine['start_time'] * 1000)}"
            sine_filter = f"sine=frequency={sine['frequency']}:duration={sine['duration']}[sine{index}]"
            filtergraph.append(f"[0:a]{adelay}[sine{index}]")

        (
            ffmpeg.input(audio_file_path)
            .output(
                filter_complex,
                out_file
            )
            .run()
        )

        # Mix the audio streams using amix
        filter_complex = ";".join(filtergraph)
        filter_complex += f";{''.join([f'[sine{i}]' for i in range(len(sine_filters))])}amix=inputs={len(sine_filters)}:duration=shortest"

    elif delete:
        concat = ""
        n = 0
        previous_end_time = None  # Initialize to None for the first iteration
        filter_complex = ""

        for index,match in enumerate(match_list):
            start_time = match['start']
            end_time = match['end']
            start_sample = round(sample_rate * start_time)
            end_sample = round(sample_rate * end_time)

            #first one, we start from the beginning
            if index == 0:
                # atrim extracts a portion of audio for usage. We extract the audio before the word until the word
                filter_complex += f'[0:a]atrim=start=0:end_sample={start_sample}[clip{index}_1],'
                n += 1
                concat += f"[clip{index}_1]"

            if index == (len(match_list) - 1):#on the last iteration start from the end of our word and go until the end
                filter_complex += f'[0:a]atrim=start_sample={end_sample}[clip{index}_2],'
                n += 1
                concat += f"[clip{index}_2]"

            #any regular detection that isn't the first or last
            if index != 0 and index != (len(match_list) - 1):
                previous_index = index - 1
                previous_end_time = match_list[previous_index]['end']
                previous_end_sample = round(sample_rate * previous_end_time)
                filter_complex += f'[0:a]atrim=start_sample={previous_end_sample}:end_sample={start_sample}[clip{index}_1],' #cut from previous end until next start
                n += 1
                #filter_complex.append(f'[0:a]atrim=start_sample={previous_end_sample}:end_sample={start_sample},asetpts=PTS-STARTPTS[clip{index}_1]')#cut from previous end until next start
                concat += f"[clip{index}_1]"

        filter_complex += f'{concat}concat=n={n}:v=0:a=1[out]'
        print(f"ffmpeg complex filter: '{filter_complex}'")
        #ffmpeg.input(audio_file_path).filter(filter_complex, map="[out]").output(out_file).run()
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", audio_file_path,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            out_file
        ]
        #print(f"\n\n {ffmpeg_cmd}")
        subprocess.run(ffmpeg_cmd, check=True)

    else:
        # Initialize the filter complex
        #filter_complex = []
        filter_complex = ""

        #curr_edit_time_samples=0#tracks where we last left off with copying audio from

        #print(f"removing {len(match_list)} bad words")
        # Create a filter chain for each bad word match
        for index,match in enumerate(match_list):
            #bad_word = match['word']
            start_time = match['start']
            end_time = match['end']
            #start_sample = round(sample_rate * start_time)
            #end_sample = round(sample_rate * end_time)

            # on the last iteration, no comma
            if index == (len(match_list) - 1):
                filter_complex += f"volume=enable='between(t,{start_time},{end_time})':volume=0"
            else:
                filter_complex += f"volume=enable='between(t,{start_time},{end_time})':volume=0,"


        # Concatenate all the segments together to create the final output
        #print(f"ffmpeg complex filter: '{filter_complex}'")
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", audio_file_path,
            "-filter_complex", filter_complex,
            #"-map", "[out]",
            out_file
        ]
        #print(f"\n\n {ffmpeg_cmd}")
        subprocess.run(ffmpeg_cmd, check=True)

    ########### pydub method ######################
    #####for index, match in enumerate(match_list):
    #####    current_time = time.time()
    #####    elapsed_time = current_time - start_time
    #####    average_time_per_iteration = elapsed_time / (index + 1)
    #####    # Calculate the estimated time remaining
    #####    iterations_remaining = len(match_list) - (index + 1)
    #####    estimated_time_remaining = iterations_remaining * average_time_per_iteration
    #####    print(f"Removing bad word {index + 1} of {len(match_list)} - Elapsed Time: {elapsed_time:.2f} seconds - Estimated Time Remaining: {estimated_time_remaining:.2f} seconds", end='\r')
    #####
    #####    bad_word = match.get('word') # get returns value of specified key
    #####    start = float(match.get('start'))
    #####    end = float(match.get('end'))
    #####    #print(f"removing {bad_word} found between {start} and {end}")
    #####    audio = censor_audio(audio, start, end, bleep)

    #for match in match_list:
    #    bad_word = match.get('word') # get returns value of specified key
    #    start = float(match.get('start'))
    #    end = float(match.get('end'))
    #    #print(f"removing {bad_word} found between {start} and {end}")
    return out_file
