from pydub import AudioSegment
import os
import math

def split_for_analysis(input_file, output_dir, segment_length_s):
    print("splitting audio file into smaller pieces for easier analysis")
    segment_length_ms = segment_length_s * 1000
    # Load the input audio file
    audio = AudioSegment.from_file(input_file)
    segment_file_paths = []

    # Calculate the total duration of the input audio in milliseconds
    total_duration_ms = len(audio)
    print(f"total duration: {total_duration_ms}")
    # Calculate the number of segments needed
    num_segments = math.ceil(total_duration_ms / segment_length_ms)
    
    
    print(f"audio file will be split into {num_segments} segments with a max length of {segment_length_s} seconds each")
        
        
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split the audio into segments and export them as 16-bit mono PCM WAV
    for i in range(num_segments):
        start_time = i * segment_length_ms
        end_time = (i + 1) * segment_length_ms
        segment = audio[start_time:end_time]
        output_file = os.path.join(output_dir, f"segment_{i + 1}.wav")
        segment = segment.set_channels(1)  # Convert to mono
        segment = segment.set_sample_width(2)  # 16-bit PCM
        segment.export(output_file, format="wav")
        segment_file_paths.append(output_file)
    
    return segment_file_paths