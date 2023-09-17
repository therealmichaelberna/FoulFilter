from pydub.generators import Sine
from pydub import AudioSegment
from pydub.silence import split_on_silence

def generate_replacement_sound(duration_ms, bleep):
    # Generate a beep sound with the specified duration in milliseconds    
    if bleep == True:
        frequency = 1000  # 1000 Hz
        tone = Sine(frequency)
        replacement_sound = tone.to_audio_segment(duration=duration_ms)
    else:
        replacement_sound = AudioSegment.silent(duration=duration_ms)
        
    return replacement_sound

def censor_audio(audio_in, start_time, stop_time, bleep):
    # Calculate the start and stop time in milliseconds
    start_ms = start_time * 1000
    stop_ms = stop_time * 1000
    
    replacement_duration_ms = stop_ms - start_ms
    replacement_sound = generate_replacement_sound(replacement_duration_ms, bleep)# if bleep is true, use bleep. Otherwise use silence
    
    part1 = audio_in[:start_ms]
    part2 = replacement_sound
    part3 = audio_in[stop_ms:]
    # Split the audio into three parts: before, censor beep, and after

    # Concatenate the parts together
    edited_audio = part1 + part2 + part3
    return edited_audio
    
    # Export the edited audio to a new MP3 file
    #edited_audio.export(output_file, format="mp3")

def delete_audio_using_word_match_list(audio_file_path, match_list, bleep):
    print(f"Deleting bad words from {audio_file_path}")
    audio = AudioSegment.from_mp3(audio_file_path)
    
    for match in match_list:
        bad_word = match.get('word') # get returns value of specified key
        start = float(match.get('start'))
        end = float(match.get('end'))
        #print(f"removing {bad_word} found between {start} and {end}")
        audio = censor_audio(audio, start, end, bleep)
    
    return audio