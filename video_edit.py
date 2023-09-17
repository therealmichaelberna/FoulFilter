import ffmpeg

def replace_audio_in_video(video_path, audio_path, output_path):
    # Input video file
    input_video = ffmpeg.input(video_path)

    # Input audio file (MP3)
    input_audio = ffmpeg.input(audio_path)

    # Replace the audio of the video with the input audio
    output = ffmpeg.output(input_video.video, input_audio.audio, output_path, vcodec="copy")

    # Run FFmpeg command
    ffmpeg.run(output, overwrite_output=True)