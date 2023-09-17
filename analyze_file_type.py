import magic

def detect_media_type(file_path):
    mime = magic.Magic()
    file_mime = mime.from_file(file_path)
    #print(f"file mime: {file_mime}")
    if 'audio' in file_mime:
        return 'audio'
    elif 'video' in file_mime or 'MP4' in file_mime:
        return 'video'
    else:
        return 'unknown'