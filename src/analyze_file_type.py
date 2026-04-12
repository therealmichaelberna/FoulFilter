"""analyze_file_type.py"""
import magic

def detect_media_type(file_path):
    # Quick extension override for audiobooks
    if file_path.lower().endswith(('.m4b', '.m4a')):
        return 'audio'

    mime = magic.Magic()
    file_mime = mime.from_file(file_path)
    
    # Catch lowercase or uppercase keywords, plus specific audiobook flags
    if any(x in file_mime.lower() for x in ['audio', 'audio book', 'aac']): 
        return 'audio'
    elif any(x in file_mime.lower() for x in ['video', 'mp4', 'quicktime']):
        return 'video'
    else:
        print(f"file mime: {file_mime}")
        return 'unknown'
