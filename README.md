# FoulFilter
Automated Word Filtering of Audio/Video Files using VOSK.

## Installation
### Pre-requisites
- Python3
- Pip3
  
1. Download and unzip git repo
```cd foulfilter```

2. ```pip3 install -r requirements.txt```

## Usage
```cd foulfilter```

```python3 find_and_remove.py input_video_file.mp4 "bad_words.txt"``` (bad_words.txt should be a plain text file with one word per line)

#### Help:
```python3 find_and_remove.py -h```

#### Extra Parameters
**--bleep** use a beep instead of replacing detected bad words with silence

**--analysis_chunk_size=15** How many seconds to split files into when running through detection algorithm. Use this to lower system resource usage, especially with large files.

**--no_edit** Just analyze the file, don't edit

**--resume** Resume using previous analysis

**--transcript** Path to transcript (optional, can improve accuracy if transcript is correct) It will not help with missed detections, but can improve incorrect detections.
