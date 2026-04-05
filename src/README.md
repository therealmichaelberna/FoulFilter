# FoulFilter
<img src="logo.png" width="300">

Automated Word Filtering of Audio/Video Files using Whisper (or VOSK) and Python.
## Docker service (recommended)

### Configuring
Create a copy of the example environment settings file `.env.example` and name it `.env`
Fill in selected options

```docker build -t foulfilter .```

```docker compse up -d``

### Usage
Swagger docs available at:
```http://localhost:8000/docs```
You can use swagger upload endpoint to upload files too, by using the upload function and clicking test it.
You can use status endpoint to check on the status. It will give you a download link when complete.

## Python (may still work, but no longer officially supported, please use docker service)
### Installation
#### Pre-requisites
- Python3
- Pip3
  
1. Download and unzip git repo
```cd foulfilter```

2. ```pip3 install -r requirements.txt```

### Usage
```cd foulfilter```

```python3 find_and_remove.py input_video_file.mp4 "bad_words.txt"``` (bad_words.txt should be a plain text file with one word per line)

##### Help:
```python3 find_and_remove.py -h```

##### Extra Parameters
**--bleep** use a beep instead of replacing detected bad words with silence

**--analysis_chunk_size=15** How many seconds to split files into when running through detection algorithm. Use this to lower system resource usage, especially with large files. Recommended over 15 seconds. Longer may be more accurate but will use more vram.

**--no_edit** Just analyze the file, don't edit

**--resume** Resume using previous analysis

**--transcript** Path to transcript (optional, can improve accuracy if transcript is correct) It will not help with missed detections, but can improve incorrect detections.

**--delete** delete words instead of replacing with silent audio
