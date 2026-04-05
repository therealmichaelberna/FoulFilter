""" word_finder.py Word finder helps to locate swear words and generate a cutlist"""
#!/usr/bin/env python3
import time
import wave
import re
import os
import json
import sys
import logging
import string
from bs4 import BeautifulSoup
import difflib

# Load Environment Config
AI_ENHANCE = os.getenv("AI_ENHANCE", "False").lower() == "true" # == true helps cast it to a bool value
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ENGINE_TYPE = os.getenv("TRANSCRIPTION_ENGINE", "whisper")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FoulFilter")

# Initialization of Engines and Helpers
WhisperEngine = None
if ENGINE_TYPE == "whisper":
    try:
        from whisper_handler import WhisperEngine
    except ImportError:
        logger.error("Whisper selected but whisper_handler.py not found.")

process_vosk = None
if ENGINE_TYPE == "vosk":
    try:
        from vosk_handler import process_vosk
    except ImportError:
        logger.error("Vosk selected but vosk_handler.py not found.")

get_ai_smart_cut = None
if len(GOOGLE_API_KEY) > 0:
    if AI_ENHANCE:
        try:
            from ai_helper import get_ai_smart_cut
            logger.info("AI Enhancement module loaded successfully.")
        except ImportError:
            logger.error("AI Enhance enabled but ai_helper.py missing.")
else:
    if AI_ENHANCE:
        logger.warning("AI_ENHANCE is True but GOOGLE_API_KEY is missing. Feature disabled.")

def update_job_status(job_id, data):
    if not job_id:
        return
    # Use the shared data volume so FastAPI can see it
    status_file = f"/app/data/jobs/{job_id}.json"
    os.makedirs("/app/data/jobs", exist_ok=True)
    with open(status_file, "w") as f:
        json.dump(data, f)


def find_all_words_in_all_segments(
    analysis_files, tmp_dir, transcript_path, job_id=None
):
    # INITIALIZE ENGINE ONCE
    whisper_engine = None
    if WhisperEngine:
        # Load the weights into VRAM/RAM once here
        whisper_engine = WhisperEngine()

    discovered_words_list = []
    timer_start_time = time.time()
    cur_segment_time = 0
    progress_percent = 0
    estimated_time_remaining = 0
    total_segments = len(analysis_files)

    for i in range(total_segments):
        # 1. Update the Job Status (External JSON for FastAPI)
        update_job_status(
            job_id,
            {
                "status": "processing",
                "progress_percent": progress_percent,
                "eta": f"{int(estimated_time_remaining)}s",
                "current_step": f"Analyzing segment {i + 1}/{total_segments}",
            },
        )

        if progress_percent % 10 == 0:
            logger.info(
                f"Progress: {progress_percent}% - ETA: {estimated_time_remaining:.2f}s"
            )

        # 2. Progress Logging (Internal Console)
        # We use sys.stdout for the \r effect because logger doesn't support it
        sys.stdout.write(
            f"\r[Job {job_id}] Progress: {progress_percent}% - ETA: {estimated_time_remaining:.2f}s"
        )
        sys.stdout.flush()

        # 3. Process the segment
        # words_discovered, segment_len = find_words_in_segment(analysis_files[i])
        words_discovered, segment_len = find_words_in_segment(
            analysis_files[i], engine=whisper_engine
        )

        for item in words_discovered:
            item["start"] += cur_segment_time
            item["end"] += cur_segment_time

        if words_discovered:
            discovered_words_list.extend(words_discovered)

        cur_segment_time += segment_len

        # 4. CALCULATE PROGRESS & ETA for the NEXT iteration
        current_time = time.time()
        elapsed_time = current_time - timer_start_time
        avg_time_per_iteration = elapsed_time / (i + 1)
        iterations_remaining = total_segments - (i + 1)
        estimated_time_remaining = iterations_remaining * avg_time_per_iteration
        progress_percent = int(((i + 1) / total_segments) * 100)

    print(
        f"\nTotal Time Taken to Analyze Files: {time.time() - timer_start_time:.2f} seconds"
    )

    # Final cleanup of transcript matching
    if transcript_path:
        discovered_words_list = improve_match_with_transcript(
            discovered_words_list, pre_process_transcript(transcript_path)
        )

    return discovered_words_list


def find_words_in_segment(audio_file_path, engine=None):
    try:
        with wave.open(audio_file_path, "rb") as wf:
            audio_length_seconds = wf.getnframes() / wf.getframerate()
    except Exception as e:
        logger.error(f"Could not read audio file {audio_file_path}: {e}")
        return [], 0

    # 1. Primary Engine (Whisper)
    if ENGINE_TYPE == "whisper" and engine:
        try:
            return engine.process_segment(audio_file_path), audio_length_seconds
        except Exception as e:
            logger.error(f"WhisperX engine failed: {e}")
    # 2. Fallback/Alternative (Vosk)
    if ENGINE_TYPE == "vosk" and process_vosk:
        try:
            return process_vosk(audio_file_path), audio_length_seconds
        except Exception as e:
            logger.error(f"Vosk engine failed: {e}")

    logger.error("No functional transcription engine found.")
    return [], audio_length_seconds


def pre_process_transcript(transcript_path):
    transcript_word_array = []
    # Compile regex once for speed
    clean_pattern = re.compile(r"^[\W]+|[\W]+$")

    if not os.path.exists(transcript_path):
        logger.error(f"Transcript at '{transcript_path}' not found.")
        return []

    with open(transcript_path, "r") as file:
        for line in file:
            words = line.split()
            for word in words:
                # Clean and lowercase in one go
                cleaned = clean_pattern.sub("", word.lower())
                if cleaned:
                    transcript_word_array.append(cleaned)

    return transcript_word_array


def improve_match_with_transcript(
    results_list, transcript_word_array
):  # attempts to improve accuracy of the match using transcript
    print("using transcript")

    improved_detection = []
    results_words_only_list = []  # only words not start, end, or anything else.

    for item in results_list:  # combine our results list with words only
        results_words_only_list.append(item.get("word"))

    # differ = difflib.Differ()

    html_table = difflib.HtmlDiff().make_table(
        fromlines=transcript_word_array,
        tolines=results_words_only_list,
        fromdesc="Transcript",
        todesc="Results",
        context=False,  # Show surrounding context
    )

    print(f"table:'{html_table}'")

    soup = BeautifulSoup(
        html_table, "html.parser"
    )  # Parse the HTML using BeautifulSoup
    # Find the table element by its class name
    table = soup.find("table", class_="diff")
    # Initialize an empty list to store the rows
    result_rows = []

    # Find all the rows in the table body
    rows = table.tbody.find_all("tr")

    # for row in rows:
    for i in range(0, len(rows), 1):
        cells = rows[i].find_all("td")

        # Check if the first cell is 'n' or 't'
        if cells[0].text.strip() == "n" or cells[0].text.strip() == "t":
            # Extract the data from the cells
            result_line_index = cells[4].text.strip()
            # if the word wasn't detected in our original result, we must continue anyways because we have no start and end
            if result_line_index == "":
                continue
            transcript = cells[2].text.strip()
            results = cells[5].text.strip()

            # Create a dictionary for the row data
            row_data = {
                "Transcript": transcript,
                "Results": results,
                "Res_index": result_line_index,
            }

            # Append the row data to the list
            result_rows.append(row_data)

    improved_detection = results_list

    for item in result_rows:
        results = item["Results"]
        transcript_word = item["Transcript"]
        res_index = item["Res_index"]
        # print(f"trans: {transcript_word} res: {results} res_index: {res_index}")
        try:
            list_index = int(res_index) - 1
            if 0 <= list_index < len(improved_detection):
                improved_detection[list_index]["word"] = transcript_word
        except (ValueError, IndexError):
            continue

    # print(f"improved detection: {improved_detection}")
    return improved_detection

def find_bad_words(discovered_words_list, bad_words):
    filtered_list = []
    clean_bad_words = [w.lower().strip() for w in bad_words]

    for i, item in enumerate(discovered_words_list):
        # We look at the word lowercase and WITHOUT punctuation for our check
        raw_word = item.get("word").lower()
        word_for_comparison = raw_word.strip(string.punctuation)

        # Check if the cleaned word is in our bad list
        if any(bad == word_for_comparison for bad in clean_bad_words):
            
            if AI_ENHANCE and get_ai_smart_cut:
                # We send the RAW context (with punctuation) to the AI
                # so it has the best chance of understanding the sentence.
                start_idx = max(0, i - 6)
                end_idx = min(len(discovered_words_list), i + 7)
                context_window = discovered_words_list[start_idx:end_idx]
                
                # word_for_comparison here ensures the AI knows exactly which word to check
                smart_cut = get_ai_smart_cut(context_window, word_for_comparison)

                if smart_cut is None:
                    logger.info(f"AI Enhancement: False positive detected for '{word_for_comparison}'. Skipping.")
                    continue
                
                item["start"] = smart_cut["cut_start"]
                item["end"] = smart_cut["cut_end"]
                
            filtered_list.append(item)
    return filtered_list
