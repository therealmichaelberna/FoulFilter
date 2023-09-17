#!/usr/bin/env python3
import json
import wave
import sys
#import pylcs
import pickle
import re
from vosk import Model, KaldiRecognizer, SetLogLevel
import difflib
#import os
from bs4 import BeautifulSoup

def find_all_words_in_all_segments(analysis_files, tmp_dir, transcript_path):
    discovered_words_list = []
    #for file_to_analyze in analysis_files:
    
    trans_word_array = []
    
    if transcript_path:
        trans_word_array=pre_process_transcript(transcript_path)
        transcript_word_array_string = " ".join(trans_word_array)
    
    for i in range(len(analysis_files)):
        print(f"analyzing file {i+1} of {len(analysis_files)}")
        words_discovered = find_words_in_segment(analysis_files[i], tmp_dir, trans_word_array)

        if words_discovered:
            print("words have been discovered in this segment")
            discovered_words_list.extend(words_discovered)
        else:
            print("no words discovered in this segment")
    
    if transcript_path: #if transcript is specified, let's try to use it to improve our match
        print(f"transcript path is specified as {transcript_path}. Using this file to improve accuracy.")
        transcript_word_array = pre_process_transcript(transcript_path)
        discovered_words_list = improve_match_with_transcript(discovered_words_list, transcript_word_array)
    print("discovered word list: {discovered_words_list}")
    return discovered_words_list

def find_bad_words(discovered_words_list, bad_words):
    #search for bad words
    filtered_list = [item for item in discovered_words_list if any(word in item.get('word') for word in bad_words)] #multi words
    #print(filtered_list)
    return filtered_list

def find_words_in_segment(audio_file_path, tmp_dir, possible_words=None):
    print(f"identifying bad words in {audio_file_path}")
    # You can set log level to -1 to disable debug messages
    SetLogLevel(-1)
    #SetLogLevel(0)
    
    wf = wave.open(audio_file_path, "rb")

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        sys.exit(1)

    model = Model(lang="en-us")
    #model_path = "/media/sf_swear_remover_python_proj/vosk-model-en-us-0.22-lgraph"
    #model = Model(lang="en-us", model_path=model_path)
    results_list = []#this list holds all of our results
        #if possible_words and len(possible_words)>0:
        #    #print("using word whitelist")
        #    #print(f"list: {possible_words}")
        #    possible_words_string = '[' + ', '.join(['"{}"'.format(str(item)) for item in possible_words]) + ']'
        #    #print(f"possible words string: {possible_words_string}")
        #    #rec = KaldiRecognizer(model, wf.getframerate(), '["im", "convinced", "humans", "we", "cant", "handle", "starbucks", "theres", "too", "many", "options", "way", "too", "many", "options", "in", "fast", "food", "places", "theres", "an", "efficiency", "ill", "have", "a", "number", "three", "in", "starbucks", "people", "sound", "like", "brats", "theyre", "like", "okay", "lets", "see", "what", "does", "baby", "feel", "like", "sipping", "on", "ill", "have", "a", "venti", "half", "caf", "caramel", "macchiato", "with", "no", "sugar", "no", "dairy", "and", "no", "coffee"]')
        #    rec = KaldiRecognizer(model, wf.getframerate(), possible_words_string)
        #else:
        #    rec = KaldiRecognizer(model, wf.getframerate())
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    rec.SetPartialWords(True)

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            json_data = json.loads(rec.Result())
            
            if 'result' in json_data:
                results_list.extend(json_data['result'])
   
        
    #print(filtered_list)
    #results_list contains all unfiltered results along with timestamp.
    return results_list

def pre_process_transcript(transcript_path): #pre-proccess the transcript to remove all punctuation and convert it to an array
    transcript_word_array = []
    pattern = r'^[\W]+|[\W]+$'
    
    try:
        with open(transcript_path, 'r') as file:
            for line in file:
                # Split each line into words by whitespace
                words = line.split()
                transcript_word_array.extend(words)
    
    except FileNotFoundError:
        print(f"Transcript at '{transcript_path}' not found.")
    
    transcript_word_array_cleaned = []
    for item in transcript_word_array:
        lowercase_item = item.lower() #make it lowercase so that the diff can process it accurately
        transcript_word_array_cleaned.append(re.sub(pattern, '', lowercase_item))#replace special characters
    #transcript_word_array_cleaned = [re.sub(pattern, '', string) for string in transcript_word_array]
    #transcript_word_array_cleaned_double_quotes = [item.replace("'", "\"") for item in transcript_word_array_cleaned] #convert it to use all double quotes because it likes to mix it and that causes issues
    
    return transcript_word_array_cleaned

def improve_match_with_transcript(results_list, transcript_word_array): # attempts to improve accuracy of the match using transcript
    print("using transcript")
    
    improved_detection = []
    results_words_only_list = []#only words not start, end, or anything else.
    
    
    for item in results_list:#combine our results list with words only
        results_words_only_list.append(item.get('word'))

    differ = difflib.Differ()
    
    html_table = difflib.HtmlDiff().make_table(
        fromlines=transcript_word_array,
        tolines=results_words_only_list,
        fromdesc="Transcript",
        todesc="Results",
        context=False  # Show surrounding context
    )

    soup = BeautifulSoup(html_table, 'html.parser')# Parse the HTML using BeautifulSoup
    # Find the table element by its class name
    table = soup.find('table', class_='diff')
    # Initialize an empty list to store the rows
    result_rows = []
    
    # Find all the rows in the table body
    rows = table.tbody.find_all('tr')
    
    #for row in rows:
    for i in range(0, len(rows), 1):
        cells = rows[i].find_all('td')
        
        # Check if the first cell is 'n' or 't'
        if cells[0].text.strip() == 'n' or cells[0].text.strip() == 't':
            # Extract the data from the cells
            transcript = cells[2].text.strip()
            results = cells[5].text.strip()
            result_line_index = cells[4].text.strip()
            
            # Create a dictionary for the row data
            row_data = {
                'Transcript': transcript,
                'Results': results,
                'Res_index' : result_line_index
            }
            
            # Append the row data to the list
            result_rows.append(row_data)

    improved_detection = results_list
    
    for index, item in enumerate(result_rows):
        transcript_word = item["Transcript"]
        res_index = item["Res_index"]
        list_index = int(res_index)-1
        #print(f"Transcript: {transcript_word}, Res_index: {res_index}, List index: {list_index}")
        improved_detection[list_index]['word'] = transcript_word

    #print(f"improved detection: {improved_detection}")
    return improved_detection