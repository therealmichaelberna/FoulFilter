import json
import re
import time
import logging
import os
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FoulFilter.aihelper")

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def llm_inference(prompt, retries=3):
    model = genai.GenerativeModel('gemini-3-flash')
    for i in range(retries):
        try:
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig( temperature=0.1, max_output_tokens=150,))
            return response.text.strip()
        except Exception as e:
            if "429" in str(e): # Rate limit hit
                time.sleep(2 ** i) # Exponential backoff (2s, 4s, 8s)
                continue
            logger.error(f"Gemini Error: {e}")
            break
    return "NONE"

def get_ai_smart_cut(context_window, target_word):
    # Prepare a numbered list for the LLM to choose from
    # e.g., "0: oh, 1: ****, 2: he, 3: whispered"
    indexed_text = "\n".join([f"{idx}: {w['word']}" for idx, w in enumerate(context_window)])
    
    prompt = f"""
    You are a editor who removes immorality. I have a sequence of words where one is a swear word.
    I want to remove the swear word SEAMLESSLY. 
    If the sentence is short or the swear is the core of the phrase, suggest removing the whole phrase.
    We don't want to remove anything that actually affects the storyline.
    If the entire phrase is somehow critical to the story, just remove the swear or smallest logical portion of the phrase.
    Please also remove anything sexual or referring to homosexuality.
    
    WORD SEQUENCE:
    {indexed_text}
    
    TARGET WORD: "{target_word}"
    
    TASK:
    1. Determine if this is a false positive. If it is NOT a swear, return "NONE".
    2. If it IS a swear, identify the START index and END index of the words to be removed to make the cut sound natural.
    
    Return ONLY a JSON object: {{"start_index": int, "end_index": int}} or "NONE".
    """

    response_text = llm_inference(prompt)
    
    try:
        # Example of processing the response
        if "NONE" in response_text.upper():
            return None
        
        clean_json = re.sub(r"```json|```", "", response_text).strip()
        data = json.loads(clean_json)

        data = json.loads(response_text)
        s_idx = data["start_index"]
        e_idx = data["end_index"]
        
        return {
            "cut_start": context_window[s_idx]["start"],
            "cut_end": context_window[e_idx]["end"]
        }
    except Exception as e:
        logger.error(f"Smart Cut failed: {e}")
        return None # Or return the original word's timestamps as a fallback
