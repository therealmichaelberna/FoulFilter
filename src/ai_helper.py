import json
import re
import time
import logging
import os
import requests  # Added for local LLM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FoulFilter.aihelper")

# Load Config ("or" fallbacks: compose exports unset vars as empty strings)
AI_MODE = (os.getenv("AI_MODE") or "google").lower()
LOCAL_URL = os.getenv("LOCAL_LLM_URL") or "http://localhost:8080/v1/chat/completions"
LOCAL_MODELS_URL = LOCAL_URL.replace("/chat/completions", "/models")
LOCAL_MODEL = os.getenv("LOCAL_LLM_MODEL") or "gpt-oss-20b-Q4_K_M"

_google_client_instance = None


def _google_client():
    """Lazily build the Gemini client; returns None when unconfigured."""
    global _google_client_instance
    if not os.getenv("GOOGLE_API_KEY"):
        return None
    if _google_client_instance is None:
        try:
            from google import genai
        except ImportError:
            logger.warning("GOOGLE_API_KEY set but google-genai not installed.")
            return None
        _google_client_instance = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _google_client_instance


def llm_inference_local(prompt):
    """Talks to your llama-server router"""
    payload = {
        "model": LOCAL_MODEL,
        "messages": [
            {"role": "system", "content": "You are a video editor. Output ONLY JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "response_format": {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "start_index": {"type": "integer"},
                    "end_index": {"type": "integer"}
                },
                "required": ["reasoning", "start_index", "end_index"]
            }
        }
    }
    try:
        # Generous timeout: a cold router may load the model into VRAM first
        response = requests.post(LOCAL_URL, json=payload, timeout=240)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except requests.HTTPError as e:
        body = (e.response.text or "").lower() if e.response is not None else ""
        if e.response is not None and e.response.status_code == 400 and "not found" in body:
            # Server runs a different model than LOCAL_LLM_MODEL says - ask
            # it for its actual model id and retry once (self-healing).
            try:
                models = requests.get(LOCAL_MODELS_URL, timeout=15).json().get("data", [])
                if models:
                    payload["model"] = models[0]["id"]
                    logger.info("Local LLM: retrying with server model '%s'", payload["model"])
                    response = requests.post(LOCAL_URL, json=payload, timeout=240)
                    response.raise_for_status()
                    return response.json()['choices'][0]['message']['content'].strip()
            except Exception as retry_err:
                logger.error("Local LLM model discovery failed: %s", retry_err)
        logger.error(f"Local LLM Error: {e}")
        return "API_UNAVAILABLE"
    except Exception as e:
        logger.error(f"Local LLM Error: {e}")
        return "API_UNAVAILABLE"

def llm_inference_google(prompt, retries=3):
    """Existing Gemini logic"""
    client = _google_client()
    if not client:
        return "API_UNAVAILABLE"

    from google.genai import types

    model_id = "gemini-2.5-flash-lite"
    for i in range(retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=300,
                    response_mime_type="application/json",
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    ]
                )
            )
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()
            return "ERROR_OR_REFUSAL"
        except Exception as e:
            err_str = str(e).upper()
            if "429" in err_str or "503" in err_str or "DEMAND" in err_str:
                logger.warning(f"Gemini Busy (Attempt {i+1}): {e}")
                time.sleep(2 ** i)
                continue
            logger.error(f"Gemini API Error: {e}")
            break
    return "API_UNAVAILABLE"


SURGICAL_RULE = (
    "4. SURGICAL MODE: Widening is FORBIDDEN for this edit. Return "
    "start_index == end_index == the target index exactly, unless rule 5 applies."
)

PROMPT_TEMPLATE = """
    TASK: You are a video editor. Your goal is to remove profanity in the least noticable manner. You are supplied with a list of words and their timestamps.

    CRITICAL RULES:
    1. SURGICAL PRECISION: Cut ONLY the target word index unless it's a phrase-level rule.
    2. ADJECTIVE PROTECTION: Do not cut adjectives like "good", "great", "very", or "movie" even if they follow a swear.
    3. PHRASE RULES: If a target is part of an idiomatic phrase (e.g., "go to hell") or a stuttered thought (e.g., "just tell him to..."), cut the entire leading context back to the last clean break point.
    4. FALSE POSITIVES: If "hoe" is a tool or "bitch" is a dog, return -1 for the start and end.
    5. THEMATIC: For "gay/bisexual" content, remove the entire section.
    {extra_rule}

    EXAMPLES:
    - EX 1: SURGICAL REMOVAL (Mid-stream)
      Input: 0: I, 1: really, 2: think, 3: that, 4: this, 5: is, 6: a, 7: damn, 8: good, 9: movie, 10: from, 11: what, 12: I
      Target: "damn"
      Output: {{"start_index": 7, "end_index": 7}}

    - EX 2: THEMATIC PHRASE (Sexual Orientation - Seamless Erasure)
      Input: 0: was, 1: a, 2: garbage, 3: collector, 4: he, 5: turned, 6: homosexual, 7: at, 8: fourteen, 9: John, 10: was, 11: a, 12: an, 13: accountant, 14: who
      Target: "homosexuals"
      Output: {{"start_index": 4, "end_index": 8}}

    - EX 3: ACTION REMOVAL (Clean Narrative Jump)
      Input: 0: standing, 1: by, 2: the, 3: car, 4: then, 5: the, 6: two, 7: gay, 8: men, 9: kissed, 10: then, 11: they, 12: drove,  13: away
      Target: "gay"
      Output: {{"start_index": 5, "end_index": 9}}

    - EX 4: PLOT-CRITICAL PROFANITY (Action/Subject)
      Input: 0: security, 1: footage, 2: clearly, 3: shows, 4: how, 5: the, 6: bitch, 7: stole, 8: the, 9: keys, 10: before, 11: running, 12: away
      Target: "bitch"
      Output: {{"start_index": 5, "end_index": 6}}

    - EX 5: IDIOMATIC PHRASE (Complete thought)
      Input: 0: told, 1: him, 2: stay, 3: away, 4: but, 5: he, 6: wouldn't, 7: so, 8: go, 9: to, 10: hell, 11: is, 12: what, 13: I, 14: said, 15: John, 16: farted, 17: continuously
      Target: "hell"
      Output: {{"start_index": 8, "end_index": 10}}

    - EX 6: FALSE POSITIVE (Gardening/Tool)
      Input: 0: after, 1: the, 2: rain, 3: stopped, 4: the, 5: farmer, 6: grabbed, 7: a, 8: hoe, 9: to, 10: fix, 11: the, 12: flower, 13: beds
      Target: "hoe"
      Output: {{"start_index": -1, "end_index": -1}}

    - EX 7: MID-WINDOW THEMATIC REMOVAL (Full Context)
      Input: 0: yes, 1: i, 2: saw, 3: the, 4: new, 5: movie, 6: besides, 7: the, 8: part, 9: where, 10: the, 11: two, 12: gay, 13: men, 14: kissed, 15: what, 16: did, 17: you, 18: think, 19: of, 20: the, 21: movie, 22: overall
      Target: "gay"
      Output: {{
          "reasoning": "Removing the clause 'besides the part where the two gay men kissed' for a seamless transition between 'movie' and 'what'.",
          "start_index": 6,
          "end_index": 14
      }}

    - EX 8: IDIOMATIC PHRASE IN LARGE WINDOW
      Input: 0: and, 1: was, 2: and, 3: then, 4: he, 5: said, 6: that, 7: anyway, 8: it, 9: was, 10: crazy, 11: just, 12: tell, 13: him, 14: to, 15: go, 16: to, 17: hell, 18: because, 19: I, 20: don't, 21: care, 22: about
      Target: "hell"
      Output: {{
          "reasoning": "Target 'hell' is part of the idiom 'go to hell'. Cutting the entire phrase (indices 11-17) for a natural transition.",
          "start_index": 11,
          "end_index": 17
      }}

    CURRENT SEQUENCE:
    {indexed_text}
    TARGET WORD: "{target_word}"

    Return ONLY JSON in this format:
    {{
    "reasoning": "Brief explanation of why these indices were chosen",
    "start_index": int,
    "end_index": int
    }}
    """


def get_ai_smart_cut(context_window, target_word, center_index=None, allow_widening=True):
    """Ask the LLM to refine one Hit.

    Returns False (skip hit), None (keep original timestamps), or
    {cut_start, cut_end}. When allow_widening is False the result is clamped
    to the target word only (see ADR-0004).
    """
    indexed_text = "\n".join([f"{idx}: {w['word']}" for idx, w in enumerate(context_window)])

    if center_index is None:
        # Backwards compatibility: infer the target position from the text.
        head = detector_head(target_word)
        for idx, w in enumerate(context_window):
            if head and w.get("word", "").lower().strip() in head:
                center_index = idx
                break
        if center_index is None:
            center_index = len(context_window) // 2

    extra_rule = "" if allow_widening else SURGICAL_RULE
    prompt = PROMPT_TEMPLATE.format(
        extra_rule=extra_rule,
        indexed_text=indexed_text,
        target_word=target_word,
    )

    # DISPATCHER: Choose based on ENV setting
    if AI_MODE == "local":
        response_text = llm_inference_local(prompt)
    else:
        response_text = llm_inference_google(prompt)

    logger.info(f"AI ({AI_MODE}) response: {response_text}")

    # Handle the "Safe Fallback" logic
    if response_text and "NONE" in response_text.upper():
        return False # Explicit skip

    if not response_text or response_text in ["API_UNAVAILABLE", "ERROR_OR_REFUSAL"]:
        return None # Use original timestamps

    try:
        json_match = re.search(r"\{[\s\S]*?\}", response_text)
        data = json.loads(json_match.group(0))
        return apply_smart_cut_indices(
            data,
            context_window,
            center_index=center_index,
            allow_widening=allow_widening,
        )
    except Exception as e:
        logger.error(f"Mapping failed: {e}")
        return None


def detector_head(target_word):
    """Token set a context word must match to be the target's first word."""
    tokens = re.findall(r"[a-z0-9']+", (target_word or "").lower())
    return set(tokens) or None


def apply_smart_cut_indices(data, context_window, center_index, allow_widening=True):
    """Map LLM indices to timestamps. Pure function; unit-tested."""
    s_idx = int(data["start_index"])
    e_idx = int(data["end_index"])

    if s_idx == -1 or e_idx == -1:
        return False  # Tells the pipeline to skip this word

    if not allow_widening:
        s_idx = e_idx = center_index

    # Bounds Safety
    s_idx = max(0, min(s_idx, len(context_window) - 1))
    e_idx = max(0, min(e_idx, len(context_window) - 1))

    # Mapping: Convert the AI's chosen indices back to timestamps
    return {
        "cut_start": context_window[s_idx]["start"],
        "cut_end": context_window[e_idx]["end"]
    }
