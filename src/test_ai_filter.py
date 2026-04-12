import os

import pytest
from ai_helper import get_ai_smart_cut

# These scenarios exercise the real LLM (see ADR-0004). They are opt-in so
# unit runs stay deterministic: start an LLM at LOCAL_LLM_URL, then
#   RUN_LIVE_LLM_TESTS=1 pytest src
requires_api_key = pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS", "").lower() not in ("1", "true", "yes"),
    reason="opt-in: set RUN_LIVE_LLM_TESTS=1 (LLM at LOCAL_LLM_URL) to run",
)


@pytest.mark.parametrize("test_case", [
    {
        "name": "Seamless Word Removal",
        "target": "damn",
        "context": [
            {"word": "That", "start": 1.0, "end": 1.2},
            {"word": "is", "start": 1.2, "end": 1.4},
            {"word": "a", "start": 1.4, "end": 1.5},
            {"word": "damn", "start": 1.5, "end": 1.9},
            {"word": "good", "start": 1.9, "end": 2.2},
            {"word": "movie", "start": 2.2, "end": 2.5}
        ],
        "eval": lambda res: res is not None and res["cut_start"] >= 1.4 and res["cut_end"] <= 1.9
    },
{
        "name": "Pointless Phrase (Messy Context Window long)",
        "target": "hell",
        "context": [
            # 11 words of leading "noise" (Incomplete thoughts)
            {"word": "and", "start": 2.0, "end": 2.5},        # 0
            {"word": "was", "start": 2.5, "end": 3.0},        # 1
            {"word": "and", "start": 3.0, "end": 3.2},        # 2
            {"word": "then", "start": 3.2, "end": 3.4},       # 3
            {"word": "he", "start": 3.4, "end": 3.5},         # 4
            {"word": "said", "start": 3.5, "end": 3.8},       # 5
            {"word": "that", "start": 3.8, "end": 4.0},       # 6
            {"word": "anyway", "start": 4.0, "end": 4.3},     # 7
            {"word": "it", "start": 4.3, "end": 4.5},         # 8
            {"word": "was", "start": 4.5, "end": 4.7},        # 9
            {"word": "crazy", "start": 4.7, "end": 5.0},      # 10
            # THE TARGET PHRASE (Starts at Index 11)
            {"word": "just", "start": 5.0, "end": 5.2},       # 11  - Phrase start for "just tell him to go to hell"
            {"word": "tell", "start": 5.2, "end": 5.3},       # 12            
            {"word": "him", "start": 5.3, "end": 5.4},        # 13
            {"word": "to", "start": 5.4, "end": 5.5},         # 14
            {"word": "go", "start": 5.5, "end": 5.7},         # 15
            {"word": "to", "start": 5.7, "end": 5.8},         # 16
            {"word": "hell", "start": 5.8, "end": 6.1},       # 17 - TARGET            
            # 7 words of trailing "noise" (To reach end of window)
            {"word": "because", "start": 6.1, "end": 6.4},    # 18
            {"word": "I", "start": 6.4, "end": 6.5},          # 19
            {"word": "don't", "start": 6.5, "end": 6.7},      # 20
            {"word": "care", "start": 6.7, "end": 7.0},       # 21
            {"word": "about", "start": 7.0, "end": 7.2},      # 22
        ],
        # The AI should return indices 11 through 17. 
        # Mapping those back to floats: 5.3 (start of 'him') to 6.1 (end of 'hell').
        "eval": lambda res: res is not None and res["cut_start"] == 5.0 and res["cut_end"] == 6.1
    },
    {
        "name": "False Positive",
        "target": "hoe",
        "context": [
            {"word": "He", "start": 10.0, "end": 10.2},
            {"word": "used", "start": 10.2, "end": 10.4},
            {"word": "a", "start": 10.4, "end": 10.5},
            {"word": "hoe", "start": 10.5, "end": 10.8},
            {"word": "to", "start": 10.8, "end": 11.0},
            {"word": "garden", "start": 11.0, "end": 11.4}
        ],
        "eval": lambda res: res is False
    },
    {
        "name": "Plot Critical Fact",
        "target": "bitch",
        "context": [
            {"word": "The", "start": 15.0, "end": 15.2},
            {"word": "bitch", "start": 15.2, "end": 15.6},
            {"word": "stole", "start": 15.6, "end": 15.9},
            {"word": "the", "start": 15.9, "end": 16.1},
            {"word": "keys", "start": 16.1, "end": 16.4}
        ],
        "eval": lambda res: res is not None and res["cut_end"] <= 15.6
    },
    {
        "name": "Seamless Word Removal (Mid-Sentence)",
        "target": "damn",
        "context": [
            {"word": "...and", "start": 0.5, "end": 0.7},
            {"word": "honestly", "start": 0.7, "end": 1.0},
            {"word": "I", "start": 1.0, "end": 1.1},
            {"word": "think", "start": 1.1, "end": 1.3},
            {"word": "it's", "start": 1.3, "end": 1.5},
            {"word": "a", "start": 1.5, "end": 1.6},
            {"word": "damn", "start": 1.6, "end": 1.9}, # Index 6
            {"word": "shame", "start": 1.9, "end": 2.2},
            {"word": "that", "start": 2.2, "end": 2.4},
            {"word": "we", "start": 2.4, "end": 2.6},
            {"word": "missed", "start": 2.6, "end": 2.9},
            {"word": "the", "start": 2.9, "end": 3.1},
            {"word": "bus", "start": 3.1, "end": 3.4}
        ],
        "eval": lambda res: res is not None and res["cut_start"] == 1.6 and res["cut_end"] == 1.9
    },
    {
        "name": "Pointless Phrase (Messy Context short)",
        "target": "hell",
        "context": [
            {"word": "anyway", "start": 4.5, "end": 4.8},
            {"word": "just", "start": 4.8, "end": 5.0},
            {"word": "tell", "start": 5.0, "end": 5.2},
            {"word": "him", "start": 5.2, "end": 5.3},
            {"word": "to", "start": 5.3, "end": 5.4},
            {"word": "go", "start": 5.4, "end": 5.6},
            {"word": "to", "start": 5.6, "end": 5.7},
            {"word": "hell", "start": 5.7, "end": 6.0}, # Index 7
            {"word": "if", "start": 6.0, "end": 6.2},
            {"word": "he", "start": 6.2, "end": 6.3},
            {"word": "keeps", "start": 6.3, "end": 6.6},
            {"word": "calling", "start": 6.6, "end": 7.0},
            {"word": "you", "start": 7.0, "end": 7.2}
        ],
        # AI should identify the core idiomatic phrase 'go to hell'
        "eval": lambda res: res is not None and (res["cut_start"] == 5.4 or res["cut_start"] == 5.3 or res["cut_start"] == 4.8) and res["cut_end"] == 6.0
    },
    {
        "name": "Homo Removal (Full 23-word Context)",
        "target": "gay",
        "context": [
            # 11 words BEFORE the target "gay"
            {"word": "yes", "start": 10.0, "end": 10.1},      # 0
            {"word": "i", "start": 10.1, "end": 10.2},        # 1
            {"word": "saw", "start": 10.2, "end": 10.4},      # 2
            {"word": "the", "start": 10.4, "end": 10.5},      # 3
            {"word": "new", "start": 10.5, "end": 10.7},      # 4
            {"word": "movie", "start": 10.7, "end": 11.0},    # 5
            {"word": "besides", "start": 11.0, "end": 11.4},  # 6 - PHRASE START
            {"word": "the", "start": 11.4, "end": 11.5},      # 7
            {"word": "part", "start": 11.5, "end": 11.8},     # 8
            {"word": "where", "start": 11.8, "end": 12.1},    # 9  
            {"word": "the", "start": 12.1, "end": 12.2},      # 10
            {"word": "two", "start": 12.2, "end": 12.4},      # 11
            
            # THE TARGET WORD (Index 12)
            {"word": "gay", "start": 12.4, "end": 12.7},      # 12 - TARGET
            
            # 10 words AFTER the target "gay" (to complete the 23-word window)
            {"word": "men", "start": 12.7, "end": 12.9},      # 13
            {"word": "kissed", "start": 12.9, "end": 13.5},   # 14 - PHRASE END
            {"word": "what", "start": 13.5, "end": 13.7},     # 15
            {"word": "did", "start": 13.7, "end": 13.8},      # 16
            {"word": "you", "start": 13.8, "end": 14.0},      # 17
            {"word": "think", "start": 14.0, "end": 14.3},    # 18
            {"word": "of", "start": 14.3, "end": 14.4},       # 19
            {"word": "the", "start": 14.4, "end": 14.6},      # 20
            {"word": "movie", "start": 14.6, "end": 14.9},    # 21
            {"word": "overall", "start": 14.9, "end": 15.3}   # 22
        ],
        # AI should return indices 6 through 14.
        # Mapping to: 11.8 (start of 'where') to 13.5 (end of 'kissed').
        "eval": lambda res: res is not None and res["cut_start"] == 11.0 and res["cut_end"] == 13.5
    }
])
@requires_api_key
def test_smart_cut_scenarios(test_case):
    """
    Tests the AI's ability to distinguish between seamless cuts, 
    phrase removals, and false positives.
    """
    result = get_ai_smart_cut(test_case["context"], test_case["target"])
    
    # We use a lambda evaluator to allow for flexible validation logic
    assert test_case["eval"](result), f"Failed test case: {test_case['name']}. Got: {result}"
