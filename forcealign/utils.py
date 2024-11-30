import re
import random


def alphabetical(text):
    return re.sub(r"[^a-zA-Z\s]", "", text)


def get_breath_idx(transcript):
    """Detect where breaths might occur."""
    transcript = transcript.replace("—", " ")
    transcript = alpha_with_punct(transcript).upper().split()
    idxs = []
    for i in range(len(transcript) - 1):
        if "," in transcript[i]:
            idxs.append(i + 1)
        elif "." in transcript[i] and random.choice([True, False, False]):
            idxs.append(i + 1)
    return idxs


def alpha_with_punct(text):
    return re.sub(r"[^a-zA-Z\s,.]", "", text)
