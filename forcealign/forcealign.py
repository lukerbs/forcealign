import os
import re
import time
import threading
import random

import torch
import torchaudio
from g2p_en import G2p
from pydub import AudioSegment
from pydub.playback import play
from dataclasses import dataclass

from .transcriber import speech_to_text
from .utils import alphabetical, get_breath_idx

# Phonemes + graphemes + words
phonemizer = G2p()
torch.random.manual_seed(0)


# Utility functions
def relative_path(path: str):
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, path)


def alphabetical(text):
    return re.sub(r"[^a-zA-Z\s]", "", text)


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Word:
    word: str
    phonemes: list
    time_start: float
    time_end: float
    breath: bool

    def __repr__(self):
        return f"{self.word}: {self.time_start} -- {self.time_end}(s)"


@dataclass
class Phoneme:
    phoneme: str
    time_start: float
    time_end: float

    def __repr__(self):
        return f"{self.phoneme}: {self.time_start} -- {self.time_end}(s)"


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


class ForceAlign:
    def __init__(self, audio_file: str, transcript: str = None):
        """Turns an audio file with a transcript into a force alignment

        Args:
            audio_file (str): Path to an audio file of a person talking.
            transcript (str, optional): Text transcript. If None, transcript will be generated automatically.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SPEECH_FILE = audio_file
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().to(self.device)
        self.labels = self.bundle.get_labels()
        self.dictionary = {c: i for i, c in enumerate(self.labels)}

        # Load and preprocess audio
        self._load_audio()

        # Handle transcript
        if transcript is None:
            print("No transcript provided. Generating transcript using speech_to_text...")
            self.raw_text = speech_to_text(self.SPEECH_FILE)  # Use transcriber.speech_to_text
            print(f"Generated Transcript: {self.raw_text}")
        else:
            self.raw_text = transcript

        text = alphabetical(self.raw_text).upper().split()
        self.transcript = f'{"|".join(text)}|'
        self.tokens = [self.dictionary[c] for c in self.transcript]
        self.breath_idx = get_breath_idx(self.raw_text)
        self.word_alignments = None
        self.phoneme_alignments = []

    def _load_audio(self):
        """Load and preprocess the audio file."""
        with torch.inference_mode():
            if not os.path.exists(self.SPEECH_FILE):
                raise FileNotFoundError(f"Audio file not found: {self.SPEECH_FILE}")

            if self.SPEECH_FILE.lower().endswith(".mp3"):
                # Convert MP3 to WAV using pydub
                audio = AudioSegment.from_mp3(self.SPEECH_FILE)
                wav_path = self.SPEECH_FILE.rsplit(".", 1)[0] + ".wav"
                audio.export(wav_path, format="wav")
                self.SPEECH_FILE = wav_path

            self.waveform, sr = torchaudio.load(self.SPEECH_FILE)

            if sr != self.bundle.sample_rate:
                self.waveform = torchaudio.functional.resample(
                    self.waveform, orig_freq=sr, new_freq=self.bundle.sample_rate
                )

            self.emissions, _ = self.model(self.waveform.to(self.device))
            self.emissions = torch.log_softmax(self.emissions, dim=-1)
            self.emission = self.emissions[0].cpu().detach()

    def get_trellis(self):
        tokens = self.tokens
        emission = self.emission
        blank_id = 0

        num_frame = emission.size(0)
        num_tokens = len(tokens)

        trellis = torch.zeros((num_frame, num_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[-num_tokens + 1 :, 0] = float("inf")

        for t in range(num_frame - 1):
            trellis[t + 1, 1:] = torch.maximum(
                trellis[t, 1:] + emission[t, blank_id],
                trellis[t, :-1] + emission[t, tokens[1:]],
            )

        return trellis

    def backtrack(self, trellis):
        emission = self.emission
        tokens = self.tokens
        blank_id = 0

        t, j = trellis.size(0) - 1, trellis.size(1) - 1
        path = [Point(j, t, emission[t, blank_id].exp().item())]
        while j > 0:
            assert t > 0
            p_stay = emission[t - 1, blank_id]
            p_change = emission[t - 1, tokens[j]]
            stayed = trellis[t - 1, j] + p_stay
            changed = trellis[t - 1, j - 1] + p_change
            t -= 1
            if changed > stayed:
                j -= 1
            prob = (p_change if changed > stayed else p_stay).exp().item()
            path.append(Point(j, t, prob))

        while t > 0:
            prob = emission[t - 1, blank_id].exp().item()
            path.append(Point(j, t - 1, prob))
            t -= 1

        return path[::-1]

    def merge_repeats(self, path):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    self.transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments

    def merge_words(self, segments):
        separator = "|"
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words

    def inference(self):
        trellis = self.get_trellis()
        path = self.backtrack(trellis)
        segments = self.merge_repeats(path)
        word_segments = self.merge_words(segments)

        words = []
        idx = 0
        for word in word_segments:
            ratio = self.waveform.size(1) / trellis.size(0)
            start = int(ratio * word.start)
            end = int(ratio * word.end)
            time_start = round((start / self.bundle.sample_rate), 3)
            time_end = round((end / self.bundle.sample_rate), 3)

            phonemes = phonemizer(word.label)
            phoneme_duration = (time_end - time_start) / len(phonemes)

            start_phoneme = time_start
            for i, _ in enumerate(phonemes):
                end_phoneme = start_phoneme + phoneme_duration - 0.1
                phoneme = Phoneme(phoneme=phonemes[i], time_start=start_phoneme, time_end=end_phoneme)
                self.phoneme_alignments.append(phoneme)
                start_phoneme += phoneme_duration

            breath = idx in self.breath_idx
            words.append(
                Word(word=word.label, phonemes=phonemes, time_start=time_start, time_end=time_end, breath=breath)
            )
            idx += 1

        self.word_alignments = words
        return words

    def review_alignment(self):
        audio = AudioSegment.from_file(self.SPEECH_FILE, format="wav")

        timers = []
        for word in self.word_alignments:
            timer = threading.Timer(word.time_start, print, args=[repr(word)])
            timers.append(timer)
            timer.start()
        play(audio)
