import os
import re
import time
import threading 

import torch
import torchaudio
from g2p_en import G2p
from pydub import AudioSegment
from pydub.playback import play
from dataclasses import dataclass


# phonemes + graphemes + words
phonemizer = G2p()
torch.random.manual_seed(0)

def relative_path(path:str):
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, path)


def alphabetical(text):
	return re.sub(r'[^a-zA-Z\s]', '', text)

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

	def __repr__(self):
		return f'{self.word}: {self.time_start} -- {self.time_end}(s)'


@dataclass
class Phoneme:
	phoneme: str
	time_start: float
	time_end: float

	def __repr__(self):
		return f'{self.phoneme}: {self.time_start} -- {self.time_end}(s)'


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
	def __init__(self, audio_file:str, txt_file:str):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.SPEECH_FILE = audio_file # TODO: add relative path
		self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
		self.model = self.bundle.get_model().to(self.device)
		self.labels = self.bundle.get_labels()
		self.dictionary = {c: i for i, c in enumerate(self.labels)}

		# Load the audio file 
		with torch.inference_mode():
			if self.SPEECH_FILE.rsplit('.', 1)[-1] == 'mp3':
				audio = AudioSegment.from_mp3(self.SPEECH_FILE)
				self.SPEECH_FILE = self.SPEECH_FILE.replace('mp3', 'wav')
				self.audio_format = 'wav'
				audio.export(self.SPEECH_FILE, format=self.audio_format)
				self.waveform, sr = torchaudio.load(self.SPEECH_FILE)
			else:
				self.audio_format = 'wav'
				self.waveform, sr = torchaudio.load(self.SPEECH_FILE)

				
			if sr != self.bundle.sample_rate:
				self.waveform = torchaudio.functional.resample(self.waveform, orig_freq=sr, new_freq=self.bundle.sample_rate)

			self.emissions, _ = self.model(self.waveform.to(self.device))
			self.emissions = torch.log_softmax(self.emissions, dim=-1)

		self.emission = self.emissions[0].cpu().detach()

		# Read the text file to a string 
		with open(txt_file) as file:
			self.raw_text = file.read()
			text = self.raw_text.replace('—',' ')
			text = alphabetical(text).upper().split()
			self.transcript = f'{"|".join(text)}|'

		self.tokens = [self.dictionary[c] for c in self.transcript]

		self.word_alignments = None
		self.phoneme_alignments = []


	def get_trellis(self):
		tokens = self.tokens
		emission = self.emission
		blank_id=0

		num_frame = emission.size(0)
		num_tokens = len(tokens)

		trellis = torch.zeros((num_frame, num_tokens))
		trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
		trellis[0, 1:] = -float("inf")
		trellis[-num_tokens + 1 :, 0] = float("inf")

		for t in range(num_frame - 1):
			trellis[t + 1, 1:] = torch.maximum(
				# Score for staying at the same token
				trellis[t, 1:] + emission[t, blank_id],
				# Score for changing to the next token
				trellis[t, :-1] + emission[t, tokens[1:]],
			)

		return trellis



	def backtrack(self, trellis):
		# trellis = self.trellis
		emission = self.emission
		tokens = self.tokens
		blank_id=0

		t, j = trellis.size(0) - 1, trellis.size(1) - 1
		path = [Point(j, t, emission[t, blank_id].exp().item())]
		while j > 0:
			# Should not happen but just in case
			assert t > 0

			# 1. Figure out if the current position was stay or change
			# Frame-wise score of stay vs change
			p_stay = emission[t - 1, blank_id]
			p_change = emission[t - 1, tokens[j]]

			# Context-aware score for stay vs change
			stayed = trellis[t - 1, j] + p_stay
			changed = trellis[t - 1, j - 1] + p_change

			# Update position
			t -= 1
			if changed > stayed:
				j -= 1

			# Store the path with frame-wise probability.
			prob = (p_change if changed > stayed else p_stay).exp().item()
			path.append(Point(j, t, prob))

		# Now j == 0, which means, it reached the SoS.
		# Fill up the rest for the sake of visualization
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

	# Merge words
	def merge_words(self, segments):
		separator="|"
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
		for word in word_segments:
			ratio = self.waveform.size(1) / trellis.size(0)
			start = int(ratio * word.start)
			end = int(ratio * word.end)
			time_start = round((start/self.bundle.sample_rate),3)
			time_end = round((end/self.bundle.sample_rate),3)

			phonemes = phonemizer(word.label)
			phoneme_duration = (time_end - time_start) / len(phonemes)

			start_phoneme = time_start
			for i,_ in enumerate(phonemes):
				end_phoneme = start_phoneme + phoneme_duration - 0.1
				phoneme = Phoneme(phoneme=phonemes[i], time_start=start_phoneme, time_end=end_phoneme)
				self.phoneme_alignments.append(phoneme)
				start_phoneme += phoneme_duration

			words.append(Word(word=word.label, phonemes=phonemes, time_start=time_start, time_end=time_end))

		self.word_alignments = words
		return words

	def review_alignment(self):
		audio = AudioSegment.from_file(self.SPEECH_FILE, format=self.audio_format)

		timers = []
		for word in self.word_alignments:
			timer = threading.Timer(word.time_start, print, args=[repr(word)])
			timers.append(timer)
			timer.start()
		play(audio)









