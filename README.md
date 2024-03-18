# ForceAlign 
ForceAlign is a Python library for forced alignment of English text to English Audio. You can use this library to get word or [phoneme](https://en.wikipedia.org/wiki/Phoneme)-level text alignments to English audio. In short, forced alignment is the process of identifying the specific time a word (or words) was spoken within an audio recording. ForceAlign supports forced alignment for .mp3 and .wav audio file formats.

For phoneme level text alignments, ForceAlign currently only supports the [ARPABET](https://en.wikipedia.org/wiki/ARPABET) phonetic transcription encoding. 

ForceAlign uses Pytorch's WAV2VEC2 pretrained model for acoustic feature extraction and can be ran on both CPU and CUDA GPU devices.

# Installation and Dependencies
1. Pip Install ForceAlign
	- `pip3 install forcealign`
2. Install ffmpeg
	- Mac: `brew install ffmpeg`
	- Linux: `sudo apt install ffmpeg`
	- Windows: Install from [ffmpeg.org](https://ffmpeg.org/download.html)

# Usage Examples
To use ForceAlign, instantiate a ForceAlign object instance with your specified audio file and corresponding text file. 

**Example 1: Getting Word-Level Text Alignments**
```
from forcealign import ForceAlign

# Provide path to audio_file and corresponding txt_file with audio transcript
align = ForceAlign(audio_file='./speech.mp3', txt_file='./speech.txt')

# Runs prediction and returns alignment results
words = align.inference() 

# Show predicted word-level alignments
for word in words:
	print(word.word) # The word spoken in audio at associated time
	print(word.time_start) # Time (seconds) the word starts in speech.mp3
	print(word.time_end) # Time (seconds) the word ends in speech.mp3

```

**Example 2: Getting Phoneme-Level Text Alignments**
```
from forcealign import ForceAlign

# Provide path to audio_file and corresponding txt_file with audio transcript
align = ForceAlign(audio_file='./speech.mp3', txt_file='./speech.txt')

# Runs prediction and returns alignment results
words = align.inference() 

# Show predicted phenome-level alignments
for word in words:
	print(word.word)
	for phoneme in word.phonemes:
		print(phoneme.phoneme) # ARPABET phonome spoken in audio at associated time
		print(phoneme.time_start) # Time (seconds) the phoneme starts in speech.mp3
		print(phoneme.time_end) # Time (seconds) the phoneme ends in speech.mp3

```

**Example 3: Reviewing Word Level-Alignments**

You can use the review_alignment() method to check the quality of your alignment in real-time. The review_alignment() method will play the audio file and print the individual words at their predicted times. This is useful for heuristically checking the accuracy of the word-level alignment predictions.
```
from forcealign import ForceAlign

# Provide path to audio_file and corresponding txt_file with audio transcript
align = ForceAlign(audio_file='./speech.mp3', txt_file='./speech.txt')

# Runs prediction and returns alignment results
words = align.inference() 

# Plays audio and prints each word in real-time at predicted alignment time.
align.review_alignment()

```

# Use Cases
Forced alignment can be useful for generating subtitles for video, and for generating automated lip-syncing of animated characters with phoneme-level forced alignments. 

# FAQ
**1. Does ForceAlign have speech-to-text capabilities?**
No. This is a feature that I plan on adding soon when I have time.

**2. Can ForceAlign be used with both CPU and GPU?**
Yes. Running with CPU is surprisingly fast, and it will be even faster with GPU. 

# Acknowledgements
This project is heavily based upon a demo from Pytorch by Moto Hira: [FORCED ALIGNMENT WITH WAV2VEC2](https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html)