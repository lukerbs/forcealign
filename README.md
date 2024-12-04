
# ForceAlign 
ForceAlign is a Python library for forced alignment of English text to English audio. It can generate **word** or [**phoneme**](https://en.wikipedia.org/wiki/Phoneme)-level alignments, identifying the specific time a word or phoneme was spoken within an audio recording. ForceAlign supports `.mp3` and `.wav` audio file formats.

For phoneme-level alignments, ForceAlign currently supports the [ARPABET](https://en.wikipedia.org/wiki/ARPABET) phonetic transcription encoding.

ForceAlign uses PyTorch's **Wav2Vec2** pretrained model for acoustic feature extraction and can run on both CPU and CUDA GPU devices. It now includes **automatic speech-to-text transcription**, making it even more flexible for use cases where transcripts are not readily available.

---

## Features
- Fast and accurate word and phoneme-level forced alignment of text to audio.
- Includes **automatic speech transcription** if a transcript is not provided.
- Optimized for both CPU and GPU.
- OS-independent—compatible with macOS, Windows, and Linux.
- Supports `.mp3` and `.wav` audio file formats.

---

## Installation and Dependencies
1. Install ForceAlign:
   ```bash
   pip3 install forcealign
   ```
2. Install `ffmpeg` (required for audio processing):
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`
   - **Windows**: Install from [ffmpeg.org](https://ffmpeg.org/download.html)

---

## Usage Examples

### Example 1: Getting Word-Level Text Alignments with a Provided Transcript
```python
from forcealign import ForceAlign

# Provide path to audio file and corresponding transcript
transcript = "The quick brown fox jumps over the lazy dog."
align = ForceAlign(audio_file='./speech.mp3', transcript=transcript)

# Run prediction and return alignment results
words = align.inference()

# Show predicted word-level alignments
for word in words:
    print(f"Word: {word.word}, Start: {word.time_start}s, End: {word.time_end}s")
```

---

### Example 2: Getting Word-Level Text Alignments with Automatic Speech Transcription
If a transcript is not provided, ForceAlign can automatically generate one using Wav2Vec2.

```python
from forcealign import ForceAlign

# Provide path to audio file; omit transcript
align = ForceAlign(audio_file='./speech.mp3')

# Automatically generate transcript and align words
words = align.inference()

# Show the generated transcript
print("Generated Transcript:")
print(align.raw_text)

# Show predicted word-level alignments
for word in words:
    print(f"Word: {word.word}, Start: {word.time_start}s, End: {word.time_end}s")
```

---

### Example 3: Speech-to-Text Conversion
ForceAlign can be used as a standalone speech-to-text tool.

```python
from forcealign import speech_to_text

# Generate transcript directly from an audio file
generated_transcript = speech_to_text("./speech.mp3")
print(generated_transcript)
# Output: "THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG"
```

---

### Example 4: Getting Phoneme-Level Text Alignments
```python
from forcealign import ForceAlign

# Provide path to audio file and transcript
transcript = "The quick brown fox jumps over the lazy dog."
align = ForceAlign(audio_file='./speech.mp3', transcript=transcript)

# Run prediction and return alignment results
words = align.inference()

# Access predicted phoneme-level alignments
for word in words:
    print(f"Word: {word.word}")
    for phoneme in word.phonemes:
        print(f"Phoneme: {phoneme.phoneme}, Start: {phoneme.time_start}s, End: {phoneme.time_end}s")
```

---

### Example 5: Reviewing Word-Level Alignments in Real-Time
```python
from forcealign import ForceAlign

# Provide path to audio file and transcript
transcript = "The quick brown fox jumps over the lazy dog."
align = ForceAlign(audio_file='./speech.mp3', transcript=transcript)

# Play the audio while printing word alignments in real-time
align.review_alignment()
```

---


## Use Cases
- **Subtitle Generation**:
  - Generate timestamps for subtitles or closed captions for videos.
- **Phoneme Analysis**:
  - Analyze phoneme-level details for language research, speech therapy, or pronunciation training.
- **Animated Lip Syncing**:
  - Use phoneme alignments to synchronize animated character lip movements with audio.
- **Accessibility Tools**:
  - Enhance accessibility by creating aligned captions or transcripts for audio recordings.

---

## FAQ

**1. Does ForceAlign have speech-to-text capabilities?**  
Yes! If you do not provide a transcript, ForceAlign will automatically generate one using Wav2Vec2. You can also provide your own transcript for better accuracy.

**2. Can ForceAlign be used with both CPU and GPU?**  
Yes. ForceAlign is optimized for both CPU and CUDA-enabled GPU devices. Using a GPU significantly speeds up processing for longer recordings.

**3. Can ForceAlign handle non-English audio?**  
No. Currently, ForceAlign supports English only. Support for additional languages may be added in future updates.

---

## Acknowledgements
This project is heavily based upon a demo from PyTorch by Moto Hira: [FORCED ALIGNMENT WITH WAV2VEC2](https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html).
