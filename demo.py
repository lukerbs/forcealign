from forcealign import ForceAlign

# Load audio and text file (audio file supports .wav or .mp3 formats)
transcript="The dog quickly jumped over the large cat and made its way out the door."
align = ForceAlign(audio_file='./test/speech.mp3', transcript=transcript)

# Predict alignments
words = align.inference() 

# Show predictions
for word in words:
	print(word.word)
	print(word.time_start)
	print(word.time_end)

# Live audio review of results
align.review_alignment()