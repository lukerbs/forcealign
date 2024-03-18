from forcealign import ForceAlign

# Load audio and text file (audio file supports .wav or .mp3 formats)
align = ForceAlign(audio_file='./path/to/speech.mp3', txt_file='./path/to/speech.txt')

# Predict alignments
words = align.inference() 

# Show predictions
for word in words:
	print(word.word)
	print(word.time_start)
	print(word.time_end)

# Live audio review of results
align.review_alignment()