from forcealign import ForceAlign

align = ForceAlign(audio_file='./forcealign/data/speech.mp3', txt_file='./forcealign/data/speech.txt')
words = align.inference()

for word in words:
	print(word)



#align.review_alignment()