from forcealign import ForceAlign

print("\n=== Automatic Speech-to-Phoneme (with timestamps) with ForceAlign ===")
transcript = '' # If you have a transcript, add it here. though it's not necessary

# uncomment the '#' below to include written transcript, otherwise it will be generated automatically
align_no_transcript = ForceAlign(audio_file="./test")#, transcript=transcript) 

# Predict alignments with generated transcript
words_no_transcript = align_no_transcript.inference()

# Show predictions with generated transcript
print("\nGenerating Transcript (ForceAlign)...")
print(align_no_transcript.raw_text)

print("\nPhoneme-level alignments (auto-generated transcript):")
# Access predicted phoneme-level alignments
for word in words_no_transcript:
    print(f"Word: {word.word} | {word.phonemes}")

# Live audio review of results
print("\nReviewing phoneme alignments with auto-generated review...")
align_no_transcript.review_phoneme_alignment()

# I recommend rigging up a puppet in Rive and plugging this script into it! 