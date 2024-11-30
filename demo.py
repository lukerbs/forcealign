from forcealign import ForceAlign

# Case 1: Providing a transcript
print("=== Case 1: Using Provided Transcript ===")
transcript = "The dog quickly jumped over the large cat and made its way out the door."
align_with_transcript = ForceAlign(audio_file="./test/speech.mp3", transcript=transcript)

# Predict alignments with provided transcript
words_with_transcript = align_with_transcript.inference()

# Show predictions with provided transcript
print("\nWord-level alignments:")
for word in words_with_transcript:
    print(f"Word: {word.word}, Start: {word.time_start}s, End: {word.time_end}s")

# Live audio review of results
print("\nReviewing alignments with provided transcript...")
align_with_transcript.review_alignment()


# Case 2: Without providing a transcript (speech-to-text capability)
print("\n=== Case 2: Using Automatic Speech-to-Text ===")
align_no_transcript = ForceAlign(audio_file="./test/speech.mp3")  # No transcript provided

# Predict alignments with generated transcript
words_no_transcript = align_no_transcript.inference()

# Show predictions with generated transcript
print("\nGenerated Transcript:")
print(align_no_transcript.raw_text)

print("\nWord-level alignments:")
for word in words_no_transcript:
    print(f"Word: {word.word}, Start: {word.time_start}s, End: {word.time_end}s")

# Live audio review of results
print("\nReviewing alignments with generated transcript...")
align_no_transcript.review_alignment()
