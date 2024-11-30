import os
import torch
import torchaudio
from pydub import AudioSegment


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Greedy decoding for transcript generation."""
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


def speech_to_text(audio_file: str) -> str:
    """
    Converts speech from an audio file to text using Wav2Vec2.

    Args:
        audio_file (str): Path to an audio file (.mp3 or .wav).

    Returns:
        str: Transcribed text.
    """
    # Load Wav2Vec2 model and labels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()

    # Load audio file
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    if audio_file.lower().endswith(".mp3"):
        # Convert MP3 to WAV using pydub
        audio = AudioSegment.from_mp3(audio_file)
        wav_path = audio_file.rsplit(".", 1)[0] + ".wav"
        audio.export(wav_path, format="wav")
        audio_file = wav_path

    waveform, sample_rate = torchaudio.load(audio_file)

    # Resample audio if necessary
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=bundle.sample_rate)

    # Run inference
    with torch.inference_mode():
        emissions, _ = model(waveform.to(device))
    emissions = torch.log_softmax(emissions, dim=-1)

    # Decode using Greedy CTC Decoder
    decoder = GreedyCTCDecoder(labels=labels)
    transcript = decoder(emissions[0])

    # Clean up transcript and return
    return transcript.replace("|", " ").strip()
