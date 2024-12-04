from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="forcealign",
    version="1.1.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.1",
        "torchaudio>=2.2.1",
        "pydub>=0.25.1",
        "g2p-en>=2.1.0",
        "sox",
        "soundfile",
        "ffmpeg-python",
    ],
    author="Luke Kerbs",
    description="A Python library for forced alignment of English text to English audio.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lukerbs/forcealign",
    keywords=[
        "force align",
        "forced alignment",
        "audio segmentation",
        "audio forced alignment",
        "python forced alignment",
        "phoneme",
        "generate subtitles",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
