from setuptools import setup, find_packages

setup(
    name='forcealign',
    version='0.1.1',
    packages=find_packages(),
    install_requires=['torch==2.2.1', 'torchaudio==2.2.1', 'pydub==0.25.1', 'g2p-en==2.1.0'],
    author='Luke Kerbs',
    description='A Python library for forced alignment of English text to English audio.',
    url='https://github.com/lukerbs/forcealign',
    keywords=['force align', 'forced alignment', 'audio segmentation', 'audio forced alignment', 'python forced alignment', 'phoneme', 'generate subtitles'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)