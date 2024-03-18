from setuptools import setup, find_packages

def parse_requirements(file_path):
    with open(file_path, 'r') as file:
        requirements = file.read().splitlines()
    return requirements

requirements = parse_requirements('requirements.txt')

setup(
    name='forcealign',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    author='Luke Kerbs',
    description='A Python library for forced alignment of English text to English audio.',
    url='https://github.com/lukerbs/forcealign',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)