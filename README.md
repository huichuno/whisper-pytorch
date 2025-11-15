# whisper-pytorch
This example showcases inference of speech recognition Whisper Models using PyTorch

## Prerequisites
* git - distributed version control system
    * [Windows](https://git-scm.com/install/windows) (validated)
    * [Linux](https://git-scm.com/install/linux)
* uv - fast python package manager
    * [Windows](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2) (validated)
    * [Linux](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1)

## Supported Hardware
* Intel® Core™ Ultra Processors (Series 2) (validated)

## Getting Started
```sh
git clone https://github.com/huichuno/whisper-pytorch.git

cd whisper-pytorch

uv sync
```
## Usage

### Run exported Whisper model
```
uv run .\run_whisper.py

# Output:
# How are you doing today?
# Elapsed time: 3.958 seconds
```

## Reference
* https://docs.astral.sh/uv/