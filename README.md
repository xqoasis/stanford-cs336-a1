# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Training

### Quick Start

To train a Transformer language model, use the provided training script:

```bash
# Basic usage (always use 'uv run')
uv run python train.py \
    --train_data data/train.npy \
    --vocab_size 10000 \
    --max_iters 1000


**⚠️ Important**: This project uses `uv` for dependency management. Always use `uv run python` instead of just `python`.

### Documentation

- **[TRAINING.md](TRAINING.md)** - Complete training guide with examples, parameters, and implementation details

### Key Features

**Configurable hyperparameters** via command-line arguments  
**Memory-efficient data loading** with `np.memmap`  
**Checkpoint management** for saving and resuming training  
**Performance monitoring** with console logging and optional W&B integration  
**Learning rate scheduling** with cosine annealing and warmup  
**Gradient clipping** for training stability  

See [TRAINING.md](TRAINING.md) for detailed usage instructions.

