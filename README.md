# Triton Pretask
Triton pretask for PLCT.

## Usage

## Environment Preparation
First clone the repository.
```
$ git clone https://github.com/xvyv99/triton-pretask.git
```

If you use uv to manage Python dependencies, you can use the following commands to quickly create an environment and install dependencies:
```
$ uv sync
```

If you only use pip and virtual environment to manage dependencies and virtual environments, please use the following commands:
```
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install .
```

## Run tests and benchmark
After the environment is set up, you can use the following commands to run algorithm correctness verification and benchmark:
```
python Triton/main.py
```