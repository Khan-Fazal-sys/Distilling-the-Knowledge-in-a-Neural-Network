# Knowledge Distillation on MNIST

A PyTorch implementation of knowledge distillation experiments on MNIST, including soft targets, temperature sweeps, the mythical digit experiment, specialist models, and soft targets as a regularizer.

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- NumPy

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The script automatically downloads MNIST into `data/` when needed. It uses CUDA when available and otherwise runs on CPU. The default experiment trains a teacher and two student models for 20 epochs, so runtime depends on the available hardware.

Additional experiments are available in `main.py` and can be enabled in the `__main__` block:

- `run_temperature_experiment()`
- `run_mythical_digit_experiment(omit_class=3)`
- `run_specialist_experiment()`
- `run_regularization_experiment()`

## Project Structure

```text
.
├── main.py
├── requirements.txt
└── data/                 # Downloaded locally; ignored by Git
    └── MNIST/
```

## License

No license has been selected for this repository yet. Add a license before distributing the code publicly.
