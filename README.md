# Image Classification and Adversarial Attack Analysis

This project focuses on image classification and evaluating adversarial attacks on MNIST, Fashion-MNIST, and CIFAR datasets.

## Installation

Ensure you have Python installed, then clone the repository and set up the environment:

```bash
# Clone the repository
git clone https://github.com/cmaloney111/AdversarialAttacks.git
cd AdversarialAttacks

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the main script to train, evaluate, and test models against adversarial attacks:

```bash
python attack.py [OPTIONS]
```

### Available Arguments

| Argument           | Type    | Default                           | Description                                      |
|--------------------|---------|-----------------------------------|--------------------------------------------------|
| `--no-train`       | flag    | `False`                           | Skip model training and load from checkpoints    |
| `--no-eval`        | flag    | `False`                           | Skip model evaluation                           |
| `--eval-only`      | flag    | `False`                           | Only evaluate models without adversarial testing |
| `--mnist-model`    | string  | `models/mnist_model.pth`          | Path to MNIST model checkpoint                  |
| `--fashion-model`  | string  | `models/fashion_mnist_model.pth`  | Path to Fashion-MNIST model checkpoint          |
| `--cifar-model`    | string  | `models/cifar_model.pth`          | Path to CIFAR model checkpoint                  |
| `--batch-size`     | integer | `128`                             | Batch size for training and evaluation          |
| `--epochs`         | integer | `5`                               | Number of epochs for training                   |

### Example Commands

Train and evaluate all models, including adversarial testing:

```bash
python attack.py
```

Skip training and only evaluate models and run adversarial attacks:

```bash
python attack.py --no-train
```

Evaluate models without adversarial testing:

```bash
python attack.py --eval-only
```

## Output

Model training and evaluation logs will be displayed in the console and figures will be saved to the `figures` directory.

## Requirements

Ensure the required packages are installed from `requirements.txt`. Note that this project requires Python 3.x
