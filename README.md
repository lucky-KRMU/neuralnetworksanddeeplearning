# Neural Networks and Deep Learning from Scratch

A clean, self-contained implementation of a Feedforward Neural Network from scratch in Python using only `numpy` and `matplotlib` to classify handwritten digits from the MNIST dataset.

This project is built for learning purposes, inspired by the concepts in Michael Nielsen's *"Neural Networks and Deep Learning"*.

---

## 🚀 Features

- **Custom Neural Network Architecture**: Supports customizable layer sizes (e.g., `[784, 50, 30, 10]`).
- **Stochastic Gradient Descent (SGD)**: Mini-batch gradient descent with customizable batch size, learning rate, and epochs.
- **Backpropagation Algorithm**: Efficient vector-based implementation of the backpropagation algorithm.
- **Xavier (Glorot) Initialization**: Used for setting up the weights to prevent vanishing or exploding gradients.
- **Activation Function**: Sigmoid activation function and its derivative.
- **One-Hot Label Encoding**: Conversion of digit classes to 10-dimensional target vectors.
- **Training Accuracy Tracking**: Plots the training accuracy over epochs using `matplotlib` and saves the plot as a dark-mode styled graph.

---

## 📁 Repository Structure

```text
neuralnetworksanddeeplearning/
├── project/
│   ├── data/
│   │   └── train.csv          # MNIST training dataset
│   ├── graphs/
│   │   └── accuracy.png       # Generated plot of training accuracy vs epochs
│   ├── main.py                # Main script defining the Network class and running training
│   └── notes.txt              # Project development notes (one-hot encoding & initialization)
├── others/
│   ├── exercises/
│   │   └── Exercise_Design_bitwise_digit_NN.py  # Mini-exercise code for bitwise digit recognition
│   └── temp.py                # Scratchpad for temporary calculations
├── .gitignore
└── README.md                  # This file
```

---

## 🧠 Understanding the Code

The main network implementation resides in [project/main.py](file:///d:/Personal/Coding/AI/Learning/neuralnetworksanddeeplearning/project/main.py).

### Network Architecture
The network is initialized by passing a list representing the sizes of each layer. For example, `Network([784, 50, 30, 10])` constructs:
- **Input layer**: 784 neurons (representing the $28 \times 28$ pixels of an MNIST image).
- **Hidden layers**: 50 and 30 neurons.
- **Output layer**: 10 neurons (representing the digits 0-9).

### Key Components

- **Initialization**:
  ```python
  self.biases = [ np.random.randn(y, 1) for y in (net_size[1:]) ]
  self.weights = [ np.random.randn(y,x) / np.sqrt(x) for y,x in zip(net_size[ 1: ], net_size[ :-1 ]) ]
  ```
  Biases are initialized using a standard normal distribution. Weights are initialized using Xavier initialization (`/ np.sqrt(x)`) to scale weights inversely with the square root of the number of input connections, keeping activation variances stable.

- **Feedforward**:
  The input features are normalized ($x / 255.0$) before propagation. For each layer $l$, the activation is computed as:
  $$a^l = \sigma(w^l a^{l-1} + b^l)$$
  where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation function.

- **Backpropagation**:
  For each batch, error gradients are computed starting from the output layer:
  $$\delta^L = (a^L - y) \odot \sigma'(z^L)$$
  and propagated backwards to find the gradients for weights and biases in all prior layers.

- **Weight Update**:
  After calculating gradients over a mini-batch, the weights and biases are updated using the learning rate $\eta$:
  $$w \leftarrow w - \frac{\eta}{m} \sum \nabla_w C$$
  $$b \leftarrow b - \frac{\eta}{m} \sum \nabla_b C$$

---

## 🛠️ Requirements & Setup

Make sure you have `numpy` and `matplotlib` installed:

```bash
pip install numpy matplotlib
```

### Running the Project

To start training the network on the MNIST dataset:

1. Ensure your training dataset is placed at `project/data/train.csv`.
2. Navigate to the `project` directory and run `main.py`:

```bash
python project/main.py
```

The script will:
- Load the training data.
- Train the neural network for 30 epochs with a learning rate of `0.5` and a batch size of `64`.
- Output accuracy logs to the terminal for each epoch.
- Save an accuracy chart to `project/graphs/accuracy.png`.
- Output prediction samples for the first few dataset items.
