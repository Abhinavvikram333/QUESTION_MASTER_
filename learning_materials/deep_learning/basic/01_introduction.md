# Deep Learning - Basic Level

## What is Deep Learning?

Deep Learning is a subset of Machine Learning that uses artificial neural networks with multiple layers (hence "deep") to learn from data. It's inspired by the structure and function of the human brain.

### Relationship to ML and AI
```
Artificial Intelligence
    └── Machine Learning
        └── Deep Learning
```

## Why Deep Learning?

**Traditional ML**:
- Requires manual feature engineering
- Limited by hand-crafted features
- Works well with small to medium datasets

**Deep Learning**:
- Automatically learns features from raw data
- Can handle unstructured data (images, audio, text)
- Excels with large datasets
- Achieves state-of-the-art results in many domains

## The Neuron: Basic Building Block

### Biological Inspiration
- Human brain has ~86 billion neurons
- Each neuron receives signals, processes them, and sends output

### Artificial Neuron (Perceptron)

**Components**:
1. **Inputs** (x₁, x₂, x₃, ...)
2. **Weights** (w₁, w₂, w₃, ...)
3. **Bias** (b)
4. **Activation Function** (f)

**Calculation**:
```
z = (x₁ × w₁) + (x₂ × w₂) + ... + b
output = f(z)
```

**Example**: Simple AND gate
```
Inputs: x₁, x₂ (each 0 or 1)
Weights: w₁ = 1, w₂ = 1
Bias: b = -1.5

x₁ = 0, x₂ = 0: z = 0×1 + 0×1 + (-1.5) = -1.5 → output = 0
x₁ = 0, x₂ = 1: z = 0×1 + 1×1 + (-1.5) = -0.5 → output = 0
x₁ = 1, x₂ = 0: z = 1×1 + 0×1 + (-1.5) = -0.5 → output = 0
x₁ = 1, x₂ = 1: z = 1×1 + 1×1 + (-1.5) = 0.5 → output = 1
```

## Activation Functions

Activation functions introduce non-linearity, allowing neural networks to learn complex patterns.

### 1. Sigmoid
```
σ(x) = 1 / (1 + e^(-x))
Output range: 0 to 1
```
**Use case**: Binary classification output layer

**Pros**: Smooth, interpretable as probability
**Cons**: Vanishing gradient problem

### 2. Tanh (Hyperbolic Tangent)
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
Output range: -1 to 1
```
**Pros**: Zero-centered, stronger gradients than sigmoid
**Cons**: Still has vanishing gradient

### 3. ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
```
**Pros**: Simple, fast, no vanishing gradient for positive values
**Cons**: Dead ReLU problem (neurons can die)
**Use case**: Most common in hidden layers

### 4. Leaky ReLU
```
LeakyReLU(x) = max(0.01x, x)
```
**Pros**: Fixes dead ReLU problem
**Use case**: Alternative to ReLU in hidden layers

**Python Example**:
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Visualize
x = np.linspace(-5, 5, 100)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.legend()
plt.grid()
plt.show()
```

## Neural Network Architecture

### Layers

#### 1. Input Layer
- Receives raw data
- One neuron per feature
- Example: 28×28 image = 784 input neurons

#### 2. Hidden Layer(s)
- Process and transform data
- Can have multiple hidden layers (deep network)
- Each layer learns increasingly complex features

#### 3. Output Layer
- Produces final prediction
- Size depends on task:
  - Binary classification: 1 neuron
  - Multi-class classification: N neurons (N classes)
  - Regression: 1 neuron

### Example: Simple Network for Digit Recognition

```
Input Layer (784 neurons - 28×28 pixels)
    ↓
Hidden Layer 1 (128 neurons)
    ↓
Hidden Layer 2 (64 neurons)
    ↓
Output Layer (10 neurons - digits 0-9)
```

## Building Your First Neural Network

### Using Keras/TensorFlow

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),  # Hidden layer 1
    layers.Dense(64, activation='relu'),                       # Hidden layer 2
    layers.Dense(10, activation='softmax')                     # Output layer
])

# Model summary
model.summary()

# Total params: (784×128 + 128) + (128×64 + 64) + (64×10 + 10)
```

### Full Example: MNIST Digit Classification

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess
X_train = X_train.reshape(-1, 784) / 255.0  # Flatten and normalize
X_test = X_test.reshape(-1, 784) / 255.0

# Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),  # Prevents overfitting
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Make predictions
predictions = model.predict(X_test[:5])
predicted_classes = np.argmax(predictions, axis=1)
print(f"Predicted: {predicted_classes}")
print(f"Actual: {y_test[:5]}")
```

## How Neural Networks Learn

### 1. Forward Propagation
- Input data flows through the network
- Each layer transforms the data
- Final layer produces prediction

### 2. Loss Function
Measures how wrong the predictions are.

**Common Loss Functions**:
- **Binary Cross-Entropy**: Binary classification
- **Categorical Cross-Entropy**: Multi-class classification
- **Mean Squared Error (MSE)**: Regression

### 3. Backward Propagation (Backprop)
- Calculate gradient of loss with respect to each weight
- Determine how to adjust weights to reduce loss

### 4. Optimization
- Update weights using gradient descent
- Learning rate controls step size

**Gradient Descent**:
```
new_weight = old_weight - learning_rate × gradient
```

### Training Loop
```
For each epoch:
    For each batch:
        1. Forward propagation (get predictions)
        2. Calculate loss
        3. Backward propagation (compute gradients)
        4. Update weights
```

## Key Hyperparameters

### 1. Learning Rate
- How big the weight updates are
- Too high: May overshoot optimal values
- Too low: Slow training
- Typical values: 0.001, 0.01, 0.1

### 2. Batch Size
- Number of samples processed before updating weights
- Smaller batches: Noisy gradients, faster updates
- Larger batches: Stable gradients, slower updates
- Common values: 32, 64, 128

### 3. Epochs
- Number of times the entire dataset is passed through
- Too few: Underfitting
- Too many: Overfitting
- Typical range: 10-100+

### 4. Number of Layers and Neurons
- More layers/neurons: More capacity to learn
- Too many: Overfitting, slow training
- Too few: Underfitting

## Common Challenges

### 1. Overfitting
Model learns training data too well, performs poorly on new data.

**Solutions**:
- Use more training data
- Add dropout layers
- Reduce model complexity
- Use regularization (L1, L2)
- Early stopping

### 2. Underfitting
Model is too simple to capture patterns.

**Solutions**:
- Increase model complexity (more layers/neurons)
- Train longer
- Use better features

### 3. Vanishing Gradients
Gradients become very small in deep networks.

**Solutions**:
- Use ReLU activation instead of sigmoid/tanh
- Batch normalization
- Residual connections (ResNet)

### 4. Exploding Gradients
Gradients become very large.

**Solutions**:
- Gradient clipping
- Proper weight initialization
- Batch normalization

## Regularization Techniques

### 1. Dropout
Randomly "drop" neurons during training.

```python
layers.Dropout(0.2)  # Drop 20% of neurons
```

### 2. L2 Regularization (Weight Decay)
Penalize large weights.

```python
layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))
```

### 3. Early Stopping
Stop training when validation performance stops improving.

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(X_train, y_train, validation_split=0.2, callbacks=[early_stop])
```

## Real-World Applications

1. **Computer Vision**:
   - Image classification
   - Object detection
   - Face recognition

2. **Natural Language Processing**:
   - Language translation
   - Text generation
   - Sentiment analysis

3. **Speech Recognition**:
   - Voice assistants (Siri, Alexa)
   - Transcription services

4. **Healthcare**:
   - Medical image analysis
   - Disease diagnosis
   - Drug discovery

5. **Autonomous Vehicles**:
   - Object detection
   - Lane detection
   - Decision making

## Practice Exercise

**Task**: Build a network to classify fashion items (T-shirts, Dresses, etc.)

```python
# Load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Your task:
# 1. Preprocess the data
# 2. Build a neural network with at least 2 hidden layers
# 3. Train the model
# 4. Evaluate on test set
# 5. Achieve at least 85% accuracy
```

## Key Takeaways

- Deep Learning uses multi-layer neural networks
- Neurons perform weighted sum + activation function
- Networks learn through forward prop, backprop, and optimization
- Activation functions introduce non-linearity
- Regularization prevents overfitting
- Deep Learning excels with large datasets and unstructured data

---

**Next Steps**: Move to intermediate level to learn about CNNs, RNNs, and advanced architectures.
