# Deep Learning - Practice Questions

## Easy Difficulty

### Question 1
**What is Deep Learning?**

**Answer:** A subset of Machine Learning that uses artificial neural networks with multiple layers to learn from data.

---

### Question 2
**What is an activation function and why is it needed?**

**Answer:** Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns. Without them, neural networks would only learn linear transformations.

---

### Question 3
**Which activation function is most commonly used in hidden layers?**

A) Sigmoid
B) Tanh
C) ReLU
D) Softmax

**Answer:** C (ReLU)

---

### Question 4
**What is the difference between forward propagation and backward propagation?**

**Answer:**
- **Forward propagation:** Input flows through network to produce output/prediction
- **Backward propagation:** Calculate gradients and propagate error backward to update weights

---

### Question 5
**What is the purpose of a loss function?**

**Answer:** Measures how wrong the model's predictions are. The goal of training is to minimize the loss function.

---

### Question 6
**What is overfitting and how can dropout help?**

**Answer:**
- **Overfitting:** Model learns training data too well, performs poorly on new data
- **Dropout:** Randomly drops neurons during training, forcing network to learn robust features and preventing overfitting

---

### Question 7
**What is the difference between batch size and epochs?**

**Answer:**
- **Batch size:** Number of samples processed before updating weights
- **Epochs:** Number of times entire dataset passes through the network

---

### Question 8
**Why do we normalize/scale input features in neural networks?**

**Answer:**
- Speeds up training
- Prevents features with large values from dominating
- Helps gradient descent converge faster
- Reduces sensitivity to initialization

---

### Question 9
**What is the vanishing gradient problem?**

**Answer:** In deep networks, gradients become very small as they propagate backward, making it difficult to train early layers. ReLU activation and batch normalization help solve this.

---

### Question 10
**Which optimizer is most commonly used in deep learning?**

A) SGD
B) Adam
C) RMSprop
D) Adagrad

**Answer:** B (Adam)

---

## Medium Difficulty

### Question 1
**Explain Convolutional Neural Networks (CNNs) and their key components.**

**Answer:**

**CNNs:** Specialized neural networks for processing grid-like data (images).

**Key Components:**

**1. Convolutional Layer:**
- Applies filters/kernels to detect features
- Parameters: filter size, number of filters, stride, padding
- Learns hierarchical features (edges → shapes → objects)

**2. Pooling Layer:**
- Reduces spatial dimensions
- Types: Max pooling, average pooling
- Provides translation invariance

**3. Fully Connected Layer:**
- Final layers for classification
- Flattens features and performs classification

**Why CNNs for Images:**
- Parameter sharing (same filter across image)
- Spatial hierarchy
- Translation invariance
- Much fewer parameters than fully connected

**Example Architecture:**
```
Input (224×224×3)
→ Conv (64 filters, 3×3) + ReLU
→ MaxPool (2×2)
→ Conv (128 filters, 3×3) + ReLU
→ MaxPool (2×2)
→ Flatten
→ Dense (256) + ReLU
→ Dense (10) + Softmax
```

---

### Question 2
**Compare and contrast RNN, LSTM, and GRU architectures.**

**Answer:**

**RNN (Recurrent Neural Network):**
- Basic sequential architecture
- Formula: h_t = tanh(W_hh × h_(t-1) + W_xh × x_t)
- **Problems:** Vanishing gradients, short memory
- **Use:** Short sequences

**LSTM (Long Short-Term Memory):**
- Solves vanishing gradient with gates
- **Components:** Forget gate, input gate, output gate, cell state
- **Advantages:** Long-term memory, handles long sequences
- **Disadvantages:** More parameters, slower
- **Use:** Long sequences, complex dependencies

**GRU (Gated Recurrent Unit):**
- Simplified LSTM
- **Components:** Update gate, reset gate (no separate cell state)
- **Advantages:** Fewer parameters than LSTM, faster, similar performance
- **Use:** When LSTM is overkill but RNN insufficient

**Comparison:**

| Feature | RNN | LSTM | GRU |
|---------|-----|------|-----|
| Gates | 0 | 3 | 2 |
| Parameters | Least | Most | Medium |
| Speed | Fast | Slow | Medium |
| Memory | Short | Long | Long |

**Modern Alternative:** Transformers have largely replaced RNN/LSTM/GRU.

---

### Question 3
**Explain batch normalization and why it's important.**

**Answer:**

**Batch Normalization:** Normalizes inputs to each layer using batch statistics.

**Algorithm:**
```
1. Calculate batch mean: μ = (1/m) Σ x_i
2. Calculate batch variance: σ² = (1/m) Σ (x_i - μ)²
3. Normalize: x̂_i = (x_i - μ) / √(σ² + ε)
4. Scale and shift: y_i = γx̂_i + β
   where γ and β are learnable parameters
```

**Benefits:**
- **Faster training:** Higher learning rates possible
- **Regularization:** Slight regularization effect
- **Reduces internal covariate shift:** Stabilizes distribution of layer inputs
- **Less sensitive to initialization:** Easier to initialize networks
- **Allows deeper networks:** Helps gradients flow

**Where to place:**
- After linear/conv layer, before activation (common)
- Or after activation

**Implementation:**
```python
from tensorflow.keras import layers

model = Sequential([
    layers.Conv2D(64, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    # ...
])
```

---

### Question 4
**Implement a CNN from scratch for image classification.**

**Answer:**

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load data (CIFAR-10)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build CNN
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Dense layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
]

# Train
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

---

### Question 5
**Explain transfer learning and how to fine-tune pre-trained models.**

**Answer:**

**Transfer Learning:** Using pre-trained models as starting point for new tasks.

**Why It Works:**
- Pre-trained models learn general features
- Saves time and computational resources
- Requires less data
- Better performance than training from scratch

**Approaches:**

**1. Feature Extraction (Frozen Base):**
```python
from tensorflow.keras.applications import VGG16

# Load pre-trained model
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom classifier
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile with higher learning rate
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train only top layers
model.fit(X_train, y_train, epochs=10)
```

**2. Fine-Tuning (Partial Unfreezing):**
```python
# Unfreeze last few layers
base_model.trainable = True

# Freeze early layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune
model.fit(X_train, y_train, epochs=10)
```

**Best Practices:**
- Start with frozen base, train top layers
- Then unfreeze and fine-tune with low learning rate
- Use appropriate pre-trained model for your domain
- More similar domains → better transfer

**Popular Pre-trained Models:**
- VGG16/19
- ResNet50/101/152
- InceptionV3
- MobileNet
- EfficientNet

---

### Question 6
**What are GANs and how do they work? Explain the training process.**

**Answer:**

**GAN (Generative Adversarial Network):** Two neural networks competing:

**Components:**

**1. Generator (G):**
- Creates fake data from random noise
- Goal: Fool discriminator

**2. Discriminator (D):**
- Distinguishes real from fake data
- Goal: Correctly classify real vs fake

**Training Process:**

```python
for epoch in epochs:
    # Train Discriminator
    real_images = sample_real_data()
    fake_images = generator.generate(random_noise)

    d_loss_real = discriminator.train(real_images, labels=1)
    d_loss_fake = discriminator.train(fake_images, labels=0)

    # Train Generator
    random_noise = generate_noise()
    g_loss = generator.train(random_noise, labels=1)  # Want D to classify as real
```

**Mathematical Formulation:**
```
min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]

Discriminator maximizes: log D(x) + log(1 - D(G(z)))
Generator minimizes: log(1 - D(G(z)))
```

**Training Challenges:**

1. **Mode Collapse:** Generator produces limited variety
2. **Training Instability:** D too strong → G can't learn; G too strong → D provides no gradient
3. **Convergence Issues:** Hard to balance training

**Solutions:**
- Use different GAN variants (WGAN, DCGAN, StyleGAN)
- Careful learning rate tuning
- Label smoothing
- Gradient penalty (WGAN-GP)

**Applications:**
- Image generation
- Style transfer
- Data augmentation
- Super-resolution
- Text-to-image

---

### Question 7
**Explain autoencoders and their variants (VAE, denoising, sparse).**

**Answer:**

**Autoencoder:** Neural network that learns compressed representation.

**Architecture:**
```
Input → Encoder → Bottleneck (latent space) → Decoder → Output
```

**Goal:** Reconstruct input from compressed representation.

**Variants:**

**1. Standard Autoencoder:**
```python
# Encoder
encoder = Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu')  # Bottleneck
])

# Decoder
decoder = Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(784, activation='sigmoid')
])

# Full autoencoder
autoencoder = Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')
```

**2. Variational Autoencoder (VAE):**
- Learns probabilistic latent space
- Encoder outputs mean and variance
- Uses reparameterization trick
- Good for generation

```python
# z = μ + σ * ε, where ε ~ N(0,1)
z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Loss = reconstruction_loss + KL_divergence
```

**3. Denoising Autoencoder:**
- Add noise to input
- Train to reconstruct clean input
- Learns robust features

**4. Sparse Autoencoder:**
- Add sparsity constraint on activations
- Learns sparse representations

**Applications:**
- Dimensionality reduction
- Anomaly detection
- Denoising
- Feature learning
- Data compression
- Generation (VAE)

---

### Question 8
**Implement a simple LSTM for time series forecasting.**

**Answer:**

```python
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler

# Generate sample time series data
def generate_time_series(n=1000):
    t = np.arange(n)
    return np.sin(0.1 * t) + 0.1 * np.random.randn(n)

# Prepare data
data = generate_time_series(1000)

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 50
X, y = create_sequences(data_scaled, seq_length)

# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = models.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    layers.Dropout(0.2),
    layers.LSTM(50, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(25, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

# Predict
predictions = model.predict(X_test)

# Inverse transform
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# Evaluate
mae = np.mean(np.abs(predictions - y_test_actual))
print(f"Mean Absolute Error: {mae:.4f}")

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
```

---

## Hard Difficulty

### Question 1
**Implement ResNet with skip connections from scratch.**

**Answer:** See Deep Learning Advanced section (01_advanced_architectures.md) for complete ResNet implementation.

---

### Question 2
**Design and train a GAN for image generation with Wasserstein loss.**

**Answer:** See Deep Learning Advanced section for WGAN implementation with gradient penalty.

---

### Question 3
**Implement attention mechanism and build Transformer from scratch.**

**Answer:** See NLP Advanced section (01_transformers.md) for complete Transformer implementation.

---

### Question 4
**Create a U-Net architecture for image segmentation.**

**Answer:** See Deep Learning Advanced section for complete U-Net implementation.

---

### Question 5
**Implement neural architecture search using evolutionary algorithms.**

**Answer:** Requires meta-learning and architecture optimization. See Deep Learning Advanced section for NAS implementations.
