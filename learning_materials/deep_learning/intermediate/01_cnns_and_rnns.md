# Deep Learning - Intermediate Level

## Convolutional Neural Networks (CNNs)

CNNs are specialized neural networks designed for processing grid-like data, especially images. They use convolution operations to automatically learn spatial hierarchies of features.

### Why CNNs for Images?

**Problem with Fully Connected Networks**:
- 200×200 RGB image = 120,000 input neurons
- Too many parameters → overfitting
- Ignores spatial structure

**CNN Advantages**:
- Parameter sharing
- Spatial hierarchy of features
- Translation invariance
- Much fewer parameters

### CNN Building Blocks

#### 1. Convolutional Layer

**Operation**: Slide a filter (kernel) across the input to detect features.

**Example - 3×3 Filter**:
```
Input (5×5):              Filter (3×3):         Output Feature Map:
1  2  3  4  5            1  0  1                ...
2  3  4  5  6            0  1  0
3  4  5  6  7     *      1  0  1              = convolution result
4  5  6  7  8
5  6  7  8  9
```

**Key Parameters**:
- **Filter size**: 3×3, 5×5, 7×7
- **Number of filters**: Depth of output
- **Stride**: Step size (usually 1 or 2)
- **Padding**: Add zeros around input (preserve size)

**Python Example**:
```python
from tensorflow.keras import layers

# Convolutional layer
layers.Conv2D(
    filters=32,           # Number of filters
    kernel_size=(3, 3),   # Filter size
    strides=(1, 1),       # Stride
    padding='same',       # 'same' or 'valid'
    activation='relu'
)
```

#### 2. Pooling Layer

**Purpose**: Reduce spatial dimensions, decrease computation, provide translation invariance.

**Max Pooling**:
```
Input (4×4):              Max Pool 2×2:
1  3  2  4
5  6  7  8         →      6  8
3  2  1  2                4  3
4  3  2  1
```

**Types**:
- **Max Pooling**: Take maximum value
- **Average Pooling**: Take average value
- **Global Average Pooling**: Average entire feature map

**Python Example**:
```python
layers.MaxPooling2D(pool_size=(2, 2), strides=2)
layers.AveragePooling2D(pool_size=(2, 2))
layers.GlobalAveragePooling2D()  # Reduces to 1D
```

#### 3. Fully Connected Layer

Final layers that perform classification based on extracted features.

### Complete CNN Architecture

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    # Input: 28×28×1 (grayscale image)

    # Conv Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # Output: 13×13×32

    # Conv Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # Output: 5×5×64

    # Conv Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    # Output: 3×3×128

    # Flatten
    layers.Flatten(),
    # Output: 1152 neurons

    # Dense layers
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### Data Augmentation

Artificially expand training data by applying transformations.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,           # Rotate images
    width_shift_range=0.1,       # Horizontal shift
    height_shift_range=0.1,      # Vertical shift
    horizontal_flip=True,        # Flip horizontally
    zoom_range=0.1,              # Zoom in/out
    shear_range=0.1              # Shear transformation
)

# Train with augmentation
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_val, y_val)
)
```

### Transfer Learning

Use pre-trained models as feature extractors or fine-tune them.

```python
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2

# Load pre-trained model (without top classification layer)
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model weights
base_model.trainable = False

# Add custom classifier
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning (optional)
# Unfreeze some layers and train with lower learning rate
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Famous CNN Architectures

#### 1. LeNet-5 (1998)
- First successful CNN
- Used for digit recognition

#### 2. AlexNet (2012)
- Won ImageNet competition
- Introduced ReLU, dropout
- 8 layers

#### 3. VGG (2014)
- Very deep (16-19 layers)
- Small 3×3 filters
- Simple architecture

#### 4. ResNet (2015)
- Introduced residual connections
- Very deep (50, 101, 152 layers)
- Solves vanishing gradient problem

#### 5. MobileNet (2017)
- Lightweight for mobile devices
- Depthwise separable convolutions

---

## Recurrent Neural Networks (RNNs)

RNNs are designed for sequential data where order matters (time series, text, speech).

### Why RNNs for Sequential Data?

**Problem with Feedforward Networks**:
- Fixed input size
- No memory of previous inputs
- Can't model temporal dependencies

**RNN Advantages**:
- Variable-length input
- Maintains hidden state (memory)
- Shares parameters across time steps

### RNN Architecture

**Basic RNN Cell**:
```
Input at time t: x_t
Hidden state: h_t
Output: y_t

h_t = tanh(W_hh × h_(t-1) + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y
```

**Visualization**:
```
x_0 → [RNN] → y_0
       ↓
x_1 → [RNN] → y_1
       ↓
x_2 → [RNN] → y_2
```

### RNN Variants

#### 1. One-to-One
Standard neural network
- Image → Label

#### 2. One-to-Many
Single input, sequence output
- Image → Caption

#### 3. Many-to-One
Sequence input, single output
- Sentiment analysis (text → sentiment)
- Video → Action classification

#### 4. Many-to-Many (same length)
Sequence to sequence, aligned
- Video frame labeling

#### 5. Many-to-Many (different length)
Sequence to sequence, not aligned
- Machine translation

### Basic RNN Implementation

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.SimpleRNN(
        units=128,              # Number of RNN units
        return_sequences=True,  # Return full sequence or just last output
        input_shape=(None, 10)  # (timesteps, features)
    ),
    layers.SimpleRNN(64),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
```

### Problems with Basic RNNs

1. **Vanishing Gradient**: Hard to learn long-term dependencies
2. **Exploding Gradient**: Gradients become very large

**Solutions**: LSTM and GRU

---

## Long Short-Term Memory (LSTM)

LSTM is an advanced RNN architecture that solves the vanishing gradient problem.

### LSTM Cell Components

1. **Forget Gate**: What to forget from cell state
2. **Input Gate**: What new information to store
3. **Output Gate**: What to output

**Gates use sigmoid activation (0 to 1)**:
- 0 = forget completely
- 1 = keep completely

### LSTM Architecture

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.LSTM(
        units=128,
        return_sequences=True,
        input_shape=(timesteps, features)
    ),
    layers.Dropout(0.2),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

### Time Series Forecasting Example

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('stock_prices.csv')
prices = data['Close'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(prices)

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = models.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    layers.Dropout(0.2),
    layers.LSTM(50, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(25),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.1
)

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
```

---

## Gated Recurrent Unit (GRU)

Simpler variant of LSTM with fewer parameters.

**Differences from LSTM**:
- 2 gates instead of 3 (update gate, reset gate)
- No separate cell state
- Faster training
- Similar performance

```python
model = models.Sequential([
    layers.GRU(128, return_sequences=True, input_shape=(timesteps, features)),
    layers.Dropout(0.2),
    layers.GRU(64),
    layers.Dropout(0.2),
    layers.Dense(1)
])
```

---

## Bidirectional RNNs

Process sequences in both forward and backward directions.

```python
model = models.Sequential([
    layers.Bidirectional(
        layers.LSTM(64, return_sequences=True),
        input_shape=(timesteps, features)
    ),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(1)
])
```

**Use cases**:
- Text classification (sentiment analysis)
- Named entity recognition
- Any task where future context helps

---

## Sequence-to-Sequence Models

For tasks like machine translation, text summarization.

### Encoder-Decoder Architecture

```python
# Encoder
encoder_inputs = layers.Input(shape=(None, num_encoder_features))
encoder_lstm = layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = layers.Input(shape=(None, num_decoder_features))
decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
```

---

## Batch Normalization

Normalizes inputs to each layer, stabilizes training.

```python
layers.Conv2D(64, (3, 3), activation='relu')
layers.BatchNormalization()
```

**Benefits**:
- Faster training
- Allows higher learning rates
- Reduces overfitting
- Less sensitive to initialization

---

## Advanced Optimizers

### 1. Adam (Adaptive Moment Estimation)
Most popular optimizer, combines momentum and RMSprop.

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy'
)
```

### 2. SGD with Momentum
```python
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss='mse'
)
```

### 3. RMSprop
```python
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    loss='mse'
)
```

### 4. AdaGrad, AdaDelta
Adaptive learning rates per parameter.

---

## Learning Rate Scheduling

Adjust learning rate during training.

```python
# Reduce LR on plateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)

# Exponential decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.96
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Cosine decay
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.01,
    decay_steps=10000
)
```

---

## Model Checkpointing

Save best model during training.

```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint]
)

# Load best model
model = tf.keras.models.load_model('best_model.h5')
```

---

## TensorBoard Visualization

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

model.fit(
    X_train, y_train,
    callbacks=[tensorboard_callback]
)

# Launch TensorBoard
# tensorboard --logdir=./logs
```

---

## Practical Projects

### 1. Image Classification
- Build CNN to classify CIFAR-10 dataset
- Use data augmentation
- Apply transfer learning

### 2. Time Series Prediction
- Predict stock prices using LSTM
- Forecast weather data
- Energy consumption prediction

### 3. Text Generation
- Character-level LSTM for text generation
- Train on Shakespeare, code, or custom corpus

### 4. Sentiment Analysis
- LSTM/GRU for movie review classification
- Bidirectional RNN for better context

### 5. Object Detection
- Use pre-trained YOLO or Faster R-CNN
- Detect objects in images/videos

---

## Tips for Better Performance

1. **Proper data preprocessing**: Normalize, augment
2. **Start simple**: Begin with smaller models
3. **Use callbacks**: Early stopping, checkpointing, LR scheduling
4. **Monitor both training and validation**: Check for overfitting
5. **Experiment**: Try different architectures, hyperparameters
6. **Use pre-trained models**: Transfer learning when possible
7. **Regularization**: Dropout, L2, data augmentation

---

**Next Steps**: Move to advanced level to learn about GANs, Autoencoders, Attention mechanisms, and advanced architectures.
