# Deep Learning - Advanced Level

## Generative Adversarial Networks (GANs)

GANs consist of two neural networks competing against each other: a Generator and a Discriminator.

### Architecture

**Generator (G)**: Creates fake data from random noise
**Discriminator (D)**: Distinguishes real data from fake data

**Training Process**:
1. Generator creates fake samples
2. Discriminator tries to classify real vs fake
3. Both networks improve through competition

### Mathematical Formulation

```
min_G max_D V(D,G) = E_x~p_data[log D(x)] + E_z~p_z[log(1 - D(G(z)))]

Generator minimizes: log(1 - D(G(z)))
Discriminator maximizes: log D(x) + log(1 - D(G(z)))
```

### Basic GAN Implementation

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Generator
def build_generator(latent_dim):
    model = models.Sequential([
        layers.Dense(256, input_dim=latent_dim),
        layers.LeakyReLU(0.2),
        layers.BatchNormalization(),

        layers.Dense(512),
        layers.LeakyReLU(0.2),
        layers.BatchNormalization(),

        layers.Dense(1024),
        layers.LeakyReLU(0.2),
        layers.BatchNormalization(),

        layers.Dense(28 * 28 * 1, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model

# Discriminator
def build_discriminator(img_shape):
    model = models.Sequential([
        layers.Flatten(input_shape=img_shape),

        layers.Dense(512),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Dense(256),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Build GAN
latent_dim = 100
img_shape = (28, 28, 1)

generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)

# Compile discriminator
discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Build combined model (generator + discriminator)
discriminator.trainable = False
z = layers.Input(shape=(latent_dim,))
img = generator(z)
validity = discriminator(img)

combined = models.Model(z, validity)
combined.compile(
    optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
    loss='binary_crossentropy'
)
```

### Training Loop

```python
def train_gan(generator, discriminator, combined, epochs, batch_size=128):
    # Load and preprocess data
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train.astype('float32') - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=-1)

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid)

        if epoch % 200 == 0:
            print(f"Epoch {epoch} | D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f}")

train_gan(generator, discriminator, combined, epochs=10000)
```

### GAN Variants

#### 1. DCGAN (Deep Convolutional GAN)
Uses convolutional layers instead of fully connected.

```python
def build_dcgan_generator(latent_dim):
    model = models.Sequential([
        layers.Dense(7 * 7 * 256, input_dim=latent_dim),
        layers.Reshape((7, 7, 256)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2D(1, (5, 5), padding='same', activation='tanh')
    ])
    return model

def build_dcgan_discriminator(img_shape):
    model = models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=img_shape),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
```

#### 2. Conditional GAN (cGAN)
Generate samples conditioned on additional information (labels).

```python
# Generator with label conditioning
def build_conditional_generator(latent_dim, num_classes):
    noise = layers.Input(shape=(latent_dim,))
    label = layers.Input(shape=(1,), dtype='int32')

    label_embedding = layers.Embedding(num_classes, 50)(label)
    label_embedding = layers.Flatten()(label_embedding)

    merged = layers.Concatenate()([noise, label_embedding])

    x = layers.Dense(256)(merged)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(28 * 28 * 1, activation='tanh')(x)
    output = layers.Reshape((28, 28, 1))(x)

    return models.Model([noise, label], output)
```

#### 3. Wasserstein GAN (WGAN)
Improved training stability using Wasserstein distance.

```python
# Critic (not discriminator)
def build_critic(img_shape):
    model = models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=img_shape),
        layers.LeakyReLU(0.2),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),

        layers.Flatten(),
        layers.Dense(1)  # No sigmoid activation
    ])
    return model

# Wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

# Gradient penalty for WGAN-GP
def gradient_penalty(critic, real_imgs, fake_imgs):
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
    interpolated = alpha * real_imgs + (1 - alpha) * fake_imgs

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated)

    gradients = tape.gradient(pred, interpolated)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gp = tf.reduce_mean((grad_norm - 1.0) ** 2)
    return gp
```

---

## Autoencoders

Autoencoders learn compressed representations of data (dimensionality reduction, denoising, anomaly detection).

### Architecture

**Encoder**: Compresses input to latent representation
**Decoder**: Reconstructs input from latent representation

### Basic Autoencoder

```python
from tensorflow.keras import layers, models

# Encoder
encoder_input = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(encoder_input)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
latent = layers.Dense(32, activation='relu')(x)  # Bottleneck

encoder = models.Model(encoder_input, latent)

# Decoder
decoder_input = layers.Input(shape=(32,))
x = layers.Dense(128, activation='relu')(decoder_input)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(28 * 28, activation='sigmoid')(x)
decoder_output = layers.Reshape((28, 28, 1))(x)

decoder = models.Model(decoder_input, decoder_output)

# Full Autoencoder
autoencoder_input = layers.Input(shape=(28, 28, 1))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = models.Model(autoencoder_input, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_split=0.2)
```

### Convolutional Autoencoder

```python
# Encoder
encoder_input = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
latent = layers.MaxPooling2D((2, 2), padding='same')(x)

encoder = models.Model(encoder_input, latent)

# Decoder
decoder_input = layers.Input(shape=(7, 7, 64))
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_input)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoder_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

decoder = models.Model(decoder_input, decoder_output)

# Full model
autoencoder_input = layers.Input(shape=(28, 28, 1))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = models.Model(autoencoder_input, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

### Variational Autoencoder (VAE)

VAE learns a probabilistic latent space (good for generation).

```python
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder
latent_dim = 2
encoder_inputs = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(encoder_inputs)
x = layers.Dense(256, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Sampling()([z_mean, z_log_var])

encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z])

# Decoder
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(256, activation='relu')(latent_inputs)
x = layers.Dense(28 * 28, activation='sigmoid')(x)
decoder_outputs = layers.Reshape((28, 28, 1))(x)

decoder = models.Model(latent_inputs, decoder_outputs)

# VAE
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # KL divergence regularization
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed

vae = VAE(encoder, decoder)
vae.compile(optimizer='adam', loss='binary_crossentropy')
vae.fit(X_train, X_train, epochs=30, batch_size=128)
```

### Denoising Autoencoder

```python
# Add noise to images
noise_factor = 0.3
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)

# Train autoencoder on noisy images
autoencoder.fit(X_train_noisy, X_train, epochs=50, batch_size=256)
```

---

## Attention Mechanisms

Attention allows models to focus on relevant parts of input.

### Attention for Seq2Seq

```python
class BahdanauAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, values):
        # query: decoder hidden state (batch_size, hidden_dim)
        # values: encoder outputs (batch_size, max_len, hidden_dim)

        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape: (batch_size, max_len, 1)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)
        ))

        # attention_weights shape: (batch_size, max_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape: (batch_size, hidden_dim)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
```

### Self-Attention (Scaled Dot-Product)

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Args:
        q: queries (batch_size, seq_len, d_k)
        k: keys (batch_size, seq_len, d_k)
        v: values (batch_size, seq_len, d_v)
        mask: mask tensor
    Returns:
        output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # Scale
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply mask
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply by values
    output = tf.matmul(attention_weights, v)

    return output, attention_weights
```

---

## ResNet (Residual Networks)

ResNet introduces skip connections to enable very deep networks.

### Residual Block

```python
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    # First conv
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second conv
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Adjust shortcut if needed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add shortcut
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

# Build ResNet
def build_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)
```

---

## Neural Architecture Search (NAS)

Automated search for optimal network architecture.

### EfficientNet

```python
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB7

# Use pre-trained EfficientNet
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation='softmax')
])
```

---

## Object Detection

### YOLO (You Only Look Once)

```python
# Using pre-trained YOLO with TensorFlow Hub
import tensorflow_hub as hub

detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

def detect_objects(image):
    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run detection
    detections = detector(input_tensor)

    # Extract results
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)
    scores = detections['detection_scores'][0].numpy()

    return boxes, classes, scores
```

---

## Semantic Segmentation

### U-Net

```python
def unet(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)

    # Encoder (downsampling)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)

    # Decoder (upsampling)
    u1 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c3)
    u1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)

    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c4)
    u2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)

    return models.Model(inputs, outputs)
```

---

## Advanced Training Techniques

### 1. Mixed Precision Training
```python
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Build and train model as usual
# Automatic mixed precision for faster training
```

### 2. Gradient Accumulation
```python
# For simulating larger batch sizes
accumulation_steps = 4

for step, (x_batch, y_batch) in enumerate(dataset):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions) / accumulation_steps

    gradients = tape.gradient(loss, model.trainable_variables)

    if (step + 1) % accumulation_steps == 0:
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

### 3. Curriculum Learning
Train on easier examples first, gradually increase difficulty.

```python
# Sort dataset by difficulty
sorted_data = sort_by_difficulty(dataset)

# Train in stages
for stage in range(num_stages):
    subset = get_subset_by_difficulty(sorted_data, stage)
    model.fit(subset, epochs=epochs_per_stage)
```

---

## Model Interpretability

### Grad-CAM (Gradient-weighted Class Activation Mapping)

```python
def grad_cam(model, img, layer_name):
    grad_model = models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    # Compute weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()
```

---

## Key Takeaways

- GANs enable generative modeling through adversarial training
- Autoencoders learn compressed representations
- Attention mechanisms allow focusing on relevant information
- ResNet enables very deep networks with skip connections
- Advanced techniques like NAS automate architecture design
- Model interpretability is crucial for understanding predictions

---

**Next Steps**: Explore Artificial Intelligence fundamentals, search algorithms, and AI systems.
