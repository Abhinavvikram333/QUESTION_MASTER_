# Natural Language Processing - Practice Questions

## Easy Difficulty

### Question 1
**What does NLP stand for and what is its purpose?**

**Answer:** Natural Language Processing. It enables computers to understand, interpret, and generate human language.

---

### Question 2
**What is tokenization?**

A) Converting text to lowercase
B) Breaking text into individual words or sentences
C) Removing stop words
D) Translating text

**Answer:** B

---

### Question 3
**Which of these is a stop word?**

A) machine
B) learning
C) the
D) algorithm

**Answer:** C

---

### Question 4
**What is the difference between stemming and lemmatization?**

**Answer:**
- **Stemming:** Reduces words to root form by removing suffixes (may not be real words). Example: "running" → "run"
- **Lemmatization:** Reduces words to dictionary base form (always real words). Example: "better" → "good"

---

### Question 5
**What is TF-IDF used for?**

**Answer:** TF-IDF (Term Frequency-Inverse Document Frequency) measures how important a word is to a document in a collection. High TF-IDF means the word is frequent in the document but rare across all documents.

---

### Question 6
**What is sentiment analysis?**

A) Analyzing sentence structure
B) Determining emotional tone (positive/negative/neutral)
C) Translating between languages
D) Extracting named entities

**Answer:** B

---

### Question 7
**What is a corpus in NLP?**

**Answer:** A large collection of text documents used for training and analysis.

---

### Question 8
**What is Named Entity Recognition (NER)?**

**Answer:** Identifying and classifying named entities (people, organizations, locations, dates, etc.) in text.

---

### Question 9
**Why do we convert text to lowercase in preprocessing?**

**Answer:** To treat words like "Hello" and "hello" as the same word, ensuring consistency.

---

### Question 10
**What is Bag of Words (BoW)?**

**Answer:** A text representation that captures word frequencies while ignoring grammar and word order. Each document becomes a vector of word counts.

---

## Medium Difficulty

### Question 1
**Explain word embeddings and how they differ from one-hot encoding.**

**Answer:**

**One-Hot Encoding:**
- Each word is represented as a sparse vector
- Vector length = vocabulary size
- Only one element is 1, rest are 0
- No semantic relationship captured
- Example: "cat" = [1, 0, 0, 0, 0, ...]

**Word Embeddings:**
- Dense, low-dimensional vectors (e.g., 100-300 dimensions)
- Capture semantic meaning
- Similar words have similar vectors
- Example: "cat" = [0.2, 0.8, 0.5, -0.3, ...]

**Advantages of Embeddings:**
- Reduced dimensionality
- Semantic relationships preserved
- Mathematical operations possible
- Better for deep learning models

---

### Question 2
**Describe Word2Vec and explain the difference between CBOW and Skip-gram.**

**Answer:**

**Word2Vec:** Neural network-based algorithm to learn word embeddings.

**CBOW (Continuous Bag of Words):**
- Predicts target word from context words
- Input: Context words → Output: Target word
- Example: ["The", "sits", "on"] → "cat"
- Faster training
- Better for frequent words

**Skip-gram:**
- Predicts context words from target word
- Input: Target word → Output: Context words
- Example: "cat" → ["The", "sits", "on"]
- Slower training
- Better for rare words
- Generally better performance

**Key Concept:** Both learn embeddings by maximizing prediction accuracy, which forces similar words to have similar vectors.

---

### Question 3
**Implement a simple text classification pipeline using TF-IDF and Naive Bayes.**

**Answer:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Sample data
texts = [
    "I love this movie, it's fantastic",
    "Great film, highly recommend",
    "Terrible movie, waste of time",
    "Boring and predictable",
    "Amazing acting and story",
    "Worst film ever made"
]
labels = [1, 1, 0, 0, 1, 0]  # 1=positive, 0=negative

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.33, random_state=42
)

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),  # Unigrams and bigrams
        stop_words='english'
    )),
    ('clf', MultinomialNB(alpha=1.0))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))

# Predict new text
new_texts = ["This movie is excellent", "Terrible acting"]
predictions = pipeline.predict(new_texts)
probabilities = pipeline.predict_proba(new_texts)

for text, pred, prob in zip(new_texts, predictions, probabilities):
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = prob[pred]
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})\n")
```

---

### Question 4
**What are n-grams and why are they useful? Provide examples.**

**Answer:**

**N-grams:** Contiguous sequences of n items (words or characters) from text.

**Types:**
- **Unigram (1-gram):** Single words
  - "I love NLP" → ["I", "love", "NLP"]

- **Bigram (2-gram):** Two consecutive words
  - "I love NLP" → ["I love", "love NLP"]

- **Trigram (3-gram):** Three consecutive words
  - "I love NLP" → ["I love NLP"]

**Why Useful:**
1. **Capture Context:** "New York" is meaningful as bigram, not separate words
2. **Phrase Detection:** Identifies common phrases
3. **Better Features:** Improves model performance
4. **Language Modeling:** Predicts next word

**Implementation:**
```python
from sklearn.feature_extraction.text import CountVectorizer

# Bigrams
vectorizer = CountVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(["I love natural language processing"])

print(vectorizer.get_feature_names_out())
# ['language processing', 'love natural', 'natural language']

# Unigrams + Bigrams
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(["I love NLP"])
```

**Applications:**
- Text classification
- Sentiment analysis
- Language modeling
- Spell checking

---

### Question 5
**Explain the attention mechanism in NLP and why it's important.**

**Answer:**

**Attention Mechanism:** Allows models to focus on relevant parts of input when producing output.

**Problem It Solves:**
Traditional sequence models (RNN/LSTM) compress entire input into fixed-size vector, losing information for long sequences.

**How It Works:**
1. For each output position, compute attention scores for all input positions
2. Use scores to create weighted sum of input representations
3. This "context vector" focuses on relevant parts

**Mathematical Formulation:**
```
# Attention scores
scores = query · keys^T / √d_k

# Attention weights (softmax)
weights = softmax(scores)

# Context vector
context = weights · values
```

**Types:**

**1. Self-Attention:**
- Input attends to itself
- Used in Transformers (BERT, GPT)

**2. Cross-Attention:**
- Output attends to input
- Used in encoder-decoder models

**Benefits:**
- Handles long sequences better
- Provides interpretability (can visualize attention)
- Captures long-range dependencies
- Foundation for Transformer architecture

**Example Use Cases:**
- Machine translation
- Text summarization
- Question answering
- Named entity recognition

---

### Question 6
**Compare and contrast RNN, LSTM, and GRU for sequence modeling.**

**Answer:**

**RNN (Recurrent Neural Network):**
- Basic sequential architecture
- **Pros:** Simple, captures temporal dependencies
- **Cons:** Vanishing/exploding gradients, short memory
- **Use when:** Short sequences, simple patterns

**LSTM (Long Short-Term Memory):**
- Advanced RNN with gating mechanisms
- **Gates:** Forget gate, input gate, output gate
- **Cell state:** Maintains long-term information
- **Pros:** Handles long-term dependencies, avoids vanishing gradient
- **Cons:** More parameters, slower training
- **Use when:** Long sequences, complex dependencies

**GRU (Gated Recurrent Unit):**
- Simplified LSTM
- **Gates:** Update gate, reset gate (no separate cell state)
- **Pros:** Fewer parameters than LSTM, faster training, similar performance
- **Cons:** Slightly less expressive than LSTM
- **Use when:** Similar to LSTM but want faster training

**Comparison Table:**

| Feature | RNN | LSTM | GRU |
|---------|-----|------|-----|
| Gates | 0 | 3 | 2 |
| Parameters | Least | Most | Middle |
| Training Speed | Fastest | Slowest | Fast |
| Memory | Short | Long | Long |
| Complexity | Simple | Complex | Moderate |

**Modern Alternative:** Transformers (BERT, GPT) have largely replaced RNN/LSTM/GRU for many NLP tasks.

---

### Question 7
**Implement a character-level LSTM for text generation.**

**Answer:**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Sample text
text = """Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks with many layers."""

# Create character mappings
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Prepare sequences
seq_length = 40
sequences = []
next_chars = []

for i in range(len(text) - seq_length):
    sequences.append([char_to_idx[c] for c in text[i:i + seq_length]])
    next_chars.append(char_to_idx[text[i + seq_length]])

X = np.array(sequences)
y = np.array(next_chars)

# Build model
model = models.Sequential([
    layers.Embedding(len(chars), 64, input_length=seq_length),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(128),
    layers.Dropout(0.2),
    layers.Dense(len(chars), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X, y, epochs=50, batch_size=128)

# Generate text
def generate_text(model, seed_text, length=200, temperature=1.0):
    generated = seed_text

    for _ in range(length):
        # Prepare input
        x = np.array([[char_to_idx[c] for c in generated[-seq_length:]]])

        # Predict
        predictions = model.predict(x, verbose=0)[0]

        # Apply temperature
        predictions = np.log(predictions + 1e-7) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))

        # Sample next character
        next_idx = np.random.choice(len(chars), p=predictions)
        next_char = idx_to_char[next_idx]

        generated += next_char

    return generated

# Generate
seed = text[:40]
generated_text = generate_text(model, seed, length=200, temperature=0.8)
print(generated_text)
```

---

### Question 8
**What is the Transformer architecture and how does it differ from RNN-based models?**

**Answer:**

**Transformer Architecture:**
- Introduced in "Attention is All You Need" (2017)
- Relies entirely on attention mechanisms (no recurrence)

**Key Components:**

**1. Self-Attention:**
- Each position attends to all positions
- Captures relationships regardless of distance

**2. Multi-Head Attention:**
- Multiple attention mechanisms in parallel
- Captures different aspects of relationships

**3. Positional Encoding:**
- Adds position information (since no recurrence)
- Uses sine/cosine functions

**4. Feed-Forward Networks:**
- Applied to each position independently

**Differences from RNN:**

| Aspect | RNN/LSTM | Transformer |
|--------|----------|-------------|
| Processing | Sequential | Parallel |
| Long-range deps | Difficult | Easy |
| Training speed | Slow | Fast |
| Parallelization | Limited | High |
| Memory | Short-term issues | Full context |

**Advantages:**
- Handles long sequences better
- Parallelizable training (faster)
- Captures global dependencies
- Better performance on most tasks

**Disadvantages:**
- More memory intensive
- Requires more data
- Longer sequences = quadratic complexity

**Applications:**
- BERT (encoder-only)
- GPT (decoder-only)
- T5 (encoder-decoder)

---

### Question 9
**Explain transfer learning in NLP and how to fine-tune pre-trained models.**

**Answer:**

**Transfer Learning:** Using pre-trained models as starting point for specific tasks.

**Why It Works:**
- Pre-trained models learn general language understanding
- Fine-tuning adapts to specific task/domain
- Requires less data and compute

**Process:**

**1. Pre-training (already done):**
- Train on massive corpus
- Learn general language representations
- Examples: BERT, GPT, RoBERTa

**2. Fine-tuning (your task):**
- Load pre-trained model
- Add task-specific layers
- Train on your data

**Implementation:**

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load pre-trained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Binary classification
)

# Prepare data
texts = ["I love this movie", "This is terrible"]
labels = [1, 0]

encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = TextDataset(encodings, labels)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Fine-tune
trainer.train()

# Inference
model.eval()
inputs = tokenizer("New text to classify", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)
    print(predictions)
```

**Best Practices:**
- Use lower learning rate (1e-5 to 5e-5)
- Fine-tune all layers or just top layers
- Use appropriate pre-trained model for your domain
- Monitor for overfitting

---

### Question 10
**Design an end-to-end NLP pipeline for sentiment analysis of customer reviews.**

**Answer:**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

class SentimentAnalysisPipeline:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None

    def preprocess_data(self, df, text_column, label_column):
        """Clean and prepare data"""
        # Remove duplicates
        df = df.drop_duplicates(subset=[text_column])

        # Remove empty texts
        df = df[df[text_column].str.strip() != '']

        # Convert labels to integers
        if df[label_column].dtype == 'object':
            label_map = {label: idx for idx, label in enumerate(df[label_column].unique())}
            df[label_column] = df[label_column].map(label_map)

        return df

    def create_dataset(self, texts, labels):
        """Create PyTorch dataset"""
        class ReviewDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=128):
                self.encodings = tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return ReviewDataset(texts, labels, self.tokenizer)

    def train(self, train_dataset, val_dataset, num_labels):
        """Train model"""
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_labels
        )

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()
        return trainer

    def predict(self, texts):
        """Make predictions"""
        self.model.eval()
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_classes = torch.argmax(predictions, dim=1)

        return predicted_classes.numpy(), predictions.numpy()

    def run_pipeline(self, df, text_column, label_column):
        """Execute complete pipeline"""
        print("1. Preprocessing data...")
        df = self.preprocess_data(df, text_column, label_column)

        print(f"Dataset size: {len(df)}")
        print(f"Label distribution:\n{df[label_column].value_counts()}")

        # Split data
        print("\n2. Splitting data...")
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            df[text_column].tolist(),
            df[label_column].tolist(),
            test_size=0.3,
            random_state=42,
            stratify=df[label_column]
        )

        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=0.5,
            random_state=42,
            stratify=temp_labels
        )

        # Create datasets
        print("\n3. Creating datasets...")
        train_dataset = self.create_dataset(train_texts, train_labels)
        val_dataset = self.create_dataset(val_texts, val_labels)
        test_dataset = self.create_dataset(test_texts, test_labels)

        # Train
        print("\n4. Training model...")
        num_labels = len(set(train_labels))
        trainer = self.train(train_dataset, val_dataset, num_labels)

        # Evaluate
        print("\n5. Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
        print(f"Test results: {test_results}")

        # Example predictions
        print("\n6. Example predictions...")
        sample_texts = [
            "This product is amazing! I love it.",
            "Terrible quality, do not buy.",
            "It's okay, nothing special."
        ]

        predicted_classes, probabilities = self.predict(sample_texts)

        for text, pred_class, probs in zip(sample_texts, predicted_classes, probabilities):
            print(f"\nText: {text}")
            print(f"Predicted class: {pred_class}")
            print(f"Probabilities: {probs}")

        return trainer, test_results

# Usage
# df = pd.read_csv('reviews.csv')
# pipeline = SentimentAnalysisPipeline()
# trainer, results = pipeline.run_pipeline(df, 'review_text', 'sentiment')
```

---

## Hard Difficulty

### Question 1
**Implement BERT from scratch including multi-head attention, positional encoding, and masked language modeling.**

**Answer:**

See the NLP Advanced section (01_transformers.md) for complete BERT implementation with all components.

---

### Question 2
**Design a neural machine translation system using Transformer architecture with attention visualization.**

**Answer:**

Requires implementing encoder-decoder Transformer with cross-attention. See NLP Advanced section for complete implementation.

---

### Question 3
**Explain how to implement and train a custom Named Entity Recognition model using BERT.**

**Answer:**

See NLP Advanced section (01_transformers.md) for complete custom NER training implementation.

---

### Question 4
**Implement a question-answering system using retrieval-augmented generation (RAG).**

**Answer:**

Combines dense retrieval (embeddings) with generative model. See NLP Advanced section for implementation details.

---

### Question 5
**Design a multi-lingual NLP system that handles code-switching and transliteration.**

**Answer:**

Requires multi-lingual embeddings, language detection, and transfer learning across languages. Advanced implementation covered in research-level NLP.
