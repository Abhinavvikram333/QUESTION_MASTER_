# Natural Language Processing - Advanced Level

## Transformer Architecture

The Transformer architecture, introduced in "Attention is All You Need" (2017), revolutionized NLP by relying entirely on attention mechanisms without recurrence.

### Key Components

#### 1. Self-Attention Mechanism

**Concept**: Each word attends to all other words in the sequence to understand context.

**Mathematical Formulation**:
```
Q (Query) = X * W_Q
K (Key) = X * W_K
V (Value) = X * W_V

Attention(Q, K, V) = softmax(QK^T / √d_k) * V
```

**Intuition**:
- Query: What am I looking for?
- Key: What do I contain?
- Value: What do I actually output?

#### 2. Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Linear projections
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output, attention_weights
```

#### 3. Positional Encoding

Since Transformers don't have recurrence, we need to inject position information:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

---

## BERT (Bidirectional Encoder Representations from Transformers)

### Architecture
- **Encoder-only** Transformer
- Pre-trained on massive text corpus
- Fine-tuned for specific tasks

### Pre-training Objectives

#### 1. Masked Language Modeling (MLM)
```
Input:  "The [MASK] sat on the mat"
Target: "cat"
```

#### 2. Next Sentence Prediction (NSP)
```
Sentence A: "The cat sat on the mat"
Sentence B: "It was sleeping" → IsNext
Sentence B: "I love pizza" → NotNext
```

### Using Pre-trained BERT

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode text
text = "Hello, how are you?"
encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Get embeddings
with torch.no_grad():
    outputs = model(**encoded)

# Last hidden states
last_hidden_states = outputs.last_hidden_state
# Shape: (batch_size, sequence_length, hidden_size)

# CLS token embedding (sentence representation)
cls_embedding = last_hidden_states[:, 0, :]
```

### Fine-tuning BERT for Classification

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load model for classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Prepare dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
eval_dataset = TextDataset(eval_texts, eval_labels, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train
trainer.train()

# Evaluate
trainer.evaluate()
```

---

## GPT (Generative Pre-trained Transformer)

### Architecture
- **Decoder-only** Transformer
- Autoregressive generation
- Pre-trained on next-token prediction

### Using GPT for Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate text
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate
output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=3,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

# Decode
for i, sequence in enumerate(output):
    text = tokenizer.decode(sequence, skip_special_tokens=True)
    print(f"\nGeneration {i+1}:\n{text}")
```

### Fine-tuning GPT

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Prepare dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="train.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
```

---

## T5 (Text-to-Text Transfer Transformer)

T5 frames all NLP tasks as text-to-text problems.

### Examples:
```
Translation:    "translate English to German: Hello" → "Hallo"
Summarization:  "summarize: [long text]" → "[summary]"
Classification: "sentiment: I love this!" → "positive"
```

### Using T5

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Summarization
input_text = "summarize: " + long_article
input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids

outputs = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Translation
input_text = "translate English to French: Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Advanced Named Entity Recognition

### Using Transformers for NER

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Load NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
results = ner_pipeline(text)

for entity in results:
    print(f"{entity['word']:20} {entity['entity']:10} {entity['score']:.2f}")
```

### Custom NER Training

```python
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np

# Load model
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(label_list)
)

# Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Training
training_args = TrainingArguments(
    output_dir="./ner-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
```

---

## Question Answering

### Extractive QA with BERT

```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

context = """
The Amazon rainforest is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are covered by the rainforest.
"""

question = "How large is the Amazon rainforest?"

result = qa_pipeline(question=question, context=context)
print(f"Answer: {result['answer']}")
print(f"Score: {result['score']:.2f}")
```

### Custom QA Model

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

start_scores = outputs.start_logits
end_scores = outputs.end_logits

start_idx = torch.argmax(start_scores)
end_idx = torch.argmax(end_scores)

answer_tokens = inputs.input_ids[0][start_idx:end_idx+1]
answer = tokenizer.decode(answer_tokens)
```

---

## Text Summarization

### Abstractive Summarization

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
[Long article text here]
"""

summary = summarizer(article, max_length=130, min_length=30, do_sample=False)
print(summary[0]['summary_text'])
```

### Custom Summarization with T5

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

def summarize(text, max_length=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(
        inputs,
        max_length=max_length,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Machine Translation

### Using MarianMT

```python
from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-de'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

text = "Hello, how are you today?"
translation = translate(text)
print(translation)
```

---

## Semantic Search

### Using Sentence Transformers

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus of documents
documents = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Python is a programming language",
    "Natural language processing helps computers understand text"
]

# Encode documents
doc_embeddings = model.encode(documents, convert_to_tensor=True)

# Query
query = "What is AI and machine learning?"
query_embedding = model.encode(query, convert_to_tensor=True)

# Calculate similarity
similarities = util.cos_sim(query_embedding, doc_embeddings)

# Get top results
top_results = torch.topk(similarities[0], k=2)

for score, idx in zip(top_results[0], top_results[1]):
    print(f"Score: {score:.4f} | Document: {documents[idx]}")
```

---

## Zero-Shot Classification

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "I love this new smartphone, it has amazing features!"

candidate_labels = ["technology", "politics", "sports", "entertainment"]

result = classifier(text, candidate_labels)

print(f"Text: {text}\n")
for label, score in zip(result['labels'], result['scores']):
    print(f"{label}: {score:.4f}")
```

---

## Few-Shot Learning with GPT

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2-large')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

# Few-shot prompt
prompt = """
Classify the sentiment of the following reviews:

Review: "This movie was amazing! Best film I've seen this year."
Sentiment: Positive

Review: "Terrible service, very disappointed."
Sentiment: Negative

Review: "It was okay, nothing special."
Sentiment: Neutral

Review: "I absolutely loved it! Highly recommend."
Sentiment: """

input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, max_length=len(input_ids[0]) + 10)

result = tokenizer.decode(output[0], skip_special_tokens=True)
```

---

## Advanced Evaluation Metrics

### BLEU Score (Translation)
```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

reference = [['the', 'cat', 'is', 'on', 'the', 'mat']]
candidate = ['the', 'cat', 'is', 'on', 'the', 'mat']

score = sentence_bleu(reference, candidate)
```

### ROUGE Score (Summarization)
```python
from rouge import Rouge

rouge = Rouge()

hypothesis = "the cat is on the mat"
reference = "the cat is sitting on the mat"

scores = rouge.get_scores(hypothesis, reference)
print(scores)
```

### BERTScore (Contextual Similarity)
```python
from bert_score import score

candidates = ["the cat is on the mat"]
references = ["the cat is sitting on the mat"]

P, R, F1 = score(candidates, references, lang="en")
print(f"Precision: {P.mean():.4f}")
print(f"Recall: {R.mean():.4f}")
print(f"F1: {F1.mean():.4f}")
```

---

## Model Compression and Optimization

### Knowledge Distillation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets from teacher
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard targets from labels
        hard_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

### Quantization
```python
import torch

# Post-training static quantization
model_fp32 = ... # Your trained model
model_fp32.eval()

# Prepare for quantization
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model_fp32)

# Calibrate with representative data
with torch.no_grad():
    for data in calibration_data:
        model_prepared(data)

# Convert to quantized model
model_quantized = torch.quantization.convert(model_prepared)
```

### Pruning
```python
import torch.nn.utils.prune as prune

# Prune 30% of connections
prune.l1_unstructured(model.linear1, name="weight", amount=0.3)
prune.l1_unstructured(model.linear2, name="weight", amount=0.3)

# Make pruning permanent
prune.remove(model.linear1, 'weight')
prune.remove(model.linear2, 'weight')
```

---

## Production Deployment

### Using ONNX Runtime
```python
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer

# Export to ONNX
# (assuming model is already exported)

# Load ONNX model
ort_session = ort.InferenceSession("model.onnx")

# Prepare input
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer("Hello world", return_tensors="np")

# Run inference
outputs = ort_session.run(None, dict(inputs))
```

### Serving with FastAPI
```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
classifier = pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: TextInput):
    result = classifier(input.text)
    return {"sentiment": result[0]['label'], "score": result[0]['score']}
```

---

**Key Takeaways**:
- Transformers are the state-of-the-art architecture for NLP
- Pre-trained models can be fine-tuned for specific tasks
- Different architectures: BERT (encoder), GPT (decoder), T5 (encoder-decoder)
- Model compression is crucial for production deployment
- Always evaluate models with appropriate metrics

**Next Steps**: Explore Deep Learning fundamentals and advanced neural network architectures.
