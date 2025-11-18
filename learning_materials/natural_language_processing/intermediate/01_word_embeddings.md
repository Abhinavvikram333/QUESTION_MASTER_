# Natural Language Processing - Intermediate Level

## Word Embeddings

Word embeddings are dense vector representations of words that capture semantic meaning. Unlike bag-of-words, embeddings represent words in continuous vector space where similar words have similar vectors.

### Why Word Embeddings?

**Traditional One-Hot Encoding**:
```
cat:  [1, 0, 0, 0, 0]
dog:  [0, 1, 0, 0, 0]
king: [0, 0, 1, 0, 0]
```
Problems:
- High dimensionality (vocabulary size)
- No semantic relationship captured
- Sparse vectors

**Word Embeddings**:
```
cat:  [0.2, 0.8, 0.5, 0.1]
dog:  [0.3, 0.7, 0.4, 0.2]
king: [0.9, 0.1, 0.8, 0.3]
```
Benefits:
- Dense, low-dimensional vectors
- Captures semantic similarity
- Mathematical operations possible

---

## Word2Vec

Word2Vec learns word embeddings by predicting context words from target words (or vice versa).

### 1. CBOW (Continuous Bag of Words)
Predicts target word from context words.

**Example**:
```
Sentence: "The cat sits on the mat"
Context: ["The", "sits", "on", "the"] → Target: "cat"
```

### 2. Skip-gram
Predicts context words from target word.

**Example**:
```
Target: "cat" → Context: ["The", "sits", "on", "the"]
```

### Using Pre-trained Word2Vec

```python
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# Load pre-trained Google News vectors
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Get vector for a word
vector = model['computer']
print(vector.shape)  # (300,)

# Find similar words
similar = model.most_similar('computer', topn=5)
print(similar)
# [('computers', 0.71), ('laptop', 0.65), ('PC', 0.62), ...]

# Word arithmetic
result = model.most_similar(positive=['king', 'woman'], negative=['man'])
print(result)  # Should give 'queen'

# Similarity between words
similarity = model.similarity('cat', 'dog')
print(similarity)  # 0.76
```

### Training Your Own Word2Vec

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample corpus
sentences = [
    "I love machine learning",
    "Machine learning is great",
    "Deep learning is a subset of machine learning",
    "I enjoy learning about AI"
]

# Tokenize
tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]

# Train Word2Vec
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,      # Dimension of embeddings
    window=5,             # Context window size
    min_count=1,          # Minimum word frequency
    workers=4,            # Parallel processing
    sg=0                  # 0 for CBOW, 1 for Skip-gram
)

# Save and load
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
```

---

## GloVe (Global Vectors)

GloVe creates embeddings by factorizing word co-occurrence matrix.

```python
from glove import Corpus, Glove

# Create corpus
corpus = Corpus()
corpus.fit(tokenized_sentences, window=5)

# Train GloVe
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

# Get similar words
glove.most_similar('learning', number=5)
```

---

## FastText

Extension of Word2Vec that represents words as bag of character n-grams. Better for handling out-of-vocabulary words and morphologically rich languages.

```python
from gensim.models import FastText

# Train FastText
model = FastText(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

# Handles out-of-vocabulary words
vector = model.wv['unknownword']  # Still generates a vector!

# Subword information
vector = model.wv['running']  # Uses: 'run', 'runn', 'unni', 'nnin', 'ning'
```

---

## Part-of-Speech (POS) Tagging

Identifying grammatical roles of words.

### Common POS Tags
- **NN**: Noun (cat, dog, house)
- **VB**: Verb (run, eat, sleep)
- **JJ**: Adjective (beautiful, fast, red)
- **RB**: Adverb (quickly, very, well)
- **DT**: Determiner (the, a, an)
- **IN**: Preposition (in, on, at)

### Using spaCy for POS Tagging

```python
import spacy

nlp = spacy.load('en_core_web_sm')
text = "The quick brown fox jumps over the lazy dog"
doc = nlp(text)

for token in doc:
    print(f"{token.text:10} {token.pos_:10} {token.tag_:10} {token.dep_:10}")

# Output:
# The        DET        DT         det
# quick      ADJ        JJ         amod
# brown      ADJ        JJ         amod
# fox        NOUN       NN         nsubj
# jumps      VERB       VBZ        ROOT
# over       ADP        IN         prep
```

---

## Named Entity Recognition (NER)

Extracting entities like persons, organizations, locations, dates, etc.

### Using spaCy for NER

```python
import spacy

nlp = spacy.load('en_core_web_sm')
text = "Apple is looking to buy a startup in San Francisco for $1 billion"
doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.text:20} {ent.label_:15} {spacy.explain(ent.label_)}")

# Output:
# Apple                ORG             Companies, agencies, institutions
# San Francisco        GPE             Countries, cities, states
# $1 billion           MONEY           Monetary values
```

### Training Custom NER Model

```python
import spacy
from spacy.training import Example

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

# Add labels
ner.add_label("PRODUCT")
ner.add_label("COMPANY")

# Training data
TRAIN_DATA = [
    ("iPhone is made by Apple", {"entities": [(0, 6, "PRODUCT"), (19, 24, "COMPANY")]}),
    ("Samsung produces Galaxy phones", {"entities": [(0, 7, "COMPANY"), (17, 23, "PRODUCT")]})
]

# Train
nlp.begin_training()
for epoch in range(10):
    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example])
```

---

## Text Classification

### Using Traditional ML with TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Sample data
texts = [
    "I love this movie, it's fantastic",
    "Terrible film, waste of time",
    "Great acting and story",
    "Boring and predictable"
]
labels = [1, 0, 1, 0]  # 1: positive, 0: negative

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', MultinomialNB())
])

# Train
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)
```

### Using Word Embeddings for Classification

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def document_vector(doc, model):
    """Average word vectors for a document"""
    words = [word for word in doc.split() if word in model.wv]
    if len(words) == 0:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in words], axis=0)

# Convert documents to vectors
X_train_vectors = np.array([document_vector(doc, word2vec_model) for doc in X_train])
X_test_vectors = np.array([document_vector(doc, word2vec_model) for doc in X_test])

# Train classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train_vectors, y_train)

# Predict
predictions = clf.predict(X_test_vectors)
```

---

## Sequence Labeling

### Chunking (Phrase Extraction)

```python
import nltk

sentence = "The big brown fox jumps over the lazy dog"
tokens = nltk.word_tokenize(sentence)
pos_tags = nltk.pos_tag(tokens)

# Define chunk grammar
grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar)

# Extract chunks
tree = chunk_parser.parse(pos_tags)
print(tree)

# Extract noun phrases
for subtree in tree.subtrees():
    if subtree.label() == 'NP':
        print(' '.join([word for word, tag in subtree.leaves()]))
```

---

## Dependency Parsing

Understanding grammatical structure and relationships between words.

```python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("The cat sat on the mat")

for token in doc:
    print(f"{token.text:10} {token.dep_:10} {token.head.text:10}")

# Visualization
from spacy import displacy
displacy.render(doc, style='dep', jupyter=True)
```

---

## Text Similarity

### Cosine Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Machine learning is great",
    "I love machine learning",
    "The weather is nice today"
]

# TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate similarity
similarity_matrix = cosine_similarity(tfidf_matrix)
print(similarity_matrix)
```

### Using Word Embeddings

```python
from scipy.spatial.distance import cosine

def sentence_similarity(sent1, sent2, model):
    vec1 = document_vector(sent1, model)
    vec2 = document_vector(sent2, model)
    return 1 - cosine(vec1, vec2)

similarity = sentence_similarity("I love cats", "I adore felines", word2vec_model)
```

---

## Text Preprocessing Pipeline

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Tokenize
        tokens = nltk.word_tokenize(text)

        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 2
        ]

        return ' '.join(tokens)

# Usage
preprocessor = TextPreprocessor()
cleaned = preprocessor.clean_text("Check out this amazing article! https://example.com #NLP @johndoe")
```

---

## Topic Modeling

### Latent Dirichlet Allocation (LDA)

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Machine learning and AI are transforming technology",
    "Sports news: football team wins championship",
    "Stock market reaches new highs in trading",
    "Deep learning models improve computer vision"
]

# Vectorize
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(documents)

# Train LDA
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(doc_term_matrix)

# Display topics
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")
```

---

## Text Generation

### N-gram Language Models

```python
from nltk import ngrams
from collections import defaultdict, Counter

def train_ngram_model(text, n=2):
    tokens = nltk.word_tokenize(text.lower())
    model = defaultdict(Counter)

    for ngram in ngrams(tokens, n):
        prefix = ngram[:-1]
        word = ngram[-1]
        model[prefix][word] += 1

    return model

def generate_text(model, seed, length=20):
    current = tuple(seed.lower().split())
    result = list(current)

    for _ in range(length):
        if current not in model:
            break
        possible_words = model[current]
        next_word = max(possible_words, key=possible_words.get)
        result.append(next_word)
        current = tuple(result[-len(current):])

    return ' '.join(result)

# Train
corpus = "machine learning is great. machine learning helps solve problems."
model = train_ngram_model(corpus, n=2)

# Generate
generated = generate_text(model, "machine", length=10)
```

---

## Evaluation Metrics for NLP

### Classification Metrics
```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

### Sequence Labeling Metrics
```python
from seqeval.metrics import f1_score, classification_report

# For NER, POS tagging
y_true = [['O', 'B-PER', 'I-PER', 'O', 'B-LOC']]
y_pred = [['O', 'B-PER', 'O', 'O', 'B-LOC']]

f1 = f1_score(y_true, y_pred)
report = classification_report(y_true, y_pred)
```

---

## Practical Projects

1. **Sentiment Analysis**: Build a movie review classifier
2. **Chatbot**: Create a rule-based or retrieval-based chatbot
3. **Text Summarizer**: Extract key sentences from articles
4. **Named Entity Extractor**: Extract companies, people, dates from news
5. **Language Detector**: Identify language of input text
6. **Spam Classifier**: Email spam detection

---

**Next Steps**: Move to advanced level to learn about transformers, BERT, GPT, and state-of-the-art NLP models.
