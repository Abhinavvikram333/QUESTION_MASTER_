# Natural Language Processing - Basic Level

## What is Natural Language Processing (NLP)?

Natural Language Processing is a branch of artificial intelligence that helps computers understand, interpret, and generate human language. It bridges the gap between human communication and computer understanding.

## Why is NLP Important?

Humans communicate primarily through language, but computers work with numbers and structured data. NLP enables:
- Computers to understand text and speech
- Automatic translation between languages
- Sentiment analysis (understanding emotions in text)
- Chatbots and virtual assistants
- Text summarization and information extraction

## Basic Text Concepts

### 1. Corpus
A large collection of text documents used for analysis.
- **Example**: All Wikipedia articles, customer reviews, tweets

### 2. Document
A single piece of text (article, email, tweet, etc.)

### 3. Tokens
Individual units of text (usually words)
- **Sentence**: "I love machine learning"
- **Tokens**: ["I", "love", "machine", "learning"]

### 4. Vocabulary
The set of unique words in a corpus

---

## Basic NLP Tasks

### 1. Tokenization
Breaking text into individual words or sentences.

**Example**:
```
Text: "Hello! How are you?"
Word Tokens: ["Hello", "!", "How", "are", "you", "?"]
Sentence Tokens: ["Hello!", "How are you?"]
```

**Python Example**:
```python
text = "I love NLP. It's amazing!"

# Simple word tokenization
words = text.split()
print(words)  # ['I', 'love', 'NLP.', "It's", 'amazing!']

# Using NLTK
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

words = word_tokenize(text)
sentences = sent_tokenize(text)
```

### 2. Lowercasing
Converting all text to lowercase for consistency.

**Example**:
```
Original: "Hello World"
Lowercase: "hello world"
```

Why? "Hello" and "hello" should be treated as the same word.

### 3. Stop Words Removal
Removing common words that don't carry much meaning.

**Common Stop Words**: the, is, are, and, or, a, an, in, on, at

**Example**:
```
Original: "The cat is sitting on the mat"
After removal: "cat sitting mat"
```

**Python Example**:
```python
from nltk.corpus import stopwords

text = "The cat is sitting on the mat"
words = text.lower().split()

stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w not in stop_words]
print(filtered_words)  # ['cat', 'sitting', 'mat']
```

### 4. Stemming
Reducing words to their root form (may not be actual words).

**Examples**:
- running → run
- cats → cat
- better → better
- studies → studi (note: not a real word)

**Python Example**:
```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["running", "runs", "ran", "runner"]

stemmed = [stemmer.stem(word) for word in words]
print(stemmed)  # ['run', 'run', 'ran', 'runner']
```

### 5. Lemmatization
Reducing words to their base dictionary form (always real words).

**Examples**:
- running → run
- better → good
- studies → study
- am, are, is → be

**Python Example**:
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ["running", "runs", "ran", "better", "studies"]

lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in words]
print(lemmatized)
```

---

## Text Representation

### 1. Bag of Words (BoW)
Represents text as a collection of word frequencies, ignoring grammar and order.

**Example**:
```
Sentence 1: "I love cats"
Sentence 2: "I love dogs"

Vocabulary: [I, love, cats, dogs]

Sentence 1 vector: [1, 1, 1, 0]
Sentence 2 vector: [1, 1, 0, 1]
```

**Python Example**:
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love cats",
    "I love dogs",
    "Dogs are great"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

### 2. Term Frequency (TF)
How often a word appears in a document.

```
TF(word) = (Number of times word appears) / (Total words in document)
```

### 3. TF-IDF (Term Frequency-Inverse Document Frequency)
Measures how important a word is to a document in a collection.

- **High TF-IDF**: Word is frequent in this document but rare in others (important!)
- **Low TF-IDF**: Word is common everywhere (less important)

**Python Example**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "I love cats and dogs",
    "I love machine learning",
    "Machine learning is great"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

---

## Basic NLP Applications

### 1. Sentiment Analysis
Determining if text expresses positive, negative, or neutral sentiment.

**Examples**:
- "This movie is amazing!" → Positive
- "I hate waiting in lines" → Negative
- "The sky is blue" → Neutral

### 2. Text Classification
Categorizing text into predefined categories.

**Examples**:
- Email: Spam or Not Spam
- News: Sports, Politics, Technology, Entertainment
- Customer Review: 1-5 stars

### 3. Named Entity Recognition (NER)
Identifying and classifying named entities (people, places, organizations).

**Example**:
```
Text: "Apple is looking to buy a startup in San Francisco"

Entities:
- Apple: Organization
- San Francisco: Location
```

### 4. Language Detection
Identifying the language of text.

**Example**:
- "Hello, how are you?" → English
- "Bonjour, comment allez-vous?" → French
- "Hola, ¿cómo estás?" → Spanish

---

## Getting Started with NLP

### Essential Python Libraries

1. **NLTK (Natural Language Toolkit)**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

2. **spaCy** (Fast and production-ready)
```python
import spacy
nlp = spacy.load('en_core_web_sm')
```

3. **TextBlob** (Simple API)
```python
from textblob import TextBlob
```

### Your First NLP Program

```python
from textblob import TextBlob

# Create a TextBlob object
text = "I love learning about natural language processing!"
blob = TextBlob(text)

# Tokenization
print("Words:", blob.words)
print("Sentences:", blob.sentences)

# Sentiment Analysis (polarity: -1 to 1)
print("Sentiment:", blob.sentiment.polarity)

# Part-of-speech tagging
print("POS tags:", blob.tags)

# Noun phrases
print("Noun phrases:", blob.noun_phrases)
```

---

## Common Challenges in NLP

### 1. Ambiguity
**Example**: "I saw her duck"
- Did I see her pet duck?
- Did I see her duck down?

### 2. Context Dependency
**Example**: "That's sick!"
- In medical context: someone is ill
- In slang: something is cool/awesome

### 3. Sarcasm and Irony
**Example**: "Oh great, another rainy day" (probably negative despite "great")

### 4. Spelling and Grammar Errors
**Example**: "i luv nlp its awsome"

### 5. Multiple Languages
Mixing languages in the same text (code-switching)

---

## Real-World Applications

1. **Chatbots**: Customer service, virtual assistants (Siri, Alexa)
2. **Search Engines**: Understanding user queries
3. **Machine Translation**: Google Translate
4. **Email Filtering**: Spam detection
5. **Social Media**: Sentiment analysis on tweets
6. **Healthcare**: Extracting information from medical records
7. **Finance**: Analyzing news for stock market predictions
8. **Content Recommendation**: Suggesting articles or products

---

## Practice Exercise

**Task**: Analyze customer reviews

Given reviews:
1. "This product is excellent! I love it."
2. "Terrible quality, waste of money."
3. "It's okay, nothing special."

**Questions**:
1. Tokenize each review
2. Remove stop words
3. Identify sentiment (positive/negative/neutral)
4. Find the most common words

---

## Key Takeaways

- NLP helps computers understand human language
- Text preprocessing (tokenization, lowercasing, stop word removal) is crucial
- Text must be converted to numbers (vectors) for ML algorithms
- Common tasks: sentiment analysis, classification, NER
- Start with simple libraries like TextBlob or NLTK

---

**Next Steps**: Move to intermediate level to learn about word embeddings, advanced text processing, and building NLP models.
