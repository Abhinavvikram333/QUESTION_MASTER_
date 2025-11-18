# Machine Learning - Intermediate Level

## Core Machine Learning Algorithms

### 1. Linear Regression

**Purpose**: Predict continuous values by finding the best-fit line through data points.

**Mathematical Formula**:
```
y = mx + b
or
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**How it works**:
- Finds the line that minimizes the distance between predicted and actual values
- Uses Mean Squared Error (MSE) as the cost function
- Optimized using Gradient Descent

**Example Use Cases**:
- Predicting house prices based on size, location, age
- Sales forecasting
- Stock price trends

**Code Example**:
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make prediction
prediction = model.predict([[6]])  # Predicts 12
```

---

### 2. Logistic Regression

**Purpose**: Binary classification (yes/no, true/false, 0/1)

**Mathematical Formula**:
```
P(y=1) = 1 / (1 + e^(-(β₀ + β₁x₁ + ... + βₙxₙ)))
```

**Key Concept**: Uses sigmoid function to convert linear output to probability (0 to 1)

**How it works**:
- Outputs probability between 0 and 1
- Threshold (usually 0.5) determines final classification
- Uses log-loss as cost function

**Example Use Cases**:
- Email spam detection
- Customer churn prediction
- Disease diagnosis (has disease/doesn't have disease)

**Code Example**:
```python
from sklearn.linear_model import LogisticRegression

# Features: [study_hours, previous_score]
X = [[2, 60], [5, 75], [8, 85], [1, 50]]
y = [0, 1, 1, 0]  # 0=Fail, 1=Pass

model = LogisticRegression()
model.fit(X, y)

# Predict probability
prob = model.predict_proba([[6, 80]])
prediction = model.predict([[6, 80]])
```

---

### 3. Decision Trees

**Purpose**: Make decisions by asking a series of questions about features

**How it works**:
- Splits data based on features that provide maximum information gain
- Creates a tree structure with decision nodes and leaf nodes
- Easy to interpret and visualize

**Key Concepts**:
- **Entropy**: Measure of disorder/impurity in data
- **Information Gain**: Reduction in entropy after a split
- **Gini Impurity**: Alternative to entropy for measuring split quality

**Advantages**:
- Easy to understand and visualize
- Handles both numerical and categorical data
- No feature scaling required

**Disadvantages**:
- Prone to overfitting
- Can be unstable (small data changes = different tree)

**Example Use Cases**:
- Customer segmentation
- Loan approval decisions
- Medical diagnosis

**Code Example**:
```python
from sklearn.tree import DecisionTreeClassifier

X = [[25, 50000], [35, 60000], [45, 80000], [20, 30000]]
y = ['No', 'No', 'Yes', 'No']  # Bought product?

model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

prediction = model.predict([[30, 55000]])
```

---

### 4. Random Forest

**Purpose**: Ensemble of decision trees for better accuracy and reduced overfitting

**How it works**:
- Creates multiple decision trees using random subsets of data
- Each tree votes on the final prediction
- Majority vote determines the output

**Key Concepts**:
- **Bagging**: Bootstrap Aggregating - trains each tree on random sample
- **Feature Randomness**: Each split considers random subset of features
- **Ensemble Learning**: Combining multiple models for better performance

**Advantages**:
- More accurate than single decision tree
- Reduces overfitting
- Handles missing values well
- Provides feature importance

**Code Example**:
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# Get feature importance
importances = model.feature_importances_
```

---

### 5. Support Vector Machines (SVM)

**Purpose**: Find the optimal hyperplane that separates different classes

**How it works**:
- Finds the boundary that maximizes the margin between classes
- Support vectors are the data points closest to the boundary
- Can use kernel trick for non-linear boundaries

**Key Concepts**:
- **Margin**: Distance between boundary and nearest data points
- **Kernel**: Function to transform data into higher dimensions
  - Linear kernel
  - RBF (Radial Basis Function) kernel
  - Polynomial kernel

**Advantages**:
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors only)
- Versatile with different kernels

**Code Example**:
```python
from sklearn.svm import SVC

# Linear SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# RBF kernel for non-linear data
model_rbf = SVC(kernel='rbf', gamma='auto')
model_rbf.fit(X_train, y_train)
```

---

### 6. K-Nearest Neighbors (KNN)

**Purpose**: Classify based on the majority class of K nearest neighbors

**How it works**:
1. Calculate distance to all training points
2. Find K nearest neighbors
3. Take majority vote for classification
4. Or average for regression

**Key Concepts**:
- **K value**: Number of neighbors to consider
- **Distance metrics**: Euclidean, Manhattan, Minkowski
- **Lazy learning**: No training phase, computation happens at prediction

**Choosing K**:
- Small K: More sensitive to noise
- Large K: Smoother boundaries but may miss local patterns
- Use cross-validation to find optimal K

**Code Example**:
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

prediction = model.predict(X_test)
```

---

### 7. K-Means Clustering

**Purpose**: Unsupervised learning to group similar data points

**How it works**:
1. Randomly initialize K cluster centers
2. Assign each point to nearest center
3. Recalculate centers as mean of assigned points
4. Repeat until convergence

**Key Concepts**:
- **Inertia**: Sum of squared distances to nearest cluster center
- **Elbow Method**: Finding optimal K
- **Silhouette Score**: Measuring cluster quality

**Code Example**:
```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

labels = model.labels_
centers = model.cluster_centers_
```

---

## Model Evaluation Metrics

### For Classification:

1. **Accuracy**: (Correct Predictions) / (Total Predictions)
2. **Precision**: True Positives / (True Positives + False Positives)
3. **Recall**: True Positives / (True Positives + False Negatives)
4. **F1-Score**: Harmonic mean of Precision and Recall
5. **Confusion Matrix**: Table showing true vs predicted classifications

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
```

### For Regression:

1. **Mean Squared Error (MSE)**: Average of squared differences
2. **Root Mean Squared Error (RMSE)**: Square root of MSE
3. **Mean Absolute Error (MAE)**: Average of absolute differences
4. **R² Score**: Proportion of variance explained (0 to 1)

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

---

## Data Preprocessing

### 1. Handling Missing Values
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
X = imputer.fit_transform(X)
```

### 2. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalization (range 0 to 1)
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

### 3. Encoding Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encoding
le = LabelEncoder()
y = le.fit_transform(y)

# One-Hot Encoding
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)
```

---

## Cross-Validation

**Purpose**: Better estimate of model performance

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)  # 5-fold CV
mean_score = scores.mean()
```

---

## Hyperparameter Tuning

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
```

---

**Next Steps**: Move to advanced level to learn about ensemble methods, neural networks, and advanced techniques.
