# Machine Learning - Practice Questions

## Easy Difficulty

### Question 1
**What is Machine Learning?**

A) A type of hardware used in computers
B) A subset of AI that enables computers to learn from data without explicit programming
C) A programming language
D) A database management system

**Answer:** B

---

### Question 2
**Which of the following is an example of supervised learning?**

A) Clustering customers based on shopping behavior
B) Predicting house prices based on size and location
C) Compressing images
D) Anomaly detection without labels

**Answer:** B

---

### Question 3
**What is overfitting in machine learning?**

A) When the model performs well on both training and test data
B) When the model learns training data too well and performs poorly on new data
C) When the model is too simple
D) When there's not enough data

**Answer:** B

---

### Question 4
**What does the term "features" refer to in machine learning?**

A) The bugs in the code
B) Individual measurable properties or characteristics of the data
C) The output predictions
D) The model architecture

**Answer:** B

---

### Question 5
**Which metric is used for regression problems?**

A) Accuracy
B) Precision
C) Mean Squared Error (MSE)
D) F1-Score

**Answer:** C

---

### Question 6
**What is the purpose of splitting data into training and test sets?**

A) To make the model run faster
B) To evaluate how well the model generalizes to unseen data
C) To reduce memory usage
D) To make the code cleaner

**Answer:** B

---

### Question 7
**Which algorithm is typically used for binary classification?**

A) Linear Regression
B) K-Means
C) Logistic Regression
D) PCA

**Answer:** C

---

### Question 8
**What does a high bias model indicate?**

A) Overfitting
B) Underfitting
C) Perfect fit
D) Data leakage

**Answer:** B

---

### Question 9
**What is the range of values for R² score?**

A) 0 to 1
B) -∞ to 1
C) 0 to 100
D) -1 to 1

**Answer:** B

---

### Question 10
**Which of these is NOT a type of machine learning?**

A) Supervised Learning
B) Unsupervised Learning
C) Reinforcement Learning
D) Deterministic Learning

**Answer:** D

---

## Medium Difficulty

### Question 1
**Explain the bias-variance tradeoff and how it relates to model performance.**

**Answer:**
The bias-variance tradeoff is a fundamental concept in machine learning:

- **Bias**: Error from overly simplistic assumptions in the learning algorithm. High bias leads to underfitting.
- **Variance**: Error from sensitivity to small fluctuations in training data. High variance leads to overfitting.

**Tradeoff:**
- Simple models: High bias, low variance (underfit)
- Complex models: Low bias, high variance (overfit)
- Goal: Find optimal balance that minimizes total error

Total Error = Bias² + Variance + Irreducible Error

---

### Question 2
**What is the difference between L1 and L2 regularization? When would you use each?**

**Answer:**

**L1 Regularization (Lasso):**
- Adds penalty: λΣ|w_i|
- Can drive coefficients to exactly zero
- Performs feature selection
- Produces sparse models
- Use when: You want automatic feature selection

**L2 Regularization (Ridge):**
- Adds penalty: λΣ(w_i²)
- Shrinks coefficients but rarely to zero
- Handles multicollinearity well
- Use when: All features are relevant, dealing with correlated features

**Elastic Net:** Combines both L1 and L2

---

### Question 3
**Describe how a Random Forest algorithm works and its advantages over a single Decision Tree.**

**Answer:**

**How Random Forest Works:**
1. Creates multiple decision trees (ensemble)
2. Each tree trained on random subset of data (bagging)
3. Each split considers random subset of features
4. Final prediction: majority vote (classification) or average (regression)

**Advantages over Single Tree:**
- Reduced overfitting through ensemble averaging
- More robust to noise and outliers
- Better generalization
- Provides feature importance
- Handles missing values better
- More stable (less variance)

**Disadvantages:**
- Less interpretable
- More computational resources
- Slower prediction time

---

### Question 4
**What is cross-validation and why is it important? Explain K-Fold cross-validation.**

**Answer:**

**Cross-Validation:** Technique to assess model performance and reduce overfitting.

**K-Fold Cross-Validation:**
1. Split data into K equal folds
2. For each fold:
   - Use that fold as validation set
   - Use remaining K-1 folds as training set
   - Train and evaluate model
3. Average performance across all K iterations

**Benefits:**
- Better use of limited data
- More reliable performance estimate
- Reduces variance in performance metrics
- Detects overfitting

**Common K values:** 5 or 10

**Special case - LOOCV:** Leave-One-Out (K = n, where n is number of samples)

---

### Question 5
**Explain the confusion matrix and calculate precision, recall, and F1-score from it.**

Given confusion matrix:
```
                Predicted
              Positive  Negative
Actual Positive  120       30
       Negative   20      330
```

**Answer:**

**Confusion Matrix Components:**
- True Positives (TP) = 120
- False Negatives (FN) = 30
- False Positives (FP) = 20
- True Negatives (TN) = 330

**Metrics:**

**Precision** = TP / (TP + FP) = 120 / (120 + 20) = 0.857 (85.7%)
- "Of all predicted positives, how many are actually positive?"

**Recall** (Sensitivity) = TP / (TP + FN) = 120 / (120 + 30) = 0.800 (80.0%)
- "Of all actual positives, how many did we catch?"

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
= 2 × (0.857 × 0.800) / (0.857 + 0.800) = 0.827 (82.7%)
- Harmonic mean of precision and recall

**Accuracy** = (TP + TN) / Total = (120 + 330) / 500 = 0.900 (90.0%)

---

### Question 6
**What is the curse of dimensionality and how can it be addressed?**

**Answer:**

**Curse of Dimensionality:** As the number of features increases, the volume of the feature space increases exponentially, making data sparse and models less effective.

**Problems:**
- Data becomes increasingly sparse
- Distance metrics become less meaningful
- More data required for same statistical significance
- Increased computational complexity
- Higher risk of overfitting

**Solutions:**
1. **Feature Selection:** Remove irrelevant features
2. **Dimensionality Reduction:**
   - PCA (Principal Component Analysis)
   - t-SNE
   - Autoencoders
3. **Feature Engineering:** Create meaningful combined features
4. **Regularization:** L1/L2 to handle high dimensions
5. **Domain Knowledge:** Use expertise to select relevant features
6. **Collect More Data:** If feasible

---

### Question 7
**Compare and contrast K-Nearest Neighbors (KNN) and K-Means clustering.**

**Answer:**

**K-Nearest Neighbors (KNN):**
- **Type:** Supervised learning (classification/regression)
- **Requires:** Labeled data
- **K means:** Number of neighbors to consider
- **Training:** Lazy learning (no explicit training phase)
- **Prediction:** Based on majority vote of K nearest neighbors
- **Use case:** Classification, regression

**K-Means Clustering:**
- **Type:** Unsupervised learning (clustering)
- **Requires:** Unlabeled data
- **K means:** Number of clusters
- **Training:** Iterative algorithm (finds cluster centers)
- **Prediction:** Assigns points to nearest cluster center
- **Use case:** Grouping similar data points

**Similarities:**
- Both use distance metrics
- Both have hyperparameter "K"
- Both sensitive to feature scaling

**Key Difference:** KNN uses labels, K-Means discovers patterns without labels

---

### Question 8
**What is gradient descent and how does it work? Explain different variants.**

**Answer:**

**Gradient Descent:** Optimization algorithm to minimize loss function by iteratively moving in direction of steepest descent.

**Algorithm:**
```
repeat:
    θ = θ - α × ∇J(θ)
where:
    θ = parameters
    α = learning rate
    ∇J(θ) = gradient of loss function
```

**Variants:**

**1. Batch Gradient Descent:**
- Uses entire dataset for each update
- Pros: Stable convergence, accurate gradient
- Cons: Slow for large datasets, can get stuck in local minima

**2. Stochastic Gradient Descent (SGD):**
- Uses one sample at a time
- Pros: Fast updates, can escape local minima
- Cons: Noisy updates, doesn't converge smoothly

**3. Mini-Batch Gradient Descent:**
- Uses small batches (e.g., 32, 64, 128)
- Pros: Balance between batch and SGD, efficient computation
- Cons: Requires tuning batch size

**Advanced Variants:**
- **Momentum:** Accelerates in consistent directions
- **Adam:** Adaptive learning rates per parameter
- **RMSprop:** Adapts learning rate based on recent gradients

---

### Question 9
**Implement a simple linear regression from scratch using gradient descent.**

**Answer:**

```python
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Usage
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

predictions = model.predict(np.array([[6], [7]]))
print(predictions)  # Should be close to [12, 14]
```

---

### Question 10
**What is the difference between bagging and boosting? Provide examples of algorithms for each.**

**Answer:**

**Bagging (Bootstrap Aggregating):**
- **Concept:** Train multiple models independently on random subsets (with replacement)
- **Combination:** Average predictions (regression) or vote (classification)
- **Goal:** Reduce variance
- **Parallelizable:** Yes
- **Examples:**
  - Random Forest
  - Bagged Decision Trees

**Boosting:**
- **Concept:** Train models sequentially, each correcting errors of previous
- **Combination:** Weighted combination of models
- **Goal:** Reduce bias and variance
- **Parallelizable:** No
- **Examples:**
  - AdaBoost
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - CatBoost

**Key Differences:**

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Focus | Reduce variance | Reduce bias |
| Weighting | Equal weight | Weighted by performance |
| Overfitting | Less prone | More prone if not careful |

---

## Hard Difficulty

### Question 1
**Derive the closed-form solution for Linear Regression (Normal Equation) and explain when to use it vs. gradient descent.**

**Answer:**

**Derivation:**

Starting with the loss function (MSE):
```
J(θ) = (1/2m) Σ(h_θ(x^(i)) - y^(i))²

Where h_θ(x) = θ^T x
```

In matrix form:
```
J(θ) = (1/2m)(Xθ - y)^T(Xθ - y)
```

To minimize, take derivative and set to zero:
```
∇_θ J(θ) = (1/m)X^T(Xθ - y) = 0
X^T Xθ = X^T y
θ = (X^T X)^(-1) X^T y  ← Normal Equation
```

**When to Use:**

**Normal Equation:**
- **Pros:**
  - No need to choose learning rate
  - No iterations needed
  - Exact solution (not approximate)
- **Cons:**
  - Requires computing (X^T X)^(-1): O(n³) complexity
  - Slow if n > 10,000 features
  - Fails if X^T X is singular (not invertible)

**Gradient Descent:**
- **Pros:**
  - Works well with large n
  - O(kn²) complexity where k = iterations
  - Can use mini-batches
- **Cons:**
  - Requires choosing learning rate
  - Requires multiple iterations
  - May not find exact solution

**Recommendation:** Use Normal Equation if n < 10,000, otherwise use Gradient Descent

---

### Question 2
**Explain Support Vector Machines (SVM) with both linear and non-linear kernels. Derive the optimization problem.**

**Answer:**

**Linear SVM:**

**Goal:** Find hyperplane that maximizes margin between classes.

**Primal Problem:**
```
minimize: (1/2)||w||²
subject to: y^(i)(w^T x^(i) + b) ≥ 1 for all i
```

**Dual Problem (using Lagrange multipliers):**
```
maximize: Σα_i - (1/2)ΣΣα_i α_j y^(i) y^(j) x^(i)^T x^(j)
subject to: α_i ≥ 0 and Σα_i y^(i) = 0
```

**Decision function:**
```
f(x) = sign(w^T x + b)
where w = Σα_i y^(i) x^(i)
```

**Non-Linear SVM (Kernel Trick):**

Replace dot product x^(i)^T x^(j) with kernel function K(x^(i), x^(j)):

**Common Kernels:**

1. **Polynomial:** K(x, x') = (γx^T x' + r)^d
2. **RBF (Gaussian):** K(x, x') = exp(-γ||x - x'||²)
3. **Sigmoid:** K(x, x') = tanh(γx^T x' + r)

**Why Kernel Trick Works:**
- Implicitly maps data to higher-dimensional space
- Computationally efficient (avoid explicit transformation)
- Makes linearly inseparable data separable

**Soft Margin SVM (for non-separable data):**
```
minimize: (1/2)||w||² + C Σξ_i
subject to: y^(i)(w^T x^(i) + b) ≥ 1 - ξ_i
           ξ_i ≥ 0
```
Where C controls tradeoff between margin and violations.

---

### Question 3
**Implement and explain the AdaBoost algorithm. Why does it work?**

**Answer:**

**AdaBoost (Adaptive Boosting) Algorithm:**

```python
import numpy as np

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples = X.shape[0]

        # Initialize weights uniformly
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # Train weak learner on weighted data
            model = DecisionTreeClassifier(max_depth=1)  # Stump
            model.fit(X, y, sample_weight=weights)

            # Make predictions
            predictions = model.predict(X)

            # Calculate weighted error
            incorrect = predictions != y
            error = np.sum(weights * incorrect) / np.sum(weights)

            # Calculate alpha (model weight)
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            # Update weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)  # Normalize

            # Store model and alpha
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        # Weighted majority vote
        predictions = np.zeros(X.shape[0])

        for alpha, model in zip(self.alphas, self.models):
            predictions += alpha * model.predict(X)

        return np.sign(predictions)
```

**Why AdaBoost Works:**

1. **Adaptive Weighting:**
   - Increases weights on misclassified examples
   - Forces subsequent models to focus on hard cases

2. **Weighted Voting:**
   - Better models get more say (higher α)
   - Combines weak learners into strong learner

3. **Theoretical Guarantee:**
   - Training error decreases exponentially
   - Provably reduces both bias and variance

4. **Mathematics:**
   - Minimizes exponential loss
   - Each iteration greedily reduces training error

**Key Properties:**
- Works with any base classifier
- Resistant to overfitting (though can overfit with noisy data)
- No hyperparameters except number of estimators
- Can achieve arbitrary accuracy with enough weak learners

---

### Question 4
**Explain Principal Component Analysis (PCA) mathematically. How would you implement it from scratch?**

**Answer:**

**PCA Mathematical Foundation:**

**Goal:** Find orthogonal directions of maximum variance.

**Steps:**

1. **Center the data:**
   ```
   X_centered = X - mean(X)
   ```

2. **Compute covariance matrix:**
   ```
   Σ = (1/n) X_centered^T X_centered
   ```

3. **Eigendecomposition:**
   ```
   Σv = λv
   ```
   Where v are eigenvectors, λ are eigenvalues

4. **Select top k eigenvectors:**
   - Sort by eigenvalues (descending)
   - Select k eigenvectors with largest eigenvalues
   - These form the principal components

5. **Project data:**
   ```
   Z = X_centered × W
   ```
   Where W is matrix of k eigenvectors

**Implementation:**

```python
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        # Center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort by eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select top k components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z):
        return np.dot(Z, self.components.T) + self.mean

    def explained_variance_ratio(self):
        return self.explained_variance / np.sum(self.explained_variance)

# Usage
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)
print(f"Explained variance: {pca.explained_variance_ratio()}")
```

**Alternative: SVD-based PCA (more efficient)**

```python
def pca_svd(X, n_components):
    # Center data
    X_centered = X - np.mean(X, axis=0)

    # SVD: X = UΣV^T
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Principal components are columns of V
    components = Vt.T[:, :n_components]

    # Transform
    X_transformed = np.dot(X_centered, components)

    # Explained variance
    explained_variance = (S ** 2) / (X.shape[0] - 1)

    return X_transformed, components, explained_variance[:n_components]
```

**Use Cases:**
- Dimensionality reduction
- Visualization (reduce to 2D/3D)
- Noise reduction
- Feature extraction
- Data compression

---

### Question 5
**Design and implement a complete machine learning pipeline including data preprocessing, model selection, hyperparameter tuning, and evaluation.**

**Answer:**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

class MLPipeline:
    def __init__(self, task_type='classification'):
        self.task_type = task_type
        self.best_model = None
        self.scaler = None
        self.label_encoder = None

    def preprocess_data(self, df, target_column):
        """
        Complete data preprocessing
        """
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Encode target if classification
        if self.task_type == 'classification':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        return X, y

    def feature_engineering(self, X):
        """
        Create new features
        """
        # Example: polynomial features
        X_eng = X.copy()

        # Add interaction terms for top features
        if X.shape[1] >= 2:
            X_eng['feature_0_1_interaction'] = X.iloc[:, 0] * X.iloc[:, 1]

        # Add squared terms
        for col in X.columns[:3]:  # Top 3 features
            X_eng[f'{col}_squared'] = X[col] ** 2

        return X_eng

    def select_best_model(self, X_train, y_train, X_val, y_val):
        """
        Compare multiple models
        """
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }

        results = {}

        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

            # Fit on training data
            model.fit(X_train, y_train)

            # Evaluate on validation
            val_score = model.score(X_val, y_val)

            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'val_score': val_score
            }

            print(f"{name}:")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"  Val Score: {val_score:.4f}\n")

        # Select best model
        best_name = max(results, key=lambda k: results[k]['val_score'])
        print(f"Best Model: {best_name}")

        return results[best_name]['model']

    def hyperparameter_tuning(self, model, X_train, y_train):
        """
        Tune hyperparameters using GridSearch
        """
        if isinstance(model, RandomForestClassifier):
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif isinstance(model, GradientBoostingClassifier):
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif isinstance(model, SVC):
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        else:
            return model

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def evaluate_model(self, model, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # ROC AUC (for binary classification)
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            print(f"\nROC AUC Score: {auc:.4f}")

        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }

    def save_model(self, filename):
        """Save trained model"""
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }, filename)

    def load_model(self, filename):
        """Load trained model"""
        data = joblib.load(filename)
        self.best_model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']

    def run_complete_pipeline(self, df, target_column):
        """
        Execute complete ML pipeline
        """
        print("=" * 50)
        print("MACHINE LEARNING PIPELINE")
        print("=" * 50)

        # 1. Preprocessing
        print("\n1. Preprocessing data...")
        X, y = self.preprocess_data(df, target_column)

        # 2. Feature Engineering
        print("\n2. Feature engineering...")
        X = self.feature_engineering(X)

        # 3. Split data
        print("\n3. Splitting data...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # 4. Scale features
        print("\n4. Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # 5. Model selection
        print("\n5. Selecting best model...")
        best_model = self.select_best_model(
            X_train_scaled, y_train,
            X_val_scaled, y_val
        )

        # 6. Hyperparameter tuning
        print("\n6. Tuning hyperparameters...")
        self.best_model = self.hyperparameter_tuning(
            best_model, X_train_scaled, y_train
        )

        # 7. Final evaluation
        print("\n7. Final evaluation on test set...")
        results = self.evaluate_model(
            self.best_model, X_test_scaled, y_test
        )

        # 8. Save model
        print("\n8. Saving model...")
        self.save_model('best_model.pkl')

        print("\n" + "=" * 50)
        print("PIPELINE COMPLETE!")
        print("=" * 50)

        return results

# Usage
# pipeline = MLPipeline(task_type='classification')
# results = pipeline.run_complete_pipeline(df, 'target_column')
```

This pipeline includes:
- Data preprocessing (missing values, encoding)
- Feature engineering
- Model selection with cross-validation
- Hyperparameter tuning
- Comprehensive evaluation
- Model persistence
