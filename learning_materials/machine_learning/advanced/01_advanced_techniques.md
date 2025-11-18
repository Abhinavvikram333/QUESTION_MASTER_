# Machine Learning - Advanced Level

## Advanced Ensemble Methods

### 1. Gradient Boosting

**Concept**: Build models sequentially, where each new model corrects errors of previous models.

**Key Algorithms**:

#### XGBoost (Extreme Gradient Boosting)
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Classification
model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    objective='binary:logistic',
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1
)

model.fit(X_train, y_train,
         eval_set=[(X_val, y_val)],
         early_stopping_rounds=10,
         verbose=True)

# Feature importance
importance = model.feature_importances_
```

#### LightGBM
```python
import lightgbm as lgb

# Create dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

model = lgb.train(params,
                 train_data,
                 num_boost_round=100,
                 valid_sets=[val_data],
                 early_stopping_rounds=10)
```

#### CatBoost
```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    cat_features=['category_col1', 'category_col2']
)

model.fit(X_train, y_train,
         eval_set=(X_val, y_val),
         early_stopping_rounds=10,
         verbose=100)
```

**Key Differences**:
- **XGBoost**: Most popular, excellent performance, requires careful tuning
- **LightGBM**: Faster training, better with large datasets, leaf-wise tree growth
- **CatBoost**: Handles categorical features natively, robust to overfitting

---

### 2. Stacking and Blending

**Stacking**: Use predictions from multiple models as features for a meta-model

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('svm', SVC(probability=True)),
    ('dt', DecisionTreeClassifier())
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

stacking_model.fit(X_train, y_train)
```

**Custom Stacking Implementation**:
```python
# Level 0 models
model1 = RandomForestClassifier()
model2 = GradientBoostingClassifier()
model3 = SVC(probability=True)

# Train level 0
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# Create meta-features
meta_X_train = np.column_stack([
    model1.predict_proba(X_val)[:, 1],
    model2.predict_proba(X_val)[:, 1],
    model3.predict_proba(X_val)[:, 1]
])

# Level 1 model
meta_model = LogisticRegression()
meta_model.fit(meta_X_train, y_val)
```

---

## Advanced Feature Engineering

### 1. Polynomial Features
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# For features [a, b]:
# Output: [a, b, a², ab, b²]
```

### 2. Feature Selection Techniques

#### Recursive Feature Elimination (RFE)
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

selected_features = X.columns[rfe.support_]
```

#### Feature Importance-based Selection
```python
from sklearn.feature_selection import SelectFromModel

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

selector = SelectFromModel(model, prefit=True, threshold='median')
X_selected = selector.transform(X)
```

#### Mutual Information
```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns)
mi_scores.sort_values(ascending=False)
```

### 3. Advanced Encoding Techniques

#### Target Encoding
```python
import category_encoders as ce

encoder = ce.TargetEncoder(cols=['category_col'])
X_encoded = encoder.fit_transform(X, y)
```

#### Frequency Encoding
```python
freq_encoding = X['category'].value_counts().to_dict()
X['category_freq'] = X['category'].map(freq_encoding)
```

#### Embedding Encoding (for high cardinality)
```python
# Using entity embeddings with neural networks
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

# For categorical feature with 1000 unique values
input_cat = Input(shape=(1,))
embedding = Embedding(input_dim=1000, output_dim=10)(input_cat)
flatten = Flatten()(embedding)
```

---

## Handling Imbalanced Data

### 1. Resampling Techniques

#### SMOTE (Synthetic Minority Over-sampling Technique)
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### ADASYN (Adaptive Synthetic Sampling)
```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(sampling_strategy='minority')
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
```

#### Combined Over/Under Sampling
```python
from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek()
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
```

### 2. Class Weight Adjustment
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)

model = RandomForestClassifier(class_weight='balanced')
```

### 3. Anomaly Detection for Imbalanced Data
```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.1, random_state=42)
predictions = iso_forest.fit_predict(X)
# -1 for outliers, 1 for inliers
```

---

## Advanced Model Optimization

### 1. Bayesian Optimization
```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

search_spaces = {
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 10),
    'n_estimators': Integer(50, 500),
    'subsample': Real(0.6, 1.0)
}

bayes_search = BayesSearchCV(
    XGBClassifier(),
    search_spaces,
    n_iter=50,
    cv=5,
    n_jobs=-1
)

bayes_search.fit(X_train, y_train)
best_params = bayes_search.best_params_
```

### 2. Optuna for Hyperparameter Tuning
```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0)
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)

    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
```

---

## Calibration and Uncertainty

### 1. Probability Calibration
```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate probability estimates
calibrated_model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
calibrated_model.fit(X_train, y_train)

# Get calibrated probabilities
calibrated_probs = calibrated_model.predict_proba(X_test)
```

### 2. Confidence Intervals
```python
from sklearn.ensemble import GradientBoostingRegressor

# Quantile regression for prediction intervals
models = {}
for alpha in [0.05, 0.5, 0.95]:
    models[alpha] = GradientBoostingRegressor(loss='quantile', alpha=alpha)
    models[alpha].fit(X_train, y_train)

# Predict with confidence intervals
lower = models[0.05].predict(X_test)
median = models[0.5].predict(X_test)
upper = models[0.95].predict(X_test)
```

---

## Interpretability and Explainability

### 1. SHAP (SHapley Additive exPlanations)
```python
import shap

# For tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Force plot for single prediction
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Dependence plot
shap.dependence_plot('feature_name', shap_values, X_test)
```

### 2. LIME (Local Interpretable Model-agnostic Explanations)
```python
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['class_0', 'class_1'],
    mode='classification'
)

# Explain single prediction
exp = explainer.explain_instance(X_test.iloc[0].values,
                                 model.predict_proba,
                                 num_features=10)
exp.show_in_notebook()
```

### 3. Partial Dependence Plots
```python
from sklearn.inspection import PartialDependenceDisplay

features = ['feature1', 'feature2', ('feature1', 'feature2')]
PartialDependenceDisplay.from_estimator(model, X_train, features)
```

---

## Advanced Cross-Validation Strategies

### 1. Stratified K-Fold for Imbalanced Data
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
```

### 2. Time Series Cross-Validation
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

### 3. Group K-Fold (for grouped data)
```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)

for train_idx, val_idx in gkf.split(X, y, groups):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
```

---

## Online Learning and Incremental Learning

```python
from sklearn.linear_model import SGDClassifier

# Initialize model
model = SGDClassifier(loss='log', learning_rate='optimal')

# Partial fit with batches
for X_batch, y_batch in data_batches:
    model.partial_fit(X_batch, y_batch, classes=np.unique(y))
```

---

## Advanced Regularization Techniques

### 1. Elastic Net (L1 + L2)
```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio: mix of L1 and L2
model.fit(X_train, y_train)
```

### 2. Dropout (Neural Networks)
```python
from tensorflow.keras.layers import Dropout

model = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.5),  # Randomly drops 50% of neurons
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

---

## AutoML and Neural Architecture Search

### 1. Auto-sklearn
```python
import autosklearn.classification

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=3600,  # 1 hour
    per_run_time_limit=300,
    memory_limit=3072
)

automl.fit(X_train, y_train)
predictions = automl.predict(X_test)
```

### 2. TPOT (Tree-based Pipeline Optimization Tool)
```python
from tpot import TPOTClassifier

tpot = TPOTClassifier(
    generations=5,
    population_size=50,
    verbosity=2,
    random_state=42
)

tpot.fit(X_train, y_train)
tpot.export('best_pipeline.py')
```

---

## Model Deployment Considerations

### 1. Model Serialization
```python
import joblib
import pickle

# Joblib (recommended for sklearn)
joblib.dump(model, 'model.pkl')
loaded_model = joblib.load('model.pkl')

# Pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### 2. ONNX Export (for cross-platform deployment)
```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

---

## Advanced Metrics and Evaluation

### 1. ROC-AUC for Multi-class
```python
from sklearn.metrics import roc_auc_score

# One-vs-Rest
auc_ovr = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

# One-vs-One
auc_ovo = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')
```

### 2. Matthews Correlation Coefficient (for imbalanced data)
```python
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_true, y_pred)
# Range: -1 to 1, where 1 is perfect prediction
```

### 3. Cohen's Kappa
```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(y_true, y_pred)
# Measures inter-rater agreement
```

---

## Practical Tips for Production ML

1. **Version Control**: Track data, code, and models
2. **Monitoring**: Track model performance drift
3. **A/B Testing**: Compare model versions in production
4. **Feature Store**: Centralized feature management
5. **Model Registry**: Catalog of trained models
6. **Continuous Training**: Retrain on new data
7. **Explainability**: Ensure model decisions are interpretable
8. **Fairness**: Check for bias across demographic groups

---

**Next Steps**: Explore Deep Learning for neural network architectures and advanced AI techniques.
