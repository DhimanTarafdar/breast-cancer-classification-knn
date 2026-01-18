# K-Nearest Neighbors (KNN) - Complete Summary

## 📚 Core Concept

**KNN কী?**
- Instance-based, lazy learning algorithm
- মূল নীতি: "তুমি তোমার বন্ধুদের মতো" - Similar things exist close together
- নতুন data point classify/predict করতে K টা nearest neighbors এর vote/average নেয়

**Key Philosophy:**
- কোনো model/formula শিখে না
- সব training data মনে রাখে (memorize)
- Prediction এ সব data check করে distance দিয়ে

---

## 🎯 KNN কীভাবে কাজ করে?

### Classification এর জন্য:
1. নতুন point থেকে সব training points এর **distance** হিসাব করো
2. সবচেয়ে কাছের **K টা neighbors** বেছে নাও
3. সেই K জনের মধ্যে **majority vote** দেখো
4. যে class বেশি vote পাবে, সেটাই prediction

**Example:** Email spam detection - K=5 neighbors এর মধ্যে 4টা spam হলে → Spam!

### Regression এর জন্য:
1. একই process (distance → K neighbors)
2. Vote না নিয়ে তাদের values এর **average** নাও
3. সেই average ই prediction

**Example:** House price - K=5 neighbors এর price: 50, 52, 48, 60, 55 → Average = 53 lakh

---

## 🔑 Key Parameters

### 1. K (n_neighbors) - সবচেয়ে গুরুত্বপূর্ণ!

**Small K (K=1, K=3):**
- ✅ Local patterns ধরে ভালো
- ❌ Noise/outliers এ sensitive
- ❌ Overfitting হয় (training perfect, test খারাপ)
- **High Variance, Low Bias**

**Large K (K=50, K=100):**
- ✅ Noise থেকে stable
- ❌ Important local patterns মিস করে
- ❌ Underfitting হয় (সব জায়গায় average দেয়)
- **Low Variance, High Bias**

**Best K:**
- Medium value (সাধারণত K=5 to K=15)
- Validation/Cross-validation দিয়ে খুঁজতে হয়
- Rule of thumb: K ≈ √n (n = training samples)
- সবসময় **odd number** রাখো (tie avoid করতে)

### 2. Distance Metric

**Euclidean Distance (p=2):** [Default, বেশিরভাগ ক্ষেত্রে ভালো]
- Formula: √[(x₁-y₁)² + (x₂-y₂)² + ...]
- সরাসরি straight line distance
- Continuous features এ ভালো কাজ করে

**Manhattan Distance (p=1):**
- Formula: |x₁-y₁| + |x₂-y₂| + ...
- Grid বরাবর distance (L/R + U/D)
- High dimensional data বা outliers থাকলে better

### 3. Weights

**Uniform:** সব neighbors এর vote সমান
**Distance-weighted:** কাছের neighbor = বেশি vote (weight = 1/distance)
- Class imbalance থাকলে weighted ভালো

---

## ⚖️ Bias-Variance Tradeoff

**Bias (পক্ষপাত):**
- Model কতটা সরল/oversimplified?
- High Bias = pattern ধরতে পারে না

**Variance (অস্থিরতা):**
- Model কতটা sensitive/unstable?
- High Variance = noise কেও pattern মনে করে

**KNN তে:**
```
K ↓ (কমালে) → Bias ↓, Variance ↑ → Overfitting
K ↑ (বাড়ালে) → Bias ↑, Variance ↓ → Underfitting
```

**Goal:** Medium K দিয়ে balance করো!

---

## 🔴 Overfitting vs Underfitting

**Overfitting (K ছোট, যেমন K=1):**
- Training data perfect মনে রাখে
- Test data তে ভুল করে
- Noise ও মনে রাখে
- Example: পরীক্ষার প্রশ্ন হুবহু মুখস্থ করা

**Underfitting (K বড়, যেমন K=100):**
- কোনো pattern ই ধরে না
- সব জায়গায় average দেয়
- Training ও test দুটোতেই খারাপ
- Example: কিছুই না পড়ে পরীক্ষা দেওয়া

**Good Fit (K medium, যেমন K=5-10):**
- Pattern ধরতে পারে
- Noise ignore করে
- Test data তেও ভালো করে

---

## 🚀 Why KNN is "Lazy Learner"?

**Eager Learners (অন্য ML models):**
- Training: শিখে formula/model বানায় (সময় লাগে)
- Prediction: formula apply করে (দ্রুত)

**Lazy Learner (KNN):**
- Training: কিছু শিখে না, শুধু data store করে (instant!)
- Prediction: সব data ঘেঁটে দেখে (slow!)

**Analogy:**
- Eager = পরীক্ষার আগে পড়ে notes বানালো, পরীক্ষায় দ্রুত লিখলো
- Lazy = কিছু না পড়ে বই নিয়ে এলো, পরীক্ষায় বই খুঁজে উত্তর লিখলো

---

## ✅ When KNN Excels

- ✓ Small to medium datasets (< 10,000 samples)
- ✓ Low dimensional data (< 20 features)
- ✓ Non-linear decision boundaries
- ✓ No training time available (instant model)
- ✓ Complex/irregular patterns
- ✓ Need interpretability ("এই 5 জনও এটা করেছে, তাই...")

---

## ❌ When to Avoid KNN

- ✗ Large datasets (millions) - prediction অনেক slow
- ✗ High dimensional data (>50 features) - curse of dimensionality
- ✗ Real-time/fast prediction needed - প্রতিবার সব data check
- ✗ Class imbalance (without proper handling)
- ✗ Many irrelevant/noisy features
- ✗ Limited memory - পুরো data store করতে হয়

---

## 🛠️ Feature Scaling - Mandatory!

**কেন জরুরি?**
```
Without Scaling:
Feature 1 (Age): 20-80
Feature 2 (Income): 20000-100000

Distance = √[(5)² + (50000)²] 
→ Income dominates! Age এর effect নেই

With Scaling (StandardScaler):
Feature 1: -1.5 to +1.5
Feature 2: -1.5 to +1.5

Distance = √[(0.5)² + (0.5)²]
→ Both equal importance! ✓
```

**StandardScaler Formula:**
```
scaled = (value - mean) / std_deviation
```

**Impact:** Scaling না করলে 20-30% accuracy কমে যেতে পারে!

---

## 📊 Implementation Steps

### 1. Data Preparation
```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load data
data = load_wine()
X, y = data.data, data.target

# Split (stratify for balanced classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
```

### 2. Build Pipeline (Scaling + KNN)
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

model = Pipeline([
    ("scaler", StandardScaler()),  # Mandatory!
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
```

### 3. Train and Predict
```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 4. Evaluate
```python
from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

### 5. Tune K (Critical!)
```python
k_values = range(1, 31)
accs = []

for k in k_values:
    model_k = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    model_k.fit(X_train, y_train)
    pred = model_k.predict(X_test)
    accs.append(accuracy_score(y_test, pred))

best_k = k_values[np.argmax(accs)]
```

### 6. Compare Settings
```python
settings = [
    ("Euclidean uniform", KNeighborsClassifier(n_neighbors=best_k, p=2, weights="uniform")),
    ("Manhattan uniform", KNeighborsClassifier(n_neighbors=best_k, p=1, weights="uniform")),
    ("Euclidean weighted", KNeighborsClassifier(n_neighbors=best_k, p=2, weights="distance"))
]
```

---

## 🎓 Practical Guidelines

1. **সবসময় K odd রাখো** (tie avoid করতে)
2. **K range:** 3 ≤ K ≤ √n
3. **Start with K = √n**, তারপর tune করো
4. **Feature scaling করতেই হবে** (StandardScaler)
5. **Cross-validation ব্যবহার করো** best K খুঁজতে
6. **Class imbalance থাকলে** weighted voting ব্যবহার করো
7. **Avoid K=1** (overfitting) এবং **K=n** (underfitting)

---

## 📈 Model Selection Strategy
```
1. Quick baseline → KNN (K=5, scaled)
2. K tuning → Try 1 to 30, plot accuracy
3. Best K দিয়ে → Compare distance metrics
4. Final model → Euclidean + best K + scaling
5. Evaluate → Confusion matrix, classification report
6. Compare → With/without scaling difference
```

---

## 🔄 Validation Process

**Train-Validation-Test Split:**
```
Total Data (100%)
    ↓
Training (70%) + Test (30%)
    ↓
Training Split:
  - Training Set (80%)
  - Validation Set (20%)
```

**Cross-Validation (Better):**
- Data কে 5 ভাগে ভাগ করো
- প্রতিবার 1 ভাগ validation, বাকি 4 ভাগ training
- 5 বার repeat করো
- Average accuracy নাও

---

## 🆚 KNN vs Model-Based Learning

| Feature | KNN | Model-Based (e.g., Linear Regression) |
|---------|-----|--------------------------------------|
| Training | Instant (data store) | Takes time (learn parameters) |
| Prediction | Slow (scan all data) | Fast (apply formula) |
| Memory | High (all data) | Low (few parameters) |
| Flexibility | Very high | Limited (assumptions) |
| Interpretability | Medium | High (see coefficients) |

**Progression:**
```
KNN (simple) 
  → Logistic Regression (faster prediction)
  → Decision Trees (interpretable rules)
  → Random Forest, Neural Networks (complex)
```

---

## 💡 Key Takeaways

1. **KNN = Distance-based lazy learner** - memorizes, doesn't learn
2. **K controls complexity** - small K = complex, large K = simple
3. **Bias-Variance tradeoff** - একটা কমালে অন্যটা বাড়ে
4. **Scaling is mandatory** - নাহলে 20-30% accuracy কমে
5. **Best K through validation** - experiment করে খুঁজতে হয়
6. **Good for:** Small data, non-linear patterns, quick baseline
7. **Bad for:** Large data, high dimensions, real-time prediction
8. **Core lessons:** Distance metrics, overfitting/underfitting, hyperparameter tuning

---

## 🎯 Real-World Example (Breast Cancer)

**Dataset:** 569 patients, 30 features, 2 classes (malignant/benign)

**Results:**
- Best K = 5
- Accuracy with scaling = 97.9%
- Accuracy without scaling = 93.0%
- Improvement = +4.9%

**Insights:**
- Euclidean distance worked best
- 3 False Negatives (missed cancer cases - critical in medical context!)
- Scaling improved performance significantly
- Model ready but needs careful validation for real medical use

---
