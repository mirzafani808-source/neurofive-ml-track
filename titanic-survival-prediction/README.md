# Predict Titanic Survival — My First Classification Model

## Overview
This project builds a Logistic Regression model to predict whether a
Titanic passenger survived, using the classic Titanic dataset (891
passengers).

## Approach

1. **Load & clean the data**
   - Filled missing Age values with the median age.
   - Filled missing Embarked values with the most common port (mode).
   - Dropped PassengerId, Name, Ticket, and Cabin columns.

2. **Encode categorical columns**
   - Used pd.get_dummies() on Sex and Embarked, with drop_first=True.

3. **Split the data**
   - Used train_test_split: 80% train / 20% test, stratified.

4. **Train the model**
   - Trained a LogisticRegression model on the training set.

5. **Evaluate**
   - Measured accuracy with accuracy_score.
   - Generated a confusion matrix with confusion_matrix.

## Results

**Accuracy: 80.4%**

### Confusion Matrix

|                     | Predicted: Died | Predicted: Survived |
|---------------------|:---------------:|:--------------------:|
| **Actual: Died**    | 98               | 12                    |
| **Actual: Survived**| 23               | 46                    |

**What this tells us:**
- 98 true negatives (correctly predicted deaths)
- 46 true positives (correctly predicted survivors)
- 12 false positives (predicted survived, actually died)
- 23 false negatives (predicted died, actually survived) — the model's
  biggest weakness is missing actual survivors.

### Key Insight
Sex was the strongest predictor of survival, followed by Pclass —
consistent with "women and children first" and class-based access to
lifeboats.
## Model Evaluation & Hyperparameter Tuning

### Why Accuracy Alone Can Be Misleading
This dataset has imbalanced classes — about 62% of passengers died and
38% survived. A model that always predicts "Died" would still score
~62% accuracy while being completely useless, since it would never
correctly identify a single survivor. This is why Precision, Recall,
and F1-score matter more for imbalanced data:
- **Precision** — of all passengers predicted to survive, how many
  actually did?
- **Recall** — of all passengers who actually survived, how many did
  the model catch?
- **F1-score** — the balance between precision and recall.

### Hyperparameter Tuning (GridSearchCV)
Tuned two hyperparameters of the Logistic Regression model:
- **C** (inverse regularization strength): [0.01, 0.1, 1, 10, 100]
- **solver**: liblinear, lbfgs

Used 5-fold cross-validation, optimizing for F1-score.

**Best hyperparameters found:** `C=0.1`, `solver='lbfgs'`

### Before vs After Comparison

| Metric    | Baseline Model | Tuned Model | Change   |
|-----------|:---------------:|:------------:|:--------:|
| Accuracy  | 0.8045          | 0.7989       | -0.0056  |
| Precision | 0.7931          | 0.8000       | +0.0069  |
| Recall    | 0.6667          | 0.6377       | -0.0290  |
| F1-score  | 0.7244          | 0.7097       | -0.0147  |

### What This Tells Us
In this case, hyperparameter tuning did **not** improve the model —
performance stayed roughly the same, with a very slight drop in
F1-score. This is a realistic outcome: tuning doesn't always help.
Possible reasons include a small test set (only 179 samples) and the
default Logistic Regression settings already being close to optimal
for this dataset. The exercise still adds value — it confirms the
baseline model's settings were reasonable rather than leaving that to
guesswork, which is the entire point of systematic tuning.
