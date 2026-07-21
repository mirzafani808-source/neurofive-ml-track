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
