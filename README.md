# 🌳 Decision Tree and Random Forest Analysis

This project demonstrates the use of **Decision Trees** and **Random Forests** for classification and regression tasks.  
It explores how tree-based algorithms split data, visualize decision boundaries, and use ensemble learning for better accuracy and robustness.

---

## 📘 Overview

This notebook covers:
- Decision Tree basics  
- Visualization of tree structure  
- Regularization (depth, min samples)  
- Feature importance  
- Random Forest and Ensemble learning  
- Comparison of model performance

---

## 🧰 Tools & Libraries

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib / Seaborn**
- **Scikit-learn** (`DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`)

---

## ⚙️ Workflow

### 1. Data Preparation
- Loaded and preprocessed dataset (cleaning, encoding, scaling if required).
- Split into **training** and **testing** sets using `train_test_split`.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 2. Decision Tree Classifier
- Implemented a **Decision Tree** model for classification.
- Controlled overfitting using:
  - `max_depth`
  - `min_samples_split`
  - `criterion` = "gini" or "entropy"

```python
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train, y_train)
y_pred = tree_clf.predict(X_test)
```

- Visualized decision boundaries and tree structure using `plot_tree()` or `export_graphviz()`.

---

### 3. Decision Tree Regressor
- Used for continuous output prediction.
- Compared performance using **Mean Squared Error (MSE)** and **R² score**.

```python
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=3)
tree_reg.fit(X_train, y_train)
```

---

### 4. Random Forest Ensemble
- Combined multiple decision trees using **bagging** to reduce variance.
- Compared accuracy between single Decision Tree and Random Forest.

```python
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
forest_clf.fit(X_train, y_train)
```

- Displayed **feature importance** to interpret model insights.

---

### 5. Model Evaluation
- Used metrics such as:
  - **Accuracy Score**
  - **Confusion Matrix**
  - **Precision, Recall, F1-score**
  - **Cross-validation scores**

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
```

---

## 📊 Results & Insights

- **Decision Tree Classifier** is easy to interpret but prone to overfitting.  
- **Random Forest** improves stability and accuracy by averaging multiple trees.  
- Deeper trees → higher accuracy on training but lower generalization.  
- Feature importance visualization provides interpretability.  
- Ensemble methods consistently outperform a single decision tree.

---

## 🧩 Key Takeaways

- **Decision Trees** split data recursively to minimize impurity (Gini or Entropy).  
- **Regularization** (e.g., limiting depth) prevents overfitting.  
- **Random Forests** average multiple trees to reduce variance and improve performance.  
- Feature importance helps identify which variables have the most predictive power.  

---


## 📈 Future Enhancements

- Implement **Gradient Boosting** and **XGBoost** for comparison.  
- Add **hyperparameter tuning** using Grid Search / Randomized Search.  
- Visualize **feature importance** and **decision surfaces** for each tree.  
- Apply to a **real-world dataset** (e.g., loan approval, medical diagnosis, or churn prediction).

---

## 🏁 Conclusion

This project demonstrates how **tree-based models** balance interpretability and performance.  
By combining them into ensembles like Random Forests, we gain robustness and higher predictive power — making them essential tools for both beginners and applied ML practitioners.
