# 🧬 PCA Cancer Analysis

This project demonstrates how to use Principal Component Analysis (PCA) to reduce the dimensionality of the Breast Cancer dataset from ```scikit-learn```.
It also shows how to apply Logistic Regression on the reduced dataset for classification.

## 📌 Project Steps

1. Load Data
- We use the Breast Cancer dataset available in ```scikit-learn```.
- The dataset contains tumor features (like size, texture, etc.) and whether the tumor is malignant (cancerous) or benign.

2. Standardize Features
- Different features (like area vs. smoothness) have different scales.
- We use StandardScaler so PCA can treat them fairly.

3. Apply PCA
- PCA reduces the 30+ features into 2 main components.
- These components capture most of the important information (variance) in the data.

4. Logistic Regression (Bonus)
- We train a Logistic Regression model on the 2 PCA components.
- We then check how well it predicts tumor type.

## 📊 Output
- Explained Variance Ratio → shows how much information each PCA component keeps.
- Accuracy & Classification Report → evaluates the Logistic Regression model.

Example output:
```
Explained variance ratio:
[0.4427 0.1897]

Logistic Regression Results
Accuracy: 0.90
Classification Report:
              precision    recall  f1-score   support
           0       ...       ...      ...
           1       ...       ...      ...
```

## 🛠️ Installation & Usage
1. Clone this repository:

```
git clone https://github.com/Ebenezer-ebu/pca-cancer.git
cd pca-cancer
```

2. Install required libraries:
```
pip install scikit-learn
```

3. Run the script:
```
python pca_cancer_analysis.py
```

## 📦 Requirements

- Python 3.8+
- scikit-learn