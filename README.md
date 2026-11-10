# RetailMind AI – Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn in the telecommunications industry. RetailMind AI uses customer demographics, account information, service subscriptions, and billing details to identify customers who are likely to discontinue their services.

The objective is to help businesses proactively identify at-risk customers and develop targeted customer-retention strategies.

---

## Project Objectives

- Analyse customer data to identify churn patterns.
- Build machine learning models to predict customer churn.
- Address class imbalance in the dataset.
- Perform feature engineering to improve predictive performance.
- Compare multiple classification algorithms.
- Apply hyperparameter tuning and cross-validation.
- Explore ensemble learning techniques.
- Develop a reusable prediction pipeline.

## Dataset

**Dataset:** IBM Telco Customer Churn Dataset  
**Source:** [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Attribute        | Description           |
| ---------------- | --------------------- |
| Initial records  | 7,043                 |
| Initial features | 21                    |
| Target variable  | Churn                 |
| Target classes   | Yes / No              |
| Prediction type  | Binary Classification |

The target variable was encoded as:

- `0` – Customer did not churn.
- `1` – Customer churned.

The `customerID` column was excluded from model training because it is an identifier rather than a predictive feature.

## Technology Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Ensemble Learning:** Random Forest, XGBoost
- **Model Persistence:** Joblib
- **Development Environment:** Jupyter Notebook, VS Code
- **Version Control:** Git and GitHub

---

## Project Development

### Day 1 – Data Preprocessing and Baseline Model

The initial stage focused on understanding the dataset, handling missing values, and establishing a baseline machine learning model.

**Key activities:**

- Exploratory Data Analysis (EDA).
- Identification and treatment of missing values.
- Conversion of `TotalCharges` into a numeric feature.
- Encoding the target variable.
- Stratified train-test split (80:20).
- Development of a reusable preprocessing pipeline.

**Baseline model:** Logistic Regression

The preprocessing pipeline included:

| Feature Type | Preprocessing                              |
| ------------ | ------------------------------------------ |
| Numerical    | Median imputation and StandardScaler       |
| Categorical  | Most-frequent imputation and OneHotEncoder |

Class imbalance was addressed using `class_weight="balanced"`.

**Saved artifacts:**

- `models/churn_pipeline.joblib`
- `data/processed/test_data.csv`
- `cleaneddata.csv`

### Day 2 – Model Comparison

Three additional classification algorithms were implemented and evaluated against the Logistic Regression baseline.

**Models implemented:**

1. Decision Tree
2. Random Forest
3. Support Vector Machine (SVM)

Each model used the reusable preprocessing pipeline to ensure consistent data preparation.

**Evaluation metrics:**

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Model Comparison

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0.7381 |    0.5043 | 0.7834 |   0.6136 |  0.8413 |
| SVM                 |   0.7438 |    0.5115 | 0.7727 |   0.6155 |  0.8211 |
| Random Forest       |   0.7672 |    0.5542 | 0.6283 |   0.5889 |  0.8212 |
| Decision Tree       |   0.7402 |    0.5112 | 0.4893 |   0.5000 |  0.6603 |

### Champion Model

**Best Model: Logistic Regression**

Logistic Regression was retained as the champion model because it achieved the highest recall for the churn class at **78.34%**.

Although Random Forest achieved the highest accuracy at **76.72%**, its churn recall was lower at **62.83%**, meaning it missed more actual churn customers.

SVM performed competitively with a recall of **77.27%** and the highest F1 score, but Logistic Regression remained the champion because it achieved the highest churn recall and ROC-AUC among the compared models.

**Saved artifacts:**

- `models/decision_tree.joblib`
- `models/random_forest.joblib`
- `models/svm.joblib`

## Day 3 - Hyperparameter Tuning & Cross-Validation

### Objective

Tune the Day 2 candidate models using GridSearchCV and cross-validation, with **Recall** as the primary metric for churn detection.

### Techniques Used

- GridSearchCV
- Cross-Validation
- Hyperparameter Tuning
- Bias-Variance Analysis
- Learning Curves

### Cross-Validation Results

| Model         | Best CV Recall |
| ------------- | -------------: |
| Decision Tree |         0.7813 |
| Random Forest |         0.8067 |
| **SVM**       |     **0.8254** |

### Champion Model

**Support Vector Machine (SVM)** was selected as the Day 3 champion.

- Best CV Recall: **82.54%**
- Mean Training Recall: **82.89%**
- Train-Validation Gap: **0.35%**
- SVM achieved the highest CV Recall and the smallest train-validation gap.

### Learning Curve

A learning curve was used to compare training and validation recall and analyze model generalization.

### Saved Model

The tuned Day 3 champion was saved as:

`models/churn_v3.joblib`

## Day 4 - Feature Engineering & Ensemble Models

### Models Evaluated

- Day 3 Tuned SVM
- Random Forest
- XGBoost

Feature engineering and feature selection were applied before evaluating the Day 4 ensemble models.

### Outcome

The Day 4 models were compared against the tuned SVM from Day 3 using churn Recall as the primary metric.

**SVM remained the champion model**, as the Day 4 ensemble models did not provide sufficient improvement over the existing tuned model.

Therefore, the existing SVM was retained as the production churn model.

## Evaluation Metrics

The project uses multiple evaluation metrics to assess model performance.

| Metric    | Purpose                                                           |
| --------- | ----------------------------------------------------------------- |
| Accuracy  | Measures overall prediction correctness                           |
| Precision | Measures how many predicted churners actually churn               |
| Recall    | Measures how many actual churners are identified                  |
| F1-score  | Balances precision and recall                                     |
| ROC-AUC   | Measures the model's ability to distinguish between churn classes |

Recall is particularly important because failing to identify a customer who is likely to churn can result in a missed retention opportunity.

---

## Day 5 - Artificial Neural Network (ANN)

### Objective

Build a Keras-based Artificial Neural Network for churn prediction and compare it with the existing classical machine learning champion.

### Techniques Used

- TensorFlow / Keras
- Artificial Neural Network
- Dense Layers
- ReLU and Sigmoid Activations
- Input Normalization
- Early Stopping
- Validation Monitoring

### Model Comparison

| Model                        |     Recall | Precision |   F1 Score | Accuracy | ROC-AUC |
| ---------------------------- | ---------: | --------: | ---------: | -------: | ------: |
| Day 5 Keras ANN              |     57.49% |    61.96% |     59.64% |   79.35% |  83.78% |
| **Day 4 Classical Champion** | **62.57%** |    60.31% | **61.42%** |   79.13% |  83.38% |

### Conclusion

The ANN achieved slightly higher Accuracy, Precision, and ROC-AUC. However, the Day 4 classical champion achieved better **Recall (62.57%)** and **F1 Score (61.42%)**.

Since RetailMind prioritizes identifying as many at-risk churn customers as possible, **Recall remains the primary evaluation metric**.

Therefore, the **Day 4 Classical Champion remains the final churn model**, while the ANN serves as a useful deep-learning benchmark.

## Project Structure

```text
RetailMind/
│
├── data/
│   ├──Telco-Customer-Churn.csv
│   ├──test_data.csv
│   ├──train_data.csv
│
├── models/
│   ├── churn_pipeline.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── svm.joblib
│   ├── day2_champion.joblib
│
├── notebooks/
│   ├── day1_baseline_model.ipynb
│   ├── day2_model_Comparision.ipynb
│
├── src/
│   └── pipeline.py
│
├── README.md
└── requirements.txt
```

_Note: The structure represents the intended project organisation. Adjust filenames if your actual notebook names differ._

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/aditi-prabha/RetailMind.git
cd RetailMind
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks directory and execute the notebooks in the appropriate order.

---

## Future Improvements

- Further hyperparameter optimisation.
- Experiment with SMOTE and other class-imbalance techniques.
- Perform systematic feature selection inside cross-validation pipelines.
- Explore probability calibration and decision-threshold optimisation.
- Evaluate model explainability using SHAP.
- Build an interactive churn prediction dashboard.
- Develop a REST API using FastAPI.
- Deploy the final model for real-time customer churn predictions.
- Monitor model drift and prediction performance.

---

## Important Considerations

The current model is an experimental machine learning solution built using a public telecom churn dataset.

The test set has been used for evaluation across multiple development stages. Therefore, the Day 4 comparison should be treated as an experimental benchmark rather than an independent estimate of final production performance.

A future production version should use training-only cross-validation for model selection and a separate untouched holdout dataset for final evaluation.

## Author

**Aditi Prabha**
GitHub: [aditi-prabha](https://github.com/aditi-prabha)
Project Repository: [RetailMind](https://github.com/aditi-prabha/RetailMind)

---

_RetailMind AI — Turning customer data into actionable churn insights._
