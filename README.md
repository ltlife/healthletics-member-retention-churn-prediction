# Healthletics: AI-Powered Member Retention and Churn Prediction System

## Project Overview

This project develops an AI-powered proof-of-concept system for predicting gym member churn and supporting member retention decisions at Healthletics Fitness Club.

The project applies machine learning to membership and activity-related data to identify patterns associated with member churn and provide actionable insights that can support proactive retention strategies.

This project was developed as part of the AI & Machine Learning Capstone Project.

---

## Business Problem

Gym member retention is an important business challenge. When members discontinue their memberships, the gym loses recurring revenue and opportunities to maintain long-term member relationships.

The objective of this project is to identify members who are at higher risk of churn so that retention efforts can be prioritized proactively.

The system is intended as a decision-support tool and not as the sole basis for individual member decisions.

---

## Data Science Problem

The project is formulated as a **binary classification problem**:

- `0` = Retained
- `1` = Churned

The model predicts whether a gym member is likely to churn based on membership characteristics, demographic information, and activity-related features.

---

## Dataset

The project uses the public **Gym Customers Features and Churn** dataset from Kaggle.

Dataset characteristics:

- 4,000 member records
- 14 original variables
- No missing values
- No duplicate records
- Churned members: 1,061 (26.53%)
- Retained members: 2,939 (73.48%)

The dataset includes variables related to:

- Membership contract period
- Member age
- Lifetime
- Class attendance frequency
- Current-month activity
- Group visit participation
- Partner and promotional membership indicators
- Additional charges
- Months remaining on contract

The dataset is used as a proof-of-concept and does not represent actual Healthletics member data.

---

## Methodology

The project follows an end-to-end machine learning workflow:

1. Problem understanding and business framing
2. Dataset collection and understanding
3. Data preprocessing
4. Exploratory data analysis
5. Feature engineering
6. Feature selection
7. Standardization and PCA analysis
8. Machine learning model implementation
9. Model evaluation and comparison
10. Explainability and feature importance analysis
11. Bias and fairness assessment
12. Business interpretation and recommendations

---

## Feature Engineering

A key engineered feature was:

`frequency_change`

This measures the change between a member's current-month class frequency and their historical average class frequency.

This feature showed a strong relationship with churn and became the most important predictor in the Random Forest model.

---

## Machine Learning Models

Three classification models were implemented and compared:

- Logistic Regression
- Decision Tree
- Random Forest

Random Forest was selected as the final candidate model because it achieved the strongest overall performance across the main evaluation metrics.

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9262 | 0.8844 | 0.8302 | 0.8564 | 0.9775 |
| Decision Tree | 0.9462 | 0.9519 | 0.8396 | 0.8922 | 0.9701 |
| Random Forest | **0.9650** | **0.9600** | **0.9057** | **0.9320** | **0.9879** |

The Random Forest model achieved:

- Accuracy: **96.50%**
- Precision: **96.00%**
- Recall: **90.57%**
- F1 Score: **93.20%**
- ROC-AUC: **98.79%**

Five-fold cross-validation produced a mean ROC-AUC of approximately **0.988 ± 0.0035**.

---

## Key Findings

The analysis identified several important patterns associated with churn.

### Contract Period

Shorter contracts were associated with substantially higher churn:

- 1-month contracts: **42.3% churn**
- 6-month contracts: **12.5% churn**
- 12-month contracts: **2.4% churn**

### Member Activity

Members who reduced their class attendance were more likely to churn.

The engineered `frequency_change` feature had the strongest relationship with churn among the evaluated features.

### Member Lifetime

Longer member lifetime was associated with lower churn risk.

These findings suggest that declining engagement and shorter membership commitments may be useful signals for proactive retention strategies.

---

## Model Explainability

Model interpretation was supported using:

- Random Forest feature importance
- Permutation importance
- Partial Dependence Plot (PDP) analysis

The most important Random Forest features included:

1. `frequency_change`
2. `Lifetime`
3. `Avg_class_frequency_current_month`
4. `Age`
5. `Avg_class_frequency_total`
6. `Month_to_end_contract`

These results provide an interpretable basis for understanding the factors associated with predicted churn.

---

## Ethical AI and Bias Assessment

The project considers potential issues related to:

- Class imbalance
- Dataset limitations
- Feature leakage
- Demographic differences
- Small subgroup sample sizes
- Potential proxy effects of demographic and behavioral variables

Gender churn rates were nearly identical in the dataset.

Model performance was also examined across age groups. Differences in model performance were observed, particularly in recall for some groups. Small subgroup sizes, especially for the 35–41 age group in the test set, require cautious interpretation.

The model should therefore be used as a **decision-support tool**, with human oversight and appropriate business context.

Churn predictions should be used to support engagement and retention efforts rather than to exclude, penalize, or disadvantage members.

---

## Business Value

The proposed system can support Healthletics by helping the business:

- Identify members who may require proactive engagement
- Prioritize retention efforts
- Understand behavioral patterns associated with churn
- Improve member engagement strategies
- Support data-driven membership decisions
- Provide a foundation for future AI-powered gym management capabilities

The current project is a proof-of-concept based on a public dataset. Before operational deployment, the model should be retrained and validated using appropriately collected, anonymized, and representative Healthletics data.

---

## Repository Structure

```text
healthletics-member-retention-churn-prediction/
│
├── data/
│   └── Dataset files
│
├── models/
│   └── Saved machine learning model
│
├── notebooks/
│   ├── Healthletics_Capstone.ipynb
│   └── README.md
│
├── presentations/
│   ├── Healthletics_Technical_Capstone_FINAL.pptx
│   ├── Healthletics_Business_Facing_Capstone_Presentation.pptx
│   └── README.md
│
├── reports/
│   ├── Healthletics_AI_Member_Retention_Churn_Prediction_Capstone_Report.pdf
│   └── README.md
│
├── src/
│   └── Random Forest training script
│
├── README.md
└── requirements.txt
