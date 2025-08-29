# Banking_Churn_Prediction
Using Machine Learning and XGBoost to come up with practical use cases for churn predictions where class imbalances are common

# Banking Customer Churn Prediction

This project predicts whether a bank customer is likely to churn (leave the bank) based on their demographic and account information. It uses **machine learning** techniques to analyze patterns and identify at-risk customers. Furthermore, this project also dives into figuring out which type of model would work with which type of business use case.

## Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Feature engineering and selection
- Model training and evaluation using:
  - ANN
  - XGBoost
  - Threshold tuning
- Performance metrics: Accuracy, Precision, Recall, F1-score

## Tech Stack
- **Python 3**
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Summary
- This project addresses the critical issue of customer churn in retail banking, where losing customers significantly impacts profitability. Using a 10,000-customer dataset (20.4% churn rate), the research investigates how model complexity and threshold tuning affect both predictive performance and financial outcomes.

## Models Tested
- Simple ANN (2 hidden layers)
- Deep ANN (3 hidden layers)
- XGBoost (tree-based boosting model)

Each model was evaluated under three configurations:
1. Standard threshold (0.5) – baseline
2. Lowered threshold (0.3) – higher sensitivity
3. Lowered threshold + class weighting – aggressive recall strategy

## Key Findings
- Threshold tuning is more impactful than increasing model complexity for aligning with business goals.
- Lower thresholds significantly increased recall (catching more churners) but decreased precision.
- Weighted + low threshold configurations achieved very high recall (up to 88%), albeit with more false positives.
- Accuracy alone is misleading due to class imbalance; business-focused metrics like precision, recall, F1-score, and profit impact are essential.

## Profitability Analysis
- Without a model, the bank incurs a loss of €101,750 from missed churners.
- XGBoost with low threshold + class weights generated the highest profit (€37,950) by minimizing costly false negatives.
- Precision-focused models (threshold 0.5) resulted in a net loss (€8,925) due to missed churners, despite fewer false positives.
- False negatives cost 10 times more than false positives, so recall optimization is generally best when interventions are inexpensive.

## Conclusion
- Simple models with threshold tuning can achieve similar or better business results than complex models.
- Banks should select thresholds based on cost structures and business strategy (e.g., prioritize recall when retaining customers is crucial).
- Model interpretability, dataset generalization, and realistic cost assumptions remain areas for future research.

