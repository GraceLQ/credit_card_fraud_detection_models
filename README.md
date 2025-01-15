# Credit Card Fraud Detection Using Decision Tree and Support Vector Machine 

This project focuses on building a machine learning model to detect credit card fraud using a dataset containing transactions made by European cardholders in September 2013. The dataset includes features generated through Principal Component Analysis (PCA), which anonymizes sensitive transaction details while preserving essential patterns. 

By leveraging techniques like SMOTE for oversampling, StandardScaler, Normalizer, rigorous evaluation metrics such as AUPRC and ROC AUC, and experimenting with multiple models, this project demonstrates the application of data science methodologies to tackle real-world challenges.

Despite the inherent difficulties of class imbalance and overlapping feature distributions, the project achieves significant improvements over baseline metrics and provides actionable insights for future enhancements. The SVM with class weight adjustment achieved the best results, with an ROC AUC of 0.988 and an AUPRC of 0.769, showing its ability to handle the class imbalance effectively and provide strong fraud detection performance.

## Key Techniques
- Decision Tree and SVM Models: Implemented and tuned Decision Tree and Support Vector Machine (SVM) models to evaluate their ability to handle the imbalanced dataset.
- StandardScaler and Normalizer: Applied to standardize features by scaling them to have zero mean and unit variance (StandardScaler) and to normalize feature vectors to unit length (Normalizer).
- Synthetic Minority Oversampling Technique (SMOTE): Used to balance the dataset by generating synthetic samples for the minority class (fraud cases). Class weighting was also tested to improve the model.
- Precision-Recall and ROC AUC Evaluation: Focused on metrics tailored for imbalanced datasets to evaluate the model's performance effectively.

## Key Obstacles and Improvement
- **Imbalanced data:** Fraudulent transactions typically account for a small percentage of all transactions, leading to class imbalance that biases models towards the majority non-fraud class.
- **Overlapping features:** Fraudulent transactions often share characteristics with legitimate ones, causing high false positive and false negative cases. Improved feature engineering is suggested to specific features, sucha as device fingerprints, transaction locations, or historical behaviors.
