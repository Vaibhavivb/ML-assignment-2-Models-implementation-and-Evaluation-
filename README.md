# ML-assignment-2-Models-implementation-and-Evaluation

a) Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict the presence of heart disease in patients based on medical attributes. The goal is to compare different algorithms using standard evaluation metrics and determine which model performs best for this dataset.

b) Dataset Description

The dataset used is the UCI Heart Disease Dataset "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
, which combines data from four different sources: Cleveland,Hungary,Switzerland,VA Long Beach
Each record represents a patient and includes medical attributes such as:
| Feature  | Description                  |
| -------- | ---------------------------- |
| age      | Age of patient               |
| sex      | Gender                       |
| cp       | Chest pain type              |
| trestbps | Resting blood pressure       |
| chol     | Cholesterol level            |
| fbs      | Fasting blood sugar          |
| restecg  | Resting ECG results          |
| thalach  | Maximum heart rate           |
| exang    | Exercise induced angina      |
| oldpeak  | ST depression                |
| slope    | Slope of ST segment          |
| ca       | Number of major vessels      |
| thal     | Thalassemia                  |
| target   | Heart disease presence (0/1) |

Preprocessing Steps
Combined 4 datasets
Replaced missing values (?) with NaN
Converted all columns to numeric
Filled missing values using median
Converted target to binary classification:
0 → No disease
1 → Disease

c) Models Used and Evaluation

The following six classification models were implemented:
1) Logistic Regression
2) Decision Tree
3) k-Nearest Neighbors (kNN)
4) Naive Bayes
5) Random Forest (Ensemble)
6) XGBoost (Ensemble)
   
 Comparison Table of Model Performance
| ML Model            | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
| ------------------- | -------- | ----- | --------- | ------ | ----- | ----- |
| Logistic Regression | 0.826    | 0.894 | 0.843     | 0.843  | 0.843 | 0.648 |
| Decision Tree       | 0.766    | 0.756 | 0.757     | 0.853  | 0.802 | 0.525 |
| kNN                 | 0.842    | 0.868 | 0.841     | 0.882  | 0.861 | 0.680 |
| Naive Bayes         | 0.826    | 0.887 | 0.837     | 0.853  | 0.845 | 0.647 |
| Random Forest       | 0.832    | 0.921 | 0.838     | 0.863  | 0.850 | 0.658 |
| XGBoost             | 0.804    | 0.879 | 0.800     | 0.863  | 0.830 | 0.603 |


D) Observations on Model Performance
| Model               | Observation                                                                                    |
| ------------------- | ---------------------------------------------------------------------------------------------- |
| Logistic Regression | Performs consistently well with balanced precision and recall. Good baseline model.            |
| Decision Tree       | Lowest accuracy and MCC; tends to overfit and is less stable.                                  |
| kNN                 | Highest accuracy and F1 score; performs very well on this dataset.                             |
| Naive Bayes         | Stable performance despite strong independence assumptions.                                    |
| Random Forest       | Highest AUC score; strong generalization due to ensemble learning.                             |
| XGBoost             | Good performance but slightly lower than Random Forest and kNN; sensitive to parameter tuning. |

Final Conclusion
Best Overall Accuracy: kNN
Best AUC Score: Random Forest
Most Balanced Model: Logistic Regression
Most Stable Ensemble: Random Forest
Overall, ensemble methods (Random Forest and XGBoost) and distance-based methods (kNN) performed better than single-tree models, demonstrating that combining multiple decision rules improves prediction performance.
