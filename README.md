## Breast Cancer Diagnosis Prediction
# Programming for Data Analytics Project – Rennes School of Business

## About the Project
This project addresses the critical challenge of early and accurate breast cancer diagnosis using advanced data analytics. Breast cancer is one of the leading causes of cancer-related deaths worldwide, where early identification of malignant versus benign tumors greatly improves treatment outcomes and survival rates. Leveraging a publicly available dataset with detailed cellular features extracted from breast tumor images, our objective is to explore, analyze, and model these features to predict tumor types.
The project involves data cleaning to address incompleteness and inconsistencies, exploratory data analysis to uncover patterns and relationships, and supervised machine learning techniques for tumor classification. Our study aims not only to develop a reliable prediction model but also to provide medical practitioners with interpretable indicators for diagnostic support. Through this work, we contribute to efforts that combine data-driven insight with health care advancement.

## Team Members
```
Member 1 - Mohammad Yasir Siddiqui
Member 2 – Rohaid Bhatti
Member 3 – Muhammad Faizan
Member 4 – Adhithiya Gopalkrishnan
Member 5 – Jaseem Bhatti
```
## Dataset
The dataset comprises 569 patient records, each with 32 feature columns describing quantitative measurements of cell nuclei characteristics such as radius, texture, perimeter, area, smoothness, and symmetry. The target variable denotes tumor diagnosis: 'B' for benign and 'M' for malignant. It is sourced from the UCI Machine Learning Repository, with preprocessing steps including missing value handling and normalization applied before analysis.

## Repository Structure
```
BreastCancerDiagnosis/
│
├── data/
│   └── data.csv
│
├── notebooks/
│   └── breast_cancer.ipynb
│
├── images/
│   ├── Screenshot-2025-11-04-002700.jpg
│   ├── Screenshot-2025-11-04-002652.jpg
│   ├── WhatsApp-Image-2025-11-04-at-00.26.01_398f1773.jpg
│   ├── Screenshot-2025-11-04-002707.jpg
│   └── WhatsApp-Image-2025-11-04-at-00.25.55_aafcaacd.jpg
│
├── models/
│   └── bestbreastcancermodel.joblib
│
└── README.md
```
## Methodology
1. Data Cleaning
We started by imputing missing values and removing irrelevant columns to refine the dataset. This ensured reliable input for modeling by mitigating noise and potential bias.

2. Exploratory Data Analysis (EDA)
Using histograms and correlation heatmaps, we investigated the distributions and relationships of features. This step identified patterns suggesting which variables are most critical for tumor prediction.

3. Predictive Modeling
We developed a Logistic Regression model to classify tumors as benign or malignant. Model training included splitting data into training and testing sets, followed by evaluation using metrics such as accuracy, precision, recall, and ROC-AUC.

4. Interpretation and Validation
Performance metrics and visualizations, including confusion matrices and ROC curves, were employed to validate model robustness and interpret its predictive power.

## Results and Visualizations

# Feature Distributions
![Feature Distributions](images/Screenshot-2025-11

Histograms visualize the distribution of selected numeric features like 'radius_mean', 'texture_mean', and 'area_mean'. Distinct separations in these distributions between benign and malignant cases indicate their predictive potential.

# Correlation Heatmap
![Correlation Heatmap](images/Screenshot-2025-11

This heatmap reveals strong positive and negative correlations among features, guiding variable selection and dimensionality reduction strategies. Highly correlated features such as radius_mean and perimeter_mean confirm biological relevance.

# Confusion Matrix
![Confusion Matrix]("images/Screenshot 2025-11-04 002700.jpg")

The confusion matrix summarizes our model’s classification outcomes. It shows very few misclassifications, with 71 true negatives and 39 true positives, indicating high reliability in distinguishing tumor types.

# ROC Curve
![ROC Curve](images/WhatsApp-Image-2025-11-04-at-00.26.01_398f177 analysis demonstrates an impressive Area Under the Curve (AUC) of 0.996, reflecting excellent sensitivity and specificity across classification thresholds.

# Classification Report
![Classification Report](images/WhatsApp-Image-2025-11-04-at-00.25.55_aafca classification report details precision, recall, and F1-score metrics for benign and malignant categories, all exceeding 0.95, confirming model robustness.

## How to Run
1. Clone the repository.
2. Install dependencies:
   -- pip install pandas numpy matplotlib seaborn scikit-learn
3. Launch the Jupyter notebook and run all cells.

## Conclusions
The project evaluated multiple machine learning models—including Logistic Regression, Random Forest, and Gradient Boosting—to classify breast cancer tumors using cellular feature datasets. While Random Forest and Gradient Boosting showed strong performance, the Logistic Regression model was ultimately chosen for its excellent balance between accuracy, interpretability, and computational efficiency. The Logistic Regression model achieved a classification accuracy of 96.5%, an area under the ROC curve (AUC) of 0.996, and over 95% precision and recall for both benign and malignant classes. This combination of high predictive power and transparency makes it a valuable tool for clinical decision support. The ensemble models, Random Forest and Gradient Boosting, also demonstrated robust classification with comparable accuracy levels, confirming the reliability of tree-based methods for this task. Together, these results validate that machine learning can significantly enhance breast cancer diagnosis, offering accurate, interpretable, and scalable solutions.

## Tools and Technologies
Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
Jupyter Notebook
GitHub

## License
This project is distributed under the MIT License. Feel free to use and adapt the code for academic or research purposes.

## Contact
For any questions, feedback, or collaboration opportunities, please contact at (mohammad-yasir-siddiqui.x@rennes-sb.com).




