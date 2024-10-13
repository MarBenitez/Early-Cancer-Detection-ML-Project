# Early Cancer Detection through Multianalyte Blood Test and Machine Learning Models

## Project Overview
This project aims to improve early cancer detection through multianalyte blood analysis and machine learning models. We replicated and enhanced the findings of the study "Early Cancer Detection from Multianalyte Blood Test Results" to predict both the presence and type of cancer. A variety of supervised models were used, including Naive Bayes, decision trees, and deep learning, with a focus on the CancerA1DE model.

For detailed information, please refer to the project report provided in PDF format.

## Repository Structure
- `/binary-detection`
  - **app**: Contains the application files for binary cancer detection.
    - `CancerDetector.py`: Main script for cancer detection, allowing input of blood test parameters and selecting a model to predict cancer presence.
    - `Constantes.py`, `Functions.py`, `Models.py`: Supporting modules for constants, utility functions, and machine learning models used by `CancerDetector.py`.
    - `info_icon.png`: Icon used in the application UI.
  - **results**: Stores the results of the binary detection models.
    - `model_results_final.xlsx`, `model_results_NaiveBayes.xlsx`, `model_results.xlsx`: Excel files containing model evaluation metrics and results.
  - `cancer-binary-detection.ipynb`: Notebook for binary cancer detection (cancer vs. no cancer).

- `/classification-multiclass`
  - **app**: Contains application files for multiclass classification of cancer types.
  - **results**: Stores results for the multiclass classification models.
    - `model_proba_results.xlsx`, `model_results.xlsx`, `Results_No_supervision.xlsx`: Excel files with metrics and results for the classification models.
  - `cancer-classification-multiclass.ipynb`: Notebook for cancer type classification.

- `/data`
  - `Tables_S1_to_S11.xlsx`: Contains the original datasets used for training and evaluating models.

- `/reports`
  - `Memory.pdf`: Detailed document on methodology, results, and analysis.
  - `Presentation.pptx`: Presentation of the project's findings.

- `.gitattributes`, `.gitignore`, `LICENSE`: Configuration and licensing files.
- `README.md`: Main file with project description.

## Project Objectives
1. **Replication of Existing Models**: Validate the effectiveness of early cancer detection models developed in previous studies.
2. **Development of New Models**: Extend existing models to classify the specific type of cancer into eight predefined categories.
3. **Improvement of Precision and Sensitivity**: Increase the sensitivity in detecting stage I cancer while maintaining high specificity.

## Data Utilized
- **Binary Detection Dataset**: Contains blood test records from 1,817 patients for cancer/no cancer prediction.
- **Classification Dataset**: Includes data for 626 patients with biomarkers to categorize specific cancer types.

For more details on the datasets, please refer to the PDF project report.

## Methodology
1. **Exploratory Data Analysis (EDA)**: A thorough analysis of the data to understand biomarker distribution and importance.
2. **Model Training**: Multiple supervised models were used to predict the presence and classify the type of cancer.
3. **Evaluation**: Models were evaluated using metrics like sensitivity, specificity, and AUC-ROC.

## Key Results
- The CancerA1DE model doubled the sensitivity for stage I cancer detection while maintaining 99% specificity.
- Extended models for cancer type classification showed robust performance across various categories.

## Installation and Usage
1. Clone the repository:
   ```
   git clone https://github.com/MarBenitez/Early-Cancer-Detection-ML-Project
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Navigate to the respective directory (`/binary-detection` or `/classification-multiclass`) and run the application or open the notebooks in Jupyter to explore the analysis and results.

## Contributions
Contributions are welcome. Please open an issue or make a pull request for suggestions or improvements.

## Contact
- Authors: Florina Cretu, Mar Benitez de Lucio Villegas, Kevin Jhoan Orozco Agudelo, Daniel del Río Alonso.
- Tutor: Antonio Pita Lozano.

For any inquiries, you can contact [Mar Benitez](mar27benitez@gmail.com).

## License
This project is licensed under the MIT License. See the `LICENSE` file for more information.

---

### Detailed Project Description

#### Introduction
Early cancer detection significantly increases the chances of successful treatment and patient survival. This project aims to replicate and improve on the findings of the study "Early Cancer Detection from Multianalyte Blood Test Results" by using multianalyte blood analysis to predict both the presence and type of cancer. The study utilizes machine learning models to analyze blood markers, thereby increasing the precision in cancer detection and classification.

#### Motivation
The motivation behind this project stems from addressing the limitations identified in previous studies and exploring new techniques to enhance sensitivity and specificity in early cancer detection. Traditional methods for cancer detection, such as imaging and invasive biopsies, can be costly and inaccessible. Recent advancements in liquid biopsy technologies, such as the Galleri test developed by GRAIL, have shown promise in enabling non-invasive cancer detection through blood analysis.

Our key contributions are:
- **Replication and Validation of Existing Models**: Confirm the effectiveness of previous early detection methods.
- **Development of New Models**: Create and test new machine learning models that could outperform the existing ones in terms of accuracy and efficiency.
- **Practical Application**: Provide practical tools and methodologies for early cancer detection in clinical settings.

#### Data Description
We used two main datasets:
- **Binary Detection Dataset**: This dataset contains blood test records of 1,817 patients for binary cancer detection. It includes eight blood protein markers and an additional DNA mutation score called OmegaScore.
- **Classification Dataset**: This dataset contains records of 626 patients and includes data on 39 blood protein markers (the original eight plus 31 additional ones), the OmegaScore, and the patient's gender.

#### Data Analysis
We performed an extensive descriptive and exploratory analysis to understand the distribution and characteristics of the data:
- **Data Cleaning**: The datasets were relatively clean, but we handled missing values, particularly for variables like `AJCC stage`, which represents cancer severity.
- **Correlation Analysis**: Analyzed correlations between variables to identify relationships and reduce redundancy. Two approaches were used: Pearson Correlation and Decision Trees to assess variable importance.

#### Feature Engineering
- **Target Binarization**: The `Tumor Type` variable was binarized to simplify initial analysis (0 for no cancer, 1 for presence of cancer).
- **Scaling and Encoding**: Numerical variables were standardized using `StandardScaler`, while categorical variables were transformed using `One Hot Encoding`.

#### Model Building
We used several machine learning models, including:
- **Supervised Models**: Logistic Regression, Decision Trees, Random Forest, KNN, SVM, Naive Bayes, and Neural Networks.
- **Unsupervised Models**: K-Means, DBSCAN, ICA, among others, were tested but found to be less effective for our classification needs.

We also tested ensemble methods, like **Voting Classifier**, combining models like **XGBoost**, **LightGBM**, and **Gradient Boosting**, which showed improvements in precision and robustness.

#### Model Evaluation
- **Evaluation Metrics**: Sensitivity, specificity, precision, recall, and F1-score were used to evaluate the performance of models.
- **Overfitting Handling**: Techniques such as Lasso and Ridge regularization, increased cross-validation (10-fold), and early stopping were used to reduce overfitting.
- **Results**: The extended models achieved high accuracy with reduced overfitting. The Voting Classifier outperformed other individual models, particularly in the classification of cancer types.

#### Practical Implementation
A web application (`CancerDetector.py`) was created using Flask to provide an interface for predicting cancer presence or type based on blood test results. The application leverages the trained models to provide users with predictions, making it a valuable tool for clinical use.

The project also utilizes Docker for easier deployment and **MLFlow** for experiment tracking.

## Project Achievements
1. **Development of Supervised Models**: Implemented various machine learning models, including Logistic Regression, Random Forest, KNN, AdaBoost, Gradient Boosting, and Voting Classifier. Models like Random Forest and Gradient Boosting showed the best results in terms of accuracy and predictive capacity.
2. **Cross-Validation**: Applied cross-validation to ensure models generalized well to new data. Metrics such as precision, recall, and F1-score were used for evaluation.
3. **Development of the Application**: Created the `CancerDetector.py` application to allow users to input blood test parameters and select a model to predict cancer presence. It also provides an option to predict cancer type using an ensemble of models.
4. **Analysis of Unsupervised Models**: Although unsupervised learning techniques like KMeans, DBSCAN, and GMM were explored, they were found unsuitable for the type of data and objectives of this project.

## Critical Analysis
- **Data Quality and Quantity**: The dataset quality is high, but the small sample size limits model generalization and may lead to overfitting. CTGAN was used to generate synthetic data, but this approach can introduce biases and may not replicate complex relationships.
- **Supervised Models**: Supervised models like Random Forest and Gradient Boosting were effective for cancer prediction, but variability in predictions suggests there is room for improvement in hyperparameter tuning and data expansion.
- **Unsupervised Models**: Unsupervised models did not provide significant improvements due to the specific nature of the data and the need for accurate classification.

## Technological Architecture
The project was designed with the following technological pipeline:
1. **Data Acquisition and Preprocessing**: Collection and cleaning of blood test datasets, handling missing values, and ensuring consistency.
2. **Exploratory Data Analysis (EDA)**: Detailed descriptive and exploratory analysis of blood markers, including distribution visualization, skewness analysis, and normality testing.
3. **Modeling and Evaluation**: Training both supervised and unsupervised models, evaluating performance metrics, and ensuring generalizability.
4. **Infrastructure and Development Environment**: Leveraged Docker for model deployment, Flask for creating a user interface, and MLFlow for tracking experiments.

## Project Achievements and Contributions
1. **Replication and Validation of Existing Models**: Successfully replicated and validated previous models, confirming their effectiveness in cancer detection.
2. **Development of New Models**: Developed new supervised models that improved the accuracy of cancer prediction. Ensembles like Voting Classifier provided robustness in predictions.
3. **Practical Application**: Created the `CancerDetector.py` application that provides a user-friendly interface for clinicians to input blood test results and get predictions for cancer presence and type.
4. **Detailed Data Analysis**: Conducted an in-depth analysis of the datasets used, including handling of missing values, correlation analysis, and data visualization.

## Future Work
- **Dataset Expansion**: Collaborate with medical institutions to gather more diverse and authentic patient data. Augmenting the dataset with different cohorts and conditions will help improve model robustness and generalization.
- **Synthetic Data as a Complement**: Continue using models like CTGAN to supplement real data, but focus on validating these synthetic datasets to avoid bias and maintain integrity.
- **Model Optimization**: Further optimize hyperparameters using Grid Search or Random Search, explore deeper neural networks, and consider adding more models to the Voting Classifier to improve accuracy.
- **Practical Application**: Enhance the `CancerDetector.py` application by adding functionalities for visualizing biomarker trends and integrating it with clinical systems for real-world deployment.

## Conclusion
This project underscores the potential of combining machine learning with biomedical data to advance cancer diagnosis and improve patient outcomes. While the current models achieve promising results, further data collection, optimization, and validation are crucial for real-world clinical application.
