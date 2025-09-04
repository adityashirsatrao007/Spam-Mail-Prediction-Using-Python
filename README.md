#  Spam Mail Prediction Using Python

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A comprehensive **machine learning project** that classifies emails as **Spam** or **Ham (Not Spam)** using **Python, scikit-learn, and TF-IDF Vectorization** with **Support Vector Machine (SVM)** classifier.

##  Project Objective

The main goal of this project is to build an intelligent email classification system that can automatically identify spam emails using natural language processing and machine learning techniques. This helps in email filtering and security applications.

##  Overview

This project implements a complete machine learning pipeline for email spam detection:
- **Data preprocessing** and cleaning
- **TF-IDF vectorization** for text feature extraction
- **Support Vector Machine (Linear SVC)** for classification
- **Model evaluation** with accuracy metrics
- **Real-time prediction** capabilities

##  Key Features

-  **Data Analysis**: Comprehensive exploratory data analysis of email dataset
-  **Text Preprocessing**: Advanced text cleaning and normalization
-  **Feature Engineering**: TF-IDF vectorization with stop words removal
-  **Machine Learning**: Linear Support Vector Classifier implementation
-  **Model Evaluation**: Training and testing accuracy assessment
-  **Prediction System**: Real-time spam detection for new emails
-  **Documentation**: Well-documented Jupyter notebook with explanations

##  Technologies Used

- **Python 3.7+**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **TfidfVectorizer**: Text feature extraction
- **LinearSVC**: Support Vector Machine classifier
- **Jupyter Notebook**: Interactive development environment

##  Project Structure

`
Spam-Mail-Prediction-Using-Python/

 spam_mail_Predection.ipynb    # Main Jupyter notebook with implementation
 spamham.csv                   # Dataset containing spam/ham emails
 README.md                     # Project documentation
 requirements.txt              # Python dependencies
`

##  Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### 1. Clone the Repository
`ash
git clone https://github.com/adityashirsatrao007/Spam-Mail-Prediction-Using-Python.git
cd Spam-Mail-Prediction-Using-Python
`

### 2. Install Dependencies
`ash
pip install -r requirements.txt
`

Or install manually:
`ash
pip install pandas numpy scikit-learn jupyter matplotlib seaborn
`

### 3. Launch Jupyter Notebook
`ash
jupyter notebook spam_mail_Predection.ipynb
`

##  Project Workflow

### 1. **Data Loading**
- Import the spam/ham email dataset (spamham.csv)
- Load data using pandas with proper encoding (latin-1)

### 2. **Data Preprocessing**
- Handle missing values by replacing with empty strings
- Extract relevant columns (Category, Details)
- Map labels: ham = 1, spam = 0

### 3. **Feature Engineering**
- Apply TF-IDF Vectorization to convert text to numerical features
- Configure parameters:
  - min_df=1: Minimum document frequency
  - stop_words='english': Remove common English stop words
  - lowercase=True: Convert text to lowercase

### 4. **Model Training**
- Split data into training (80%) and testing (20%) sets
- Train Linear Support Vector Classifier (LinearSVC)
- Convert target variables to integer format

### 5. **Model Evaluation**
- Calculate training accuracy
- Calculate testing accuracy
- Compare performance metrics

### 6. **Prediction**
- Use trained model to classify new email texts
- Real-time spam detection capabilities

##  Model Performance

The Linear SVC model demonstrates excellent performance:

- **Training Accuracy**: ~99.5%
- **Testing Accuracy**: ~97.8%
- **Model Type**: Support Vector Machine (Linear)
- **Feature Extraction**: TF-IDF Vectorization

##  Usage Example

`python
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load and preprocess data
raw_data = pd.read_csv('spamham.csv', encoding='latin-1')
mail_data = raw_data[['Category', 'Details']]
mail_data['Category'] = mail_data['Category'].map({"ham": 1, "spam": 0})

# Feature extraction
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X = vectorizer.fit_transform(mail_data['Details'])
y = mail_data['Category']

# Train model
model = LinearSVC()
model.fit(X, y)

# Predict new email
new_email = ["Congratulations! You've won . Click here to claim now!"]
email_vector = vectorizer.transform(new_email)
prediction = model.predict(email_vector)

if prediction[0] == 1:
    print("Ham Mail (Not Spam)")
else:
    print("Spam Mail")
`

**Output:**
`
Spam Mail
`

##  Dataset Information

- **Source**: Email spam classification dataset
- **Format**: CSV file with encoding 'latin-1'
- **Columns**:
  - Category: Email classification (ham/spam)
  - Details: Email content/text
- **Size**: Multiple thousand email samples
- **Distribution**: Balanced dataset with both spam and ham emails

##  Future Enhancements

### Algorithm Improvements
- [ ] Implement Naive Bayes classifier
- [ ] Try Random Forest for ensemble learning
- [ ] Experiment with deep learning models (LSTM, BERT)
- [ ] Add cross-validation for better model evaluation

### Feature Engineering
- [ ] N-gram analysis (bigrams, trigrams)
- [ ] Word embeddings (Word2Vec, GloVe)
- [ ] Advanced text preprocessing (stemming, lemmatization)
- [ ] Feature selection techniques

### Model Evaluation
- [ ] Precision, Recall, F1-Score metrics
- [ ] ROC-AUC curve analysis
- [ ] Confusion matrix visualization
- [ ] Learning curves

### Deployment
- [ ] Create Flask web application
- [ ] Build Streamlit dashboard
- [ ] REST API development
- [ ] Docker containerization
- [ ] Cloud deployment (AWS, GCP, Azure)

##  Security Considerations

- Email content is processed locally
- No sensitive data is stored permanently
- Model predictions are based on text patterns only
- Regular model updates recommended for new spam patterns

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Author

**Aditya Shirsat Rao**
- GitHub: [@adityashirsatrao007](https://github.com/adityashirsatrao007)
- LinkedIn: [Connect with me](https://linkedin.com/in/adityashirsatrao007)

##  Acknowledgments

- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [pandas](https://pandas.pydata.org/) - Data manipulation tool
- [numpy](https://numpy.org/) - Numerical computing library
- [Jupyter](https://jupyter.org/) - Interactive computing environment
- Email spam dataset contributors
- Open source community

##  References

1. Vapnik, V. (1995). The Nature of Statistical Learning Theory
2. Joachims, T. (1998). Text categorization with support vector machines
3. Manning, C. D., & Schütze, H. (1999). Foundations of statistical natural language processing

---

 **If you found this project helpful, please give it a star!** 

---

*Last updated: September 4, 2025*
