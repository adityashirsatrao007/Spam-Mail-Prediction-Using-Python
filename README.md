```markdown
# Spam Mail Prediction Using Python

A machine learning project to classify emails as **spam** or **ham** using Python, scikit-learn, and TF-IDF vectorization.

## 📌 Overview

This project demonstrates how to build and train a **Logistic Regression** model to detect spam emails.  
It covers data preprocessing, feature extraction, model training, evaluation, and live predictions.

## ✨ Features

- Preprocessing of email text data  
- TF-IDF vectorization for feature extraction  
- Logistic Regression for classification  
- Evaluation with accuracy metrics  
- Predictive system for new input emails  

## 📂 Project Structure

```

├── spam\_mail\_Predection.ipynb   # Jupyter Notebook with code and workflow
├── mail\_data.csv                # Dataset (spam/ham emails)
└── README.md                    # Project documentation

````

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/adityashirsatrao007/Spam-Mail-Prediction-Using-Python.git
   cd Spam-Mail-Prediction-Using-Python
````

2. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn jupyter
   ```

3. Run the notebook:

   ```bash
   jupyter notebook spam_mail_Predection.ipynb
   ```

## 🚀 Workflow

1. **Load & preprocess dataset**

   * Handle missing values
   * Encode labels (spam → 0, ham → 1)

2. **Split data**

   * Train-test split

3. **Vectorize text**

   * Convert email text into TF-IDF features

4. **Train model**

   * Logistic Regression

5. **Evaluate model**

   * Accuracy on training and test sets

6. **Predict system**

   * Classify custom email input as spam or ham

## 🖥️ Example

```python
input_mail = ["Congratulations! You won a free ticket. Claim now!"]
input_vector = tfidf_vectorizer.transform(input_mail)
prediction = model.predict(input_vector)

if prediction[0] == 1:
    print("Ham Mail (Not Spam)")
else:
    print("Spam Mail")
```

## 📊 Results

* Training accuracy: \~96–97%
* Testing accuracy: \~95–96%

## 🔮 Future Improvements

* Try other classifiers (Naive Bayes, SVM, Random Forest)
* Add metrics like Precision, Recall, F1-score
* Build a web or GUI app for real-time email classification

---
