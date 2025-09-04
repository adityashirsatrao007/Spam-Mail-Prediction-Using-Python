```markdown
# 📧 Spam Mail Prediction Using Python

A **machine learning project** that predicts whether an email is **Spam** or **Ham (Not Spam)** using **Python, scikit-learn, and TF-IDF Vectorization**.

---

## 📌 Overview
This project demonstrates how to train a **Logistic Regression** model to classify emails as spam or ham.  
The workflow covers **data preprocessing, feature extraction, model training, evaluation, and prediction** on new inputs.

---

## ✨ Features
- Email text preprocessing  
- TF-IDF vectorization for feature extraction  
- Logistic Regression for classification  
- Accuracy evaluation on training & test data  
- Custom email spam prediction system  

---

## 📂 Project Structure
```

├── spam\_mail\_Predection.ipynb   # Jupyter Notebook with implementation
├── mail\_data.csv                # Dataset containing spam/ham emails
└── README.md                    # Documentation

````

---

## ⚙️ Installation
1. Clone the repository:
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

---

## 🚀 Workflow

1. **Load Dataset** → Import CSV data into pandas DataFrame
2. **Preprocess Data** → Handle missing values, encode labels (spam=0, ham=1)
3. **Split Data** → Train-Test split
4. **Vectorize Text** → TF-IDF vectorizer to convert text into feature vectors
5. **Train Model** → Logistic Regression classifier
6. **Evaluate Model** → Accuracy on training & testing sets
7. **Predict** → Classify custom emails as spam/ham

---

## 🖥️ Example Prediction

```python
input_mail = ["Congratulations! You have won a free ticket. Claim now!"]
input_vector = tfidf_vectorizer.transform(input_mail)
prediction = model.predict(input_vector)

if prediction[0] == 1:
    print("Ham Mail (Not Spam)")
else:
    print("Spam Mail")
```

**Output:**

```
Spam Mail
```

---

## 📊 Results

* **Training Accuracy:** \~96–97%
* **Testing Accuracy:** \~95–96%

The model performs effectively in classifying spam emails.

---

## 🔮 Future Enhancements

* Try other ML models (Naive Bayes, SVM, Random Forest)
* Add evaluation metrics (Precision, Recall, F1-Score, ROC-AUC)
* Use advanced NLP models (Word2Vec, BERT)
* Deploy as a **Flask/Streamlit web app**

---

## 📜 License

This project is **open-source** and available for educational use.

---

## 🙌 Acknowledgements

* [scikit-learn](https://scikit-learn.org/)
* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* Dataset from Kaggle / open-source repositories

```
```
