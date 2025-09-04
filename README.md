# 📧 Spam Mail Prediction Using Python

This project is a **machine learning model** that predicts whether an email is **Spam** or **Ham (Not Spam)** using **Python, scikit-learn, and TF-IDF Vectorization**.

---

## 📌 Overview

Email spam is one of the most common problems in communication systems.  
This project demonstrates how to train a **Logistic Regression** classifier to filter spam emails effectively.  
It includes data preprocessing, feature extraction, model training, evaluation, and making predictions on custom input.

---

## ✨ Features

- Clean and preprocess email dataset  
- Convert email text into numerical form using **TF-IDF Vectorizer**  
- Train a **Logistic Regression model** for classification  
- Evaluate model accuracy on training and testing sets  
- Predict whether new/unseen emails are spam or ham  

---

## 📂 Project Structure
├── spam_mail_Predection.ipynb # Jupyter Notebook with implementation
├── mail_data.csv # Dataset containing spam/ham emails
└── README.md # Documentation


---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/adityashirsatrao007/Spam-Mail-Prediction-Using-Python.git
   cd Spam-Mail-Prediction-Using-Python


Install required libraries:

pip install pandas numpy scikit-learn jupyter


Run the notebook:

jupyter notebook spam_mail_Predection.ipynb

🚀 Workflow

Load Dataset – Import CSV data into pandas DataFrame

Data Preprocessing – Handle missing values, label encode spam/ham

Train-Test Split – Split dataset for training and evaluation

Feature Extraction – Use TF-IDF Vectorizer to convert text into vectors

Model Training – Train a Logistic Regression classifier

Evaluation – Calculate accuracy on training and test sets

Prediction – Test model with custom email input

🖥️ Example Prediction
input_mail = ["Congratulations! You have won a free ticket. Claim now!"]
input_vector = tfidf_vectorizer.transform(input_mail)
prediction = model.predict(input_vector)

if prediction[0] == 1:
    print("Ham Mail (Not Spam)")
else:
    print("Spam Mail")


Output:

Spam Mail

📊 Results

Training Accuracy: ~96–97%

Testing Accuracy: ~95–96%

The Logistic Regression model performs efficiently in classifying spam vs ham emails.

🔮 Future Enhancements

Experiment with other classifiers (Naive Bayes, Random Forest, SVM)

Add advanced NLP techniques (Word Embeddings, BERT)

Deploy as a web app using Flask/Streamlit for real-time spam detection

Evaluate with additional metrics (Precision, Recall, F1-score, ROC-AUC)

📜 License

This project is open-source and free to use for educational purposes.

🙌 Acknowledgements

scikit-learn
 for machine learning algorithms

pandas
 & numpy
 for data handling

Dataset from Kaggle / open-source repositories
