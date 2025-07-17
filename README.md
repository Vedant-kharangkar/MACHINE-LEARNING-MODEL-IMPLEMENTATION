# MACHINE-LEARNING-MODEL-IMPLEMENTATION

**COMPANY: CODTECH IT SOLUTIONS

NAME:VEDANT AVINASH KHARANGKAR

INTERN ID:CT06DH1250

DOMAIN: PYTHON PROGRAMMING

DURATION:6 WEEKS

MENTOR: NEELA SANTOSH**

---

# Spam Email Detection using Scikit-learn

# Project Description
This project demonstrates how to build a machine learning model using Scikit-learn to classify emails as spam or ham (not spam). By converting raw email text into numerical features using NLP techniques, the model predicts whether an incoming message is spam. The entire process is implemented in Jupyter Notebook, making it easy to understand, modify, and experiment with.

The goal is to automate email filtering systems to improve user experience and cybersecurity.

---

# Tools & Technologies
Programming Language: Python 3

Libraries:

Pandas: Data loading and cleaning

NumPy: Numerical computations

Scikit-learn: Machine learning (TfidfVectorizer, LogisticRegression, model evaluation)

Environment: Jupyter Notebook

Vectorizer: TfidfVectorizer for feature extraction from email content

---

# Workflow

Load and Prepare Data
Load the dataset (mail_data.csv) using pandas.read_csv() and clean missing values.

Label Encoding
Encode labels as:

spam → 0

ham → 1

Split Dataset
Split data into features (X) and labels (Y), then perform an 80/20 train-test split.

Feature Extraction
Use TfidfVectorizer to convert text to numerical vectors. Convert text to lowercase and remove common stopwords.

Model Training
Train a Logistic Regression model on the training set.

Model Evaluation
Use accuracy_score to evaluate the model on both training and test datasets.

Real-Time Prediction
Build a simple system that accepts email text input and predicts whether it's spam or ham using the trained model.

---

# Learning Outcomes

Clean and preprocess text data for machine learning

Use TfidfVectorizer for feature extraction in NLP

Apply train-test split for model validation

Train a classification model using logistic regression

Evaluate model performance using accuracy

Build a basic real-time prediction interface

# Use Cases

Email Filters: Automatically detect and block spam emails

Customer Support Bots: Ignore spam inputs in live chat systems

Cybersecurity Tools: Detect suspicious messages in enterprise systems

NLP Projects: Foundation for tasks like sentiment analysis, topic classification

# Possible Future Enhancements

Add a web interface using Flask or Streamlit

Train on a larger or more balanced dataset

Compare results with Naive Bayes, SVM, or Random Forest

Use advanced NLP techniques like lemmatization, stemming, n-grams

Deploy model in production (e.g., integrate with Gmail or Outlook)

# How to Run the Code

1. Launch Jupyter Notebook (via Anaconda Navigator or terminal)

2. Open the .ipynb notebook file

3. Make sure the required packages are installed:

4. bash
Copy
Edit
pip install numpy pandas scikit-learn
Ensure mail_data.csv is in the correct path

5. Run the cells step by step to see the outputs

For real-time prediction, enter custom email content when prompted

# OUTPUT

<img width="1571" height="949" alt="Image" src="https://github.com/user-attachments/assets/ca14b6a6-95f9-4489-ab48-3deeb0cd8de5" />

<img width="1726" height="988" alt="Image" src="https://github.com/user-attachments/assets/c1cb9c90-63a5-463e-8596-4a4c3149e614" />

<img width="1627" height="992" alt="Image" src="https://github.com/user-attachments/assets/bf43c044-4f92-4f90-aa41-4665caab03f2" />

<img width="1597" height="1011" alt="Image" src="https://github.com/user-attachments/assets/5d254d29-278c-4777-83e8-df507bef5bba" />

<img width="1581" height="1010" alt="Image" src="https://github.com/user-attachments/assets/c76d6b81-f968-41c6-b146-9e75ef815349" />

<img width="1604" height="1010" alt="Image" src="https://github.com/user-attachments/assets/2f39146a-12ea-4e70-a503-958d929a34c2" />

<img width="1589" height="1013" alt="Image" src="https://github.com/user-attachments/assets/323805f2-8398-4711-bcda-62c0674bdd1c" />
