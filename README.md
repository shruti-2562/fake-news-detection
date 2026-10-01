# Fake News Detection Framework

A Machine Learning based web application that classifies news articles as **Fake or Genuine** using Natural Language Processing and Streamlit.

## Features

* Fake/Genuine news classification
* TF-IDF text feature extraction
* Logistic Regression classification
* Prediction confidence and probability
* Article statistics
* Dataset analysis and visualizations
* Model performance evaluation
* Online deployment

## Machine Learning

**Algorithm:** Logistic Regression
**Feature Extraction:** TF-IDF Vectorization

### Model Performance

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 98.8% |
| Precision | 98.4% |
| Recall    | 99.2% |
| F1-Score  | 98.8% |

## Dataset

The dataset contains **44,898 news articles**.

```text
title, text, subject, date, label
```

**Label:** `0 = Fake News`, `1 = Genuine News`

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

## Project Structure

```text
fake-news-detection/
├── app.py
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
└── requirements.txt
```

## Run Locally

Install dependencies:

```bash
py -m pip install -r requirements.txt
```

Run the application:

```bash
py -m streamlit run app.py
```

**Local Website:**
http://localhost:8504

## Deployment

**Live Website:**
https://fake-news-detection-scq7yaagawmun3jjyp7clc.streamlit.app

## Objective

To develop a machine learning based text classification system that helps identify whether a given news article is Fake or Genuine.

## Future Scope

* Multilingual fake news detection
* Advanced NLP models such as BERT
* Real-time news verification
* Explainable AI techniques

> This project is developed for educational purposes. Predictions should not be considered a definitive verification of real-world news.
