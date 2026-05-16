# 🎬 Sentiment Analysis Web App

A Deep Learning based Sentiment Analysis Web Application that predicts whether a movie review or text review is **Positive 😊** or **Negative 😡** using NLP and GRU/LSTM models.

## 🚀 Live Demo

https://anees3131-sentiment-analysis-app.hf.space

---

## 📌 Project Overview

This project uses Natural Language Processing (NLP) and Deep Learning techniques to analyze the sentiment of text reviews.

Users can enter any movie review or text input, and the model predicts:

* Positive Sentiment 😊
* Negative Sentiment 😡

along with a confidence score.

---

## 🧠 Features

* NLP text preprocessing
* Tokenization and sequence padding
* Deep Learning based sentiment classification
* GRU/LSTM model implementation
* Interactive Streamlit UI
* Real-time prediction
* Confidence score visualization
* Cloud deployment using Hugging Face Spaces

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NLP
* Streamlit
* NumPy
* Pickle
* Hugging Face Spaces
* Docker

---

## 📂 Project Structure

```bash
sentiment-analysis-app/
│
├── app.py / streamlit_app.py
├── requirements.txt
├── Dockerfile
├── gru_model.h5
├── tokenizer.pkl
├── sentiment_analysis_training.ipynb
└── README.md
```

---

## ⚙️ Model Workflow

1. Text Input
2. Text Tokenization
3. Sequence Padding
4. Sentiment Prediction using GRU/LSTM
5. Confidence Score Display

---

## 📊 Model Performance

* GRU Accuracy: ~87%
* LSTM Accuracy: ~86%

---

## ▶️ Run Locally

### Clone Repository

```bash
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 💡 What I Learned

* NLP preprocessing techniques
* Sequence modeling with GRU/LSTM
* Model deployment challenges
* TensorFlow/Keras compatibility handling
* Streamlit web app development
* Docker-based deployment
* End-to-end ML workflow

---

## 🌐 Deployment

The application is deployed on Hugging Face Spaces using Streamlit and Docker.

---

## 👨‍💻 Author

Muhammad Anees

---

## ⭐ Future Improvements

* Add neutral sentiment support
* Improve UI/UX
* Add charts and analytics
* Use transformer-based models
* Add multilingual sentiment analysis
