# A Novel Approach for Identifying Fake News through Optimized BERT

This project demonstrates how to use BERT (Bidirectional Encoder Representations from Transformers) to build a fake news detection model. The approach is validated on two benchmark datasets: **ISOT Fake News Dataset** and **WELFake Dataset**.

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Evaluation](#evaluation)
- [Prediction Function](#prediction-function)
- [Web Interface](#web-interface)
- [Author](#author)

## Introduction

Fake news is a growing problem with serious consequences. This project aims to develop a model that can automatically detect fake news articles. We leverage the power of BERT, a state-of-the-art language model, to achieve this goal. The solution is tested on both the ISOT and WELFake datasets for comprehensive evaluation.

## Datasets

### ISOT Fake News Dataset
- [ISOT Fake News dataset](https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset) 🔗
- Contains two files for fake and real news in CSV format with columns: title, text, subject, and date.
- The dataset is processed and cleaned before training the model.

### WELFake Dataset
- [WELFake dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) 🔗
- Contains news articles in CSV format with columns: title, text, and label.
- The dataset is processed and cleaned before training the model.

## Installation

1. Clone the repository:  
   `git clone https://github.com/Vikas2882/Fake_News_Detection.git`

2. Change directory to the project folder: 

    2.1 For training model on ISOT dataset `cd ISOT`

    2.1 For training model on WELFake dataset `cd WELFake`

3. Install required libraries:  
   `pip install -r requirements.txt`  
   Required packages include:
    - transformers
    - pandas
    - scikit-learn
    - seaborn
    - matplotlib
    - wordcloud
    - gensim
    - nltk
    - tensorflow

4. Run the training scripts:  
   - For ISOT: `python ISOT/ISOT_FND_Code.ipynb`
   - For WELFake: `python WELFake/WELFake_FND_Code.ipynb`

5. Run the Web App:  
   - For ISOT: `python ISOT/ISOT_UI.py`
   - For WELFake: `python WELFake/WELFake_UI.py`

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Gradio

## Usage

1. **Mount Google Drive (if using Colab):** Mount your Google Drive to access the dataset stored there.
2. **Load Data:** Load the true and fake news datasets using pandas.
3. **Data Preprocessing:** Clean and preprocess the data (removing stop words, stemming, tokenization, padding).
4. **Model Construction:** Load the pre-trained BERT model for sequence classification.
5. **Fine-tuning:** Fine-tune the BERT model on the training data.
6. **Model Saving and Loading:** Save and load the trained model for later use.
7. **Testing and Evaluation:** Test the model on the test data and evaluate its performance.

## Model

The project uses a pre-trained BERT model for sequence classification. The model is fine-tuned separately on the ISOT and WELFake datasets to achieve high accuracy in detecting fake news articles.

1. Trained BERT model on ISOT:
[https://drive.google.com/file/d/1ceHo3p2sA1pVsPFlpbsA09PiBpkPhosx/view?usp=sharing 🔗](https://drive.google.com/file/d/1ceHo3p2sA1pVsPFlpbsA09PiBpkPhosx/view?usp=sharing)
1. Trained BERT model on WELFake: 
[https://drive.google.com/file/d/1SMcjjHw9Oj7isEQq82F2NidWkU4aL7-S/view?usp=sharing 🔗](https://drive.google.com/file/d/1SMcjjHw9Oj7isEQq82F2NidWkU4aL7-S/view?usp=sharing)

## Evaluation

The models are evaluated using various metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

## Prediction Function

A function `predict_fake_news(text)` is provided to predict whether a given news article is fake or not.

## Web Interface

A simple web interface is provided using Gradio for easy interaction with the fake news detection models.

## Author

- Name: Vikas  
- Email: vk2882590@gmail.com  
- GitHub: [https://github.com/Vikas2882/Fake_News_Detection.git](https://github.com/Vikas2882/Fake_News_Detection.git) 
