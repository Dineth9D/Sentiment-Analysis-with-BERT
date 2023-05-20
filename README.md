# Sentiment Analysis with BERT Neural Network

This project aims to perform sentiment analysis using the BERT (Bidirectional Encoder Representations from Transformers) neural network. By leveraging advanced Natural Language Processing (NLP) techniques, we can accurately calculate sentiment from textual data.

## Introduction
Sentiment analysis is the process of determining the emotional tone behind a series of words or sentences. It can be valuable in various domains, such as customer feedback analysis, social media monitoring, and market research. In this project, we utilize BERT, a state-of-the-art NLP model, to perform sentiment analysis on restaurant reviews.

## Data Collection
To gather a substantial amount of data for sentiment analysis, we employed BeautifulSoup, a Python library for web scraping. Specifically, we scraped reviews from Yelp, a popular platform for restaurant reviews. By scraping a larger scale of reviews, we can obtain diverse and representative data for our sentiment analysis model.

## Dataset
The dataset used for training and evaluation consists of restaurant reviews obtained from Yelp. Each review includes the text of the review and its corresponding sentiment label (positive or negative). The dataset was carefully curated to ensure a balanced representation of both positive and negative sentiments.

## Model
For sentiment analysis, we leveraged the transformers library, which provides easy access to pre-trained NLP models such as BERT. BERT is a transformer-based model that utilizes attention mechanisms to understand the contextual relationships between words in a text. By fine-tuning BERT on our labeled dataset, we trained a sentiment analysis model capable of accurately predicting sentiment for new reviews.


## Results
After training the sentiment analysis model on the Yelp restaurant reviews dataset, we achieved an accuracy of XX%. The model demonstrated strong performance in correctly classifying the sentiment of the reviews, thereby enabling us to extract valuable insights from customer feedback.

## Conclusion
In conclusion, this project showcases the power of advanced NLP techniques, specifically leveraging BERT, to perform sentiment analysis on restaurant reviews. By scraping a larger scale of reviews from Yelp and training a BERT-based model, we can accurately predict sentiment and gain valuable insights into customer opinions. This approach can be extended to other domains and datasets to analyze sentiment and make informed decisions based on textual data.


## Development
Open your favorite Terminal and run these commands.

- Note: Use Python 3.9

- Clone this git repo

    ```sh
    git clone https://github.com/Dineth9D/BERT_Sentiment-Analysis.git
    ```

- Install requirements

    ```sh
    !pip install -r requirements.txt
    ```

