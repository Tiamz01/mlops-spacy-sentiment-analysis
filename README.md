# MLOps project Training and Deployment of Spacy model for Sentiment analysis

Pet project / Capstone project for DataTalks.Club MLOps ZoomCamp`24: 

Spacy model trained on dataset based on [Amazon Reviews'23](https://amazon-reviews-2023.github.io/) processed via my [Data Engineering project Amazon Reviews](https://github.com/dmytrovoytko/data-engineering-amazon-reviews) ETL.

![MLOps project Spacy model for Sentiment analysis](/screenshots/amazon-reviews-sentiment-mlops.png)

Project can be tested and deployed in cloud virtual machine (AWS, Azure, GCP), **GitHub CodeSpaces** (the easiest option, and free), or locally without GPU.

To reproduce and review this project it would be enough less than an hour, prepared dataset is not huge as original, so you don't need much disk space. For GitHub CodeSpace option you don't need to use anything extra at all - just your favorite web browser + GitHub account is totally enough.

## Problem statement

Modern technologies, social media, messengers and chat bots, including ChatGPT, trained us to "expect" almost immediate response. As a result slow response often becomes a way "out of business". E-commerce websites automated shopping processes, but response on customer/user feedback yet is not so fast as we'd like to have, however it's quite critical, agree?

An easy step to improve such communications would be a sentiment analysis of user's feedback to filter which messages need extra attention. And that's where Machine Learning could shine.

Can we make it happen with limited resources, so even blog could afford it without spending money on those 'chatgpt'-like platforms or resource demanding tech using TensorFlow transformers?

I decided to experiment with several fast and light ML NLP libraries that can run on CPU, so it could be deployed on inexpensive hosting supporting python apps:
- [NLTK Vader](https://www.nltk.org/howto/sentiment.html)
- [TextBlob](https://textblob.readthedocs.io/en/dev/)
- [SpaCy](https://spacy.io/)

Testing showed: yes, they are really fast and easy to implement, but accuracy is not very high - around 79-80%.

However, SpaCy is different - it can be trained, so let's do it!

What dataset can we use for this, with variety of measured feedback? Of course from Amazon - the Everything Store. 
In [my previous project](https://github.com/dmytrovoytko/data-engineering-amazon-reviews) I processed Amazon dataset 2023. Original dataset is huge, with millions of reviews on more than 30 categories - from Toys and Games to Clothing and Electronics, gigabytes of data.

For this project I chose to work with a much smaller subset - only years 2020-2022, Kindle Store books. Extracted and stored [here](https://github.com/dmytrovoytko/reviews-sentiment-dataset).

## 🎯 Goals

This is my MLOps project started during [MLOps ZoomCamp](https://github.com/DataTalksClub/mlops-zoomcamp)'24.
And **the main goal** is straight-forward: build an end-to-end Machine Learning project - load data, analyze it, preprocess, train ML model, test it, deploy it, monitor performance.  

Dataset **Reviews** (original) contains user's ratings from 1 to 5. I used book review texts with ratings 1-3 as examples of negative sentiment, and 4-5 as positive. Actually "4" is a bit tricky, because many readers described partly negative reasons why they didn't gave "5". Samples are [here](/data).

Thanks to MLOps ZoomCamp for the reason to learn many new tools! 

## :toolbox: Tech stack
