---

# Analyzing Sentiment in Trip Advisor Reviews using Hugging Face’s Transformers Model

In today's data-driven world, businesses and organizations rely heavily on customer reviews and feedback to improve their products and services. Analyzing sentiment in these reviews can provide valuable insights into customer perceptions and opinions. In this blog post, we'll explore how to perform sentiment analysis on a collection of hotel reviews using Natural Language Processing (NLP) techniques. We'll walk through the entire process, from data loading to visualization, using Python and popular NLP libraries.

## 1. Introduction

Customer reviews play a crucial role in shaping business decisions. Sentiment analysis, a subfield of NLP, involves determining the sentiment expressed in a piece of text, such as a review. In this analysis, we'll use a combination of NLTK and the transformers library to analyze the sentiment of hotel reviews and visualize the results.

## 2. Libraries and Data Loading

We start by importing the necessary Python libraries and loading our review data into a pandas DataFrame. You can download the dataset from [this Kaggle link](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews).

```python
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Load data
df = pd.read_csv('reviews.csv')
```

## 3. Data Exploration

Before diving into the analysis, let's take a quick look at the first few rows of the dataset to understand its structure and content.

```python
df.head(10)
```

## 4. Text Preprocessing

To prepare the text for sentiment analysis, we need to tokenize the words and perform part-of-speech tagging. Additionally, we'll identify named entities in the text.

```python
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Tokenization and part-of-speech tagging
ex = df['Review'][6778]
tk = nltk.word_tokenize(ex)
pos = nltk.pos_tag(tk)

# Named entity recognition
entity = nltk.chunk.ne_chunk(pos)
entity.pprint()
```

## 5. Sentiment Analysis using NLTK

NLTK provides the SentimentIntensityAnalyzer class for sentiment analysis. We can use it to calculate polarity scores for negative, neutral, and positive sentiments.

```python
sia = SentimentIntensityAnalyzer()
sia.polarity_scores(ex)
```

## 6. Sentiment Analysis using Transformers

The transformers library offers pre-trained models for various NLP tasks, including sentiment analysis. We'll use the "cardiffnlp/twitter-roberta-base-sentiment" model to analyze sentiment.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

en_text = tokenizer(ex, return_tensors='pt')
result = model(**en_text)
ratings = result[0][0].detach().numpy()
ratings = softmax(ratings)
ratings_dict = {
    'rob_neg': ratings[0],
    'rob_neu': ratings[1],
    'rob_pos': ratings[2],
}
print(ratings_dict)
```

## 7. Creating a Result DataFrame

Iterating through the reviews, we'll apply the sentiment analysis function and store the results in a DataFrame.

```python
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        review = row['Review']
        rob_res = polarity_scores_rob(review)
        res[i] = rob_res
    except RuntimeError as e:
        res[i] = np.nan
        print(f"Error occurred at id {i}: {str(e)}")
        continue
    except IndexError as e:
        res[i] = np.nan
        print(f"IndexError occurred at id {i}: {str(e)}")
        continue

result_df = pd.DataFrame(res).T
result_df = result_df.reset_index().rename(columns={'index': "Id"})
result_df = pd.concat([result_df, df], axis=1)
```


## 8. Visualizing Sentiment Analysis Results

Visualizations are an excellent way to communicate the results of your sentiment analysis. Let's create a variety of visualizations to highlight different aspects of the sentiment distribution in the hotel reviews.

### 8.1 Pair Plot of Sentiment Scores

We've already created a pair plot that showcases the relationships between sentiment scores (rob_neg, rob_neu, rob_pos) and review ratings. This plot gives a quick overview of how sentiment and ratings are distributed across the dataset.

```python
sns.pairplot(data=result_df, vars=['Id', 'rob_neg', 'rob_neu', 'rob_pos'], hue='Rating', palette='tab10')
plt.show()
```
![senti_pic](https://github.com/rahilfaizan/sentiment_analysis/assets/51293067/dc5be007-8e81-4434-8639-d76b7f9ddcec)

### 8.2 Distribution of Sentiment Scores

Let's visualize the distribution of each sentiment score (negative, neutral, positive) using histograms. This will help us understand the overall sentiment tendencies in the reviews.

```python
plt.figure(figsize=(10, 6))
sns.histplot(data=result_df, x='rob_neg', bins=20, color='red', label='Negative')
sns.histplot(data=result_df, x='rob_neu', bins=20, color='gray', label='Neutral')
sns.histplot(data=result_df, x='rob_pos', bins=20, color='green', label='Positive')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Distribution of Sentiment Scores')
plt.legend()
plt.show()
```
![pos_neg_bar](https://github.com/rahilfaizan/sentiment_analysis/assets/51293067/374e7d32-7574-46c1-a11e-434847bba13c)

### 8.3 Box Plot of Sentiment Scores by Rating

A box plot can help us visualize how sentiment scores vary based on different review ratings. This can provide insights into how sentiment relates to the perceived quality of the hotel experience.

```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=result_df, x='Rating', y='rob_pos', palette='viridis')
plt.xlabel('Review Rating')
plt.ylabel('Positive Sentiment Score')
plt.title('Distribution of Positive Sentiment by Review Rating')
plt.show()
```
![pos_neg_box](https://github.com/rahilfaizan/sentiment_analysis/assets/51293067/62bacf21-31b1-4e73-bed2-cb43240be6b8)

## 9. Conclusion

Sentiment analysis is a powerful tool for understanding customer opinions and attitudes expressed in reviews. By combining NLTK and transformers libraries, we've demonstrated a comprehensive approach to analyzing sentiment in a collection of hotel reviews. This analysis provides valuable insights that can help businesses make informed decisions to enhance customer satisfaction and improve their products and services.

In this post, we've covered data loading, data exploration, text preprocessing, sentiment analysis using NLTK and transformers, error handling, and data visualization. By following these steps, you can conduct your own sentiment analysis on customer reviews and gain meaningful insights to drive your business forward.

---
