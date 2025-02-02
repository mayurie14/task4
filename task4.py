import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

# Sample social media data
data = {
    'Post': [
        "I love the new features in the latest update!",
        "This product is terrible, completely disappointed.",
        "Not bad, but it could be better.",
        "Absolutely fantastic experience, highly recommend!",
        "Worst customer service ever encountered.",
        "The brand keeps improving every year, great job!",
        "I don’t like the new design, it feels outdated.",
        "This is the best purchase I've made this year!",
        "Service was okay, nothing special though.",
        "Horrible quality, will never buy again."
    ]
}

df = pd.DataFrame(data)

# Sentiment Analysis using TextBlob
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

df['Sentiment'] = df['Post'].apply(get_sentiment)

# Classify Sentiments
def classify_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment_Label'] = df['Sentiment'].apply(classify_sentiment)

# Display the dataset with sentiment scores
print(df)

# Visualization
# 1. Sentiment Distribution
sns.countplot(x='Sentiment_Label', data=df, palette='Set2')
plt.title('Sentiment Distribution')
plt.show()

# 2. Sentiment Polarity Distribution
sns.histplot(df['Sentiment'], bins=10, kde=True, color='purple')
plt.title('Sentiment Polarity Distribution')
plt.show()

# 3. Word Cloud for Positive and Negative Sentiments
from wordcloud import WordCloud

positive_text = ' '.join(df[df['Sentiment_Label'] == 'Positive']['Post'])
negative_text = ' '.join(df[df['Sentiment_Label'] == 'Negative']['Post'])

# Positive Word Cloud
plt.figure(figsize=(10, 5))
wordcloud_pos = WordCloud(background_color='white', colormap='Greens').generate(positive_text)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Sentiment Word Cloud')
plt.show()

# Negative Word Cloud
plt.figure(figsize=(10, 5))
wordcloud_neg = WordCloud(background_color='white', colormap='Reds').generate(negative_text)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Sentiment Word Cloud')
plt.show()
