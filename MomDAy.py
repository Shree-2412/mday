import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

#Sample Data : Simulated Mother's Day tweets
data = {
    "date": pd.date_range(start="2025-05-01", periods=10),
    "tweet_text": [
        "Happy Mother's Day to the best mom ever! Love you so much!",
        "I miss my mom every day. Mother's Day is hard without her.",
        "Just sent flowers to mom! Hope she likes them 💐 #MothersDay",
        "Mother's Day is a scam by the greeting card companies.",
        "Grateful for my mother’s endless support and love. #ThanksMom",
        "No words can express how much I love my mom. #MothersDay",
        "Mom, thanks for always being there. Happy Mother's Day!",
        "Feeling a bit down today... wish I could hug my mom.",
        "Celebrating the strength of all the mothers out there today.",
        "Another Mother’s Day, another brunch. Not really feeling it."
    ]
}

df = pd.DataFrame(data)

# Text Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]','', text) #removes non alphabetic characters
    text = text.lower() #converting into lowercase
    tokens = text.split() #splitting into words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ". join(tokens) #Join the changed words back into sentences

df["cleaned_text"] = df["tweet_text"].apply(clean_text)


# Sentiment Analysis with VADER
sia = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['tweet_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['sentiment'] = df['sentiment_score'].apply(lambda x: 'positive' if x>0.1 else ('negative' if x<-0.1 else 'neutral'))

#To[ic Modeling with LDA
vectorizer = CountVectorizer(max_df=0.9, min_df=1, stop_words=('english'))
X = vectorizer.fit_transform(df['cleaned_text'])

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)
"""Transforms cleaned text into a term-document matrix.

max_df=0.9: ignore words that appear in >90% of texts (too common).

min_df=1: include words that appear in at least one text."""

#Extracting topics
def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-no_top_words -1:-1]]
        topics.append(", ".join(top_features))
    return topics

topics = display_topics(lda, vectorizer.get_feature_names_out(),5)
df['topic'] = [topics[np.argmax(topic)] for topic in lda.transform(X)]

df.to_csv("mothers_day_sentiment.csv", index=False)

#Generate Word Cloud for cleaned_text
all_words = " ".join(df["cleaned_text"])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

#Plotting the word cloud
plt.figure(figsize=(10,5))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Mother's Day Tweets",fontsize = 16)
plt.tight_layout()
plt.show()

#Sentiment distribution bar plot using Python code
