import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import os
import warnings
warnings.filterwarnings('ignore')

# NLTK setup
import nltk
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords',    quiet=True)
nltk.download('punkt',        quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Sklearn for topic modeling
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

os.makedirs("outputs", exist_ok=True)

#  Load Reviews 
reviews = pd.read_csv("archive/olist_order_reviews_dataset.csv")
print(f"✅ Reviews loaded: {reviews.shape}")
print(reviews.head(3))

# Clean Text
    if pd.isna(text): return""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+',  '', text)
    text = re.sub(r'[^a-zA-Z\s]',     '', text)
    text = re.sub(r'\s+',             ' ', text).strip()
    return text

reviews['clean_comment'] = reviews['review_comment_message'].apply(clean_text)

# Filter rows with actual comments
reviews_with_text = reviews[reviews['clean_comment'].str.len() > 5].copy()
print(f"\n Reviews with text: {len(reviews_with_text)}")

# ── VADER Sentiment Analysis 
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if not text: return 0
    return sia.polarity_scores(text)['compound']

def classify_sentiment(score):
    if score >= 0.05:  return 'Positive'
    elif score <= -0.05: return 'Negative'
    else:              return 'Neutral'

reviews_with_text['sentiment_score'] = reviews_with_text['clean_comment'].apply(get_sentiment)
reviews_with_text['sentiment']       = reviews_with_text['sentiment_score'].apply(classify_sentiment)

print("\n📊 Sentiment Distribution:")
print(reviews_with_text['sentiment'].value_counts())

# ── Star Rating vs Sentiment 
print("\n📊 Avg Sentiment Score by Star Rating:")
print(reviews_with_text.groupby('review_score')['sentiment_score'].mean().round(3))

# ── Stopwords 
stop_words = set(stopwords.words('english'))
stop_words.update(['product','order','delivery','item','received','arrived','good','great','excellent'])

def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w not in stop_words and len(w) > 2])

reviews_with_text['filtered_text'] = reviews_with_text['clean_comment'].apply(remove_stopwords)

# LDA Topic Modeling 
vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.8)
doc_term_matrix = vectorizer.fit_transform(reviews_with_text['filtered_text'])

n_topics = 5
lda = LatentDirichletAllocation(
    n_components=n_topics, random_state=42,
    max_iter=15, learning_method='online'
)
lda.fit(doc_term_matrix)

feature_names = vectorizer.get_feature_names_out()

print("\n📋 LDA Topics:")
topic_labels = []
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    label = ' | '.join(top_words[:5])
    topic_labels.append(label)
    print(f"   Topic {topic_idx+1}: {' • '.join(top_words)}")

# Assign dominant topic
topic_assignments = lda.transform(doc_term_matrix).argmax(axis=1)
reviews_with_text['topic'] = topic_assignments

#  Plots 
fig = plt.figure(figsize=(18, 14))
fig.suptitle('NLP — Sentiment & Topic Analysis', fontsize=16, fontweight='bold')

# 1. Sentiment distribution pie
ax1 = fig.add_subplot(3, 3, 1)
sent_counts = reviews_with_text['sentiment'].value_counts()
ax1.pie(sent_counts.values, labels=sent_counts.index,
        autopct='%1.1f%%', colors=['#2ecc71','#e74c3c','#f39c12'])
ax1.set_title('Sentiment Distribution')

# 2. Star rating distribution
ax2 = fig.add_subplot(3, 3, 2)
star_counts = reviews['review_score'].value_counts().sort_index()
ax2.bar(star_counts.index, star_counts.values, color='steelblue')
ax2.set_title('Review Star Ratings')
ax2.set_xlabel('Stars')
ax2.set_ylabel('Count')

# 3. Sentiment score by star rating
ax3 = fig.add_subplot(3, 3, 3)
sentiment_by_star = reviews_with_text.groupby('review_score')['sentiment_score'].mean()
ax3.bar(sentiment_by_star.index, sentiment_by_star.values,
        color=['#e74c3c','#e74c3c','#f39c12','#2ecc71','#2ecc71'])
ax3.set_title('Avg Sentiment Score by Stars')
ax3.set_xlabel('Star Rating')
ax3.set_ylabel('Sentiment Score')
ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')

# 4. Positive word cloud
ax4 = fig.add_subplot(3, 3, 4)
pos_text = ' '.join(
    reviews_with_text[reviews_with_text['sentiment']=='Positive']['filtered_text']
)
if pos_text.strip():
    wc_pos = WordCloud(width=500, height=300, background_color='white',
                       colormap='Greens', max_words=80).generate(pos_text)
    ax4.imshow(wc_pos, interpolation='bilinear')
ax4.axis('off')
ax4.set_title('Positive Reviews — Word Cloud')

# 5. Negative word cloud
ax5 = fig.add_subplot(3, 3, 5)
neg_text = ' '.join(
    reviews_with_text[reviews_with_text['sentiment']=='Negative']['filtered_text']
)
if neg_text.strip():
    wc_neg = WordCloud(width=500, height=300, background_color='white',
                       colormap='Reds', max_words=80).generate(neg_text)
    ax5.imshow(wc_neg, interpolation='bilinear')
ax5.axis('off')
ax5.set_title('Negative Reviews — Word Cloud')

# 6. Overall word cloud
ax6 = fig.add_subplot(3, 3, 6)
all_text = ' '.join(reviews_with_text['filtered_text'])
if all_text.strip():
    wc_all = WordCloud(width=500, height=300, background_color='white',
                       colormap='Blues', max_words=80).generate(all_text)
    ax6.imshow(wc_all, interpolation='bilinear')
ax6.axis('off')
ax6.set_title('All Reviews — Word Cloud')

# 7. Topic distribution
ax7 = fig.add_subplot(3, 3, 7)
topic_counts = reviews_with_text['topic'].value_counts().sort_index()
ax7.bar([f"Topic {i+1}" for i in topic_counts.index],
        topic_counts.values, color='mediumpurple')
ax7.set_title('LDA Topic Distribution')
ax7.set_ylabel('Reviews')
ax7.tick_params(axis='x', rotation=30)

# 8. Top words per topic (topic 0 and 1)
ax8 = fig.add_subplot(3, 3, 8)
topic_data = []
for tid in range(min(3, n_topics)):
    top_w = [(feature_names[i], lda.components_[tid][i])
             for i in lda.components_[tid].argsort()[:-8:-1]]
    for word, score in top_w:
        topic_data.append({'topic': f'Topic {tid+1}', 'word': word, 'score': score})

topic_df = pd.DataFrame(topic_data)
colors_map = {f'Topic {i+1}': c for i, c in
              enumerate(['steelblue','coral','mediumseagreen','mediumpurple','darkorange'])}
for tname, grp in topic_df.groupby('topic'):
    ax8.barh(grp['word'], grp['score'],
             label=tname, alpha=0.8, color=colors_map.get(tname,'gray'))
ax8.set_title('Top Words by Topic (first 3 topics)')
ax8.legend(fontsize=8)

# 9. Sentiment over time
ax9 = fig.add_subplot(3, 3, 9)
reviews_with_text['review_creation_date'] = pd.to_datetime(
    reviews_with_text['review_creation_date'], errors='coerce'
)
monthly_sent = reviews_with_text.groupby(
    reviews_with_text['review_creation_date'].dt.to_period('M')
)['sentiment_score'].mean()
monthly_sent.index = monthly_sent.index.astype(str)
ax9.plot(monthly_sent.index, monthly_sent.values, marker='o', color='steelblue', linewidth=2)
ax9.axhline(0, color='red', linestyle='--', linewidth=0.8)
ax9.set_title('Avg Sentiment Score Over Time')
ax9.tick_params(axis='x', rotation=45)
ax9.set_ylabel('Avg Sentiment')

plt.tight_layout()
plt.savefig("outputs/module5_nlp.png", dpi=150, bbox_inches='tight')
plt.show()

# Save results
reviews_with_text[['order_id','review_score','sentiment','sentiment_score','topic']]\
    .to_csv("outputs/sentiment_results.csv", index=False)

print("\n Module 5 Complete!")
print("   Plot saved   : outputs/module5_nlp.png")
print("   Results saved: outputs/sentiment_results.csv")
