## The point of this script is to clean each title and summary, before grouping all titles/summaries for the same date into one row

import re
import pandas as pd 
import sentence_transformers
import pickle

print("Script")

df = pd.read_csv('/Users/hrishikeshsathyian/Desktop/nlp-insa/dataset/st_business_articles.csv')

print(f"Total articles: {len(df)}")
print(f"Total unique dates: {df['date'].nunique()}")

# Clean text 
# We choose to use minimal cleaning here because pre-trained models like SentenceTransformer already 
# is trained to handle the noise in training data. Over-doing the cleaning would actually hurt the model's performance at this point.
def clean_text(text): 
    if pd.isna(text):
        return ""
    AD_TEXT = "Sign up now: Get ST's newsletters delivered to your inbox"
    if AD_TEXT in text:
        text = text.replace(AD_TEXT, "")
    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = text.strip()
    text = ' '.join(text.split())
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text

df['title'] = df['title'].apply(clean_text)
df['summary'] = df['summary'].apply(clean_text)


# Group articles by date 
daily_articles = {} 
for date in df['date'].unique(): 
    date_df = df[df['date'] == date]
    all_titles = " ".join(date_df['title'].tolist())
    all_summaries = " ".join(date_df['summary'].tolist())

    daily_articles[date] = {
        'titles': all_titles,
        'summaries': all_summaries, 
        'article_count': len(date_df),
        'urls': '|'.join(date_df['url'].tolist())
    }


# Convert each to vector embeddings 
print("\n" + "="*80)
print("Convert to vector embeddings")

model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

final_data = [] 
for date in sorted(daily_articles.keys()):
    titles = daily_articles[date]['titles']
    summaries = daily_articles[date]['summaries']
    article_count = daily_articles[date]['article_count']
    urls = daily_articles[date]['urls']

    title_embedding = model.encode(titles)
    summary_embedding = model.encode(summaries)

    row = {
        'date': date,
        'title_embedding': title_embedding,
        'summary_embedding': summary_embedding,
        'article_count': article_count,
        'urls': urls
    }
    final_data.append(row)


final_df = pd.DataFrame(final_data)
final_df.to_pickle('daily_embeddings.pkl')

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Original articles: {len(df)}")
print(f"Unique dates: {len(final_df)}")
print(f"Title embedding dimensions: 384")
print(f"Summary embedding dimensions: 384")
print(f"Total features per date: 768 (384 + 384)")
print(f"Final shape: {final_df.shape}")
print("\n✅ Saved to 'daily_embeddings.csv'")

# Show example
print("\n" + "="*80)
print("EXAMPLE OUTPUT")
print("="*80)
print(final_df[['date', 'article_count']].head())
print(f"\nColumn sample: {list(final_df.columns[:10])}...")