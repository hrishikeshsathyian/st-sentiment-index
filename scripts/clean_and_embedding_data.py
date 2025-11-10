## The point of this script is to clean each title and summary, before grouping all titles/summaries for the same date into one row
import re
import pandas as pd 
import sentence_transformers
import pickle
from datetime import datetime, timedelta
from time import time

print("="*80)
print("TESTING MULTIPLE SENTENCE TRANSFORMER MODELS")
print("="*80)

df = pd.read_csv('/Users/hrishikeshsathyian/Desktop/nlp-insa/dataset/st_business_articles.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"Total articles: {len(df)}")
print(f"Total unique dates: {df['date'].nunique()}")

# Clean text 
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
def get_group_date(dt):
    # Determine preliminary day based on 5 PM cutoff
    if dt.hour >= 16:
        group_dt = dt + timedelta(days=1)
    else:
        group_dt = dt
    # Weekend handling: Friday 5 PM → Monday 5 PM
    weekday = group_dt.weekday()  # Monday=0, Sunday=6
    # If the adjusted date falls on Saturday (5) or Sunday (6), shift to Monday
    if weekday == 5:  # Saturday
        group_dt += timedelta(days=2)  # Saturday → Monday
    elif weekday == 6:  # Sunday
        group_dt += timedelta(days=1)  # Sunday → Monday
    return pd.Timestamp(group_dt.date())

# Apply the grouping function
df['group_date'] = df['date'].apply(get_group_date)

# Aggregate by the new group_date
daily_articles = {}
for group_date, group_df in df.groupby('group_date'):
    all_titles = " ".join(group_df['title'].tolist())
    all_summaries = " ".join(group_df['summary'].tolist())
    daily_articles[group_date] = {
        'titles': all_titles,
        'summaries': all_summaries,
        'article_count': len(group_df),
        'urls': '|'.join(group_df['url'].tolist())
    }

print(f"\nGrouped into {len(daily_articles)} unique trading days")

# Models to test
models_to_test = {
    'all-MiniLM-L6-v2': {
        'name': 'all-MiniLM-L6-v2',
        'dims': 384,
        'prefix': None
    },
    'all-mpnet-base-v2': {
        'name': 'all-mpnet-base-v2',
        'dims': 768,
        'prefix': None
    },
    'bge-small-en-v1.5': {
        'name': 'BAAI/bge-small-en-v1.5',
        'dims': 384,
        'prefix': None
    },
    'bge-base-en-v1.5': {
        'name': 'BAAI/bge-base-en-v1.5',
        'dims': 768,
        'prefix': None
    },
    'e5-small-v2': {
        'name': 'intfloat/e5-small-v2',
        'dims': 384,
        'prefix': 'passage: '
    },
    'e5-base-v2': {
        'name': 'intfloat/e5-base-v2',
        'dims': 768,
        'prefix': 'passage: '
    }
}

# Test each model
for model_key, model_info in models_to_test.items():
    print("\n" + "="*80)
    print(f"Testing: {model_info['name']}")
    print(f"Embedding dimensions: {model_info['dims']}")
    print("="*80)
    
    try:
        # Load model
        start_time = time()
        model = sentence_transformers.SentenceTransformer(model_info['name'])
        load_time = time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Generate embeddings
        final_data = []
        start_time = time()
        
        for date in sorted(daily_articles.keys()):
            titles = daily_articles[date]['titles']
            summaries = daily_articles[date]['summaries']
            
            # Add prefix if needed
            if model_info['prefix']:
                titles = model_info['prefix'] + titles
                summaries = model_info['prefix'] + summaries
            
            title_embedding = model.encode(titles, show_progress_bar=False)
            summary_embedding = model.encode(summaries, show_progress_bar=False)
            
            row = {
                'date': date,
                'title_embedding': title_embedding,
                'summary_embedding': summary_embedding,
                'article_count': daily_articles[date]['article_count'],
                'urls': daily_articles[date]['urls']
            }
            final_data.append(row)
        
        encoding_time = time() - start_time
        print(f"✅ Encoded {len(final_data)} days in {encoding_time:.2f}s")
        print(f"   Average: {encoding_time/len(final_data):.3f}s per day")
        
        # Save embeddings
        final_df = pd.DataFrame(final_data)
        output_file = f'daily_embeddings_{model_key}.pkl'
        final_df.to_pickle(output_file)
        print(f"✅ Saved to '{output_file}'")
        
        # Show summary
        print(f"\nModel Summary:")
        print(f"  Original articles: {len(df)}")
        print(f"  Unique trading days: {len(final_df)}")
        print(f"  Title embedding dimensions: {model_info['dims']}")
        print(f"  Summary embedding dimensions: {model_info['dims']}")
        print(f"  Total features per date: {model_info['dims'] * 2} ({model_info['dims']} + {model_info['dims']})")
        print(f"  Final shape: {final_df.shape}")
        
        # Show example
        print(f"\nExample output:")
        print(final_df[['date', 'article_count']].head())
        
    except Exception as e:
        print(f"❌ Error with {model_info['name']}: {str(e)}")
        continue

print("\n" + "="*80)
print("ALL MODELS COMPLETE")
print("="*80)
print("\nGenerated files:")
for model_key in models_to_test.keys():
    print(f"  - daily_embeddings_{model_key}.pkl")
print("\nNext step: Run the comparison training script to see which model performs best!")