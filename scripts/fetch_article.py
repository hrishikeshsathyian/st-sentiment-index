from newspaper import Article
from bs4 import BeautifulSoup
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict
import re
import pandas as pd


PAYWALL_TEXT = "For subscribers"  

articles_by_date = defaultdict(lambda: {"titles": [], "summaries": [], "urls": []})
paywalled_articles = 0
non_paywall_article_count = 0

def extract_published_date(article_html):
    pattern = r'Published(?:\s|&nbsp;|<!--.*?-->)+([A-Za-z]+\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}\s+[APM]{2})'
    match = re.search(pattern, article_html)
    if match:
        dt = datetime.strptime(match.group(1), "%b %d, %Y, %I:%M %p")
        return dt
    return None


def extract_article(loc_text):
    global paywalled_articles, non_paywall_article_count
    
    try:
        article = Article(loc_text)
        article.download()
        html = article.html
        
        # Check paywall
        if PAYWALL_TEXT in html:
            paywalled_articles += 1
            return None
        
        # Extract date
        published_date = extract_published_date(html)
        if not published_date:
            print(f"Could not extract date from: {loc_text}")
            return None
        
        # Parse article
        article.parse()
        article.nlp()
        
        # Store article info
        articles_by_date[published_date]["titles"].append(article.title)
        articles_by_date[published_date]["summaries"].append(article.summary)
        articles_by_date[published_date]["urls"].append(loc_text)
        
        non_paywall_article_count += 1
        print(f"✓ {published_date}: {article.title}")
        
    except Exception as e:
        print(f"Error processing {loc_text}: {e}")

urls_to_parse = []

for month in range(10, 0, -1): 
    url = f"https://www.straitstimes.com/sitemap/2025/{month}/feeds.xml"
    soup = BeautifulSoup(requests.get(url).content, "xml")
    
    for loc in soup.find_all("loc"):
        if "/business/" in loc.text:
            urls_to_parse.append(loc.text)

# Number of threads (adjust based on your system)
max_threads = 15

with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = [executor.submit(extract_article, url) for url in urls_to_parse]
    
    # Optional: wait for all tasks and show progress
    for future in as_completed(futures):
        future.result()  # triggers exception if any


rows = []

for date, data in articles_by_date.items():
    for title, summary, url in zip(data['titles'], data['summaries'], data['urls']):
        rows.append({
            'date': date,
            'title': title,
            'summary': summary,
            'url': url
        })

df = pd.DataFrame(rows)


df.to_csv("dataset/st_business_articles.csv", index=False)


