from newspaper import Article
from bs4 import BeautifulSoup
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict
import re


PAYWALL_TEXT = "For subscribers"  

articles_by_date = defaultdict(lambda: {"titles": [], "summaries": [], "urls": []})
paywalled_articles = 0
non_paywall_article_count = 0

def extract_published_date(article_html):

    pattern = r'Published(?:\s|&nbsp;|<!--.*?-->)+([A-Za-z]+\s+\d{1,2},\s+\d{4})'   
    match = re.search(pattern, article_html)
    if match:
        date_str = match.group(1)  
        try:
            date_obj = datetime.strptime(date_str, "%b %d, %Y")
            return date_obj.strftime("%Y-%m-%d")
        except:
            return None
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



import csv

import csv, json

out_path = "st_articles_summary.csv"

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "num_titles", "titles", "summaries", "urls"])

    for dt in sorted(articles_by_date.keys()):
        data = articles_by_date[dt]
        writer.writerow([
            dt,
            len(data["titles"]),
            json.dumps(data["titles"], ensure_ascii=False),
            json.dumps(data["summaries"], ensure_ascii=False),
            json.dumps(data["urls"], ensure_ascii=False),
        ])

print(f"Wrote {len(articles_by_date)} rows to {out_path}")


