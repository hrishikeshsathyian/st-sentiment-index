import feedparser, requests
from bs4 import BeautifulSoup
from newspaper import Article
from datetime import datetime
from collections import defaultdict
import time
import re

articles_by_date = defaultdict(lambda: {"titles": [], "summaries": [], "urls": []})
non_paywall_article_count = 0
paywalled_articles = 0
PAYWALL_TEXT = "For subscribers"


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

for month in range(10, 9, -1): 
    url = "https://www.straitstimes.com/sitemap/2025/{}/feeds.xml".format(month)
    soup = BeautifulSoup(requests.get(url).content, "xml")

    for loc in soup.find_all("loc"):
        if "/business/" in loc.text:
            try:
                article = Article(loc.text)
                article.download()
                html = article.html
                # Check paywall
                if PAYWALL_TEXT in html:
                    paywalled_articles += 1
                    continue
                
                # Extract date from HTML
                published_date = extract_published_date(html)
                
                if not published_date:
                    print(f"Could not extract date from: {loc.text}")
                    continue

                # Parse and get summary
                article.parse()
                article.nlp()
                
                # if len(articles_by_date[published_date]["titles"]) >= 1:
                #     continue  # limit to 1 articles per date

                # Store everything
                articles_by_date[published_date]["titles"].append(article.title)
                articles_by_date[published_date]["summaries"].append(article.summary)
                articles_by_date[published_date]["urls"].append(loc.text)
                
                non_paywall_article_count += 1
                print(f"✓ {published_date}: {article.title}")
                
                time.sleep(1) 
                
            except Exception as e:
                print(f"Error processing {loc.text}: {e}")
                continue



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


