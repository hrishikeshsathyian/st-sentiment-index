from bs4 import BeautifulSoup
import requests
from newspaper import Article



article_count = 0
paywalled_articles = 0 
business_articles = 0
for month in range(10,0,-1):
    url = "https://www.straitstimes.com/sitemap/2025/{}/feeds.xml".format(month)
    soup = BeautifulSoup(requests.get(url).content, "xml")
    for loc in soup.find_all("loc"):
        if "/business/" in loc.text:
            business_articles += 1
            try:
                article = Article(loc.text)
                article.download()
                article.parse()
                article.nlp()

                article_count += 1
                print(f"✓ {article.publish_date.date()}: {article.title}")

                

            except Exception as e:
                print(f"Error processing {loc.text}: {e}")
                continue


print("\n" + "="*80)
print("BUSINESS ARTICLE STATISTICS")
print("="*80)
print(f"Total business articles found: {business_articles}")
print(f"Total paywalled business articles: {paywalled_articles}")
print(f"Total non-paywalled business articles: {business_articles - paywalled_articles}")