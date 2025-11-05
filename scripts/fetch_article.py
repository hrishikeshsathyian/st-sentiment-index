import feedparser, requests
from bs4 import BeautifulSoup
from newspaper import Article


# # RSS validity check
# feed = feedparser.parse("https://www.straitstimes.com/news/business/rss.xml")

# for entry in feed.entries: 
#     print(entry.title)

## last modified stored as : 2025-10-08T15:59:07.000Z

non_paywall_business_articles = []
total_count = 0
paywalled_articles = 0
stop_limit = 10 # js to make testing faster

for month in range(10, 0, -1): 
    url = "https://www.straitstimes.com/sitemap/2025/{}/feeds.xml".format(month)
    soup = BeautifulSoup(requests.get(url).content, "xml")
    for loc in soup.find_all("loc"):
        if "/business/" in loc.text:
            last_modified = loc.find_next("lastmod")
            if not last_modified: 
                print("No last modified date found for article: ", loc.text)
                continue
            else: 
                last_modified = last_modified.text
            total_count += 1
            article = Article(loc.text)
            article.download() 
            if "For subscribers" in article.html: 
                paywalled_articles += 1
                continue
            non_paywall_business_articles.append((loc.text, last_modified))
            print("Fetching article: ", loc.text)
            if total_count >= stop_limit:
                break

print("Total non-paywalled business articles fetched: ", len(non_paywall_business_articles))
print("Total paywalled articles skipped: ", paywalled_articles)


## Test summary 
article = Article(non_paywall_business_articles[0][0])
article.download()
article.parse()
article.nlp()
print("Title: ", article.title)
print("summary: ", article.summary)