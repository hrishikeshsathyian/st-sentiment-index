import feedparser, requests
from bs4 import BeautifulSoup

# # RSS validity check
# feed = feedparser.parse("https://www.straitstimes.com/news/business/rss.xml")

# for entry in feed.entries: 
#     print(entry.title)

# Parsing through XML file 

business_articles = []
count = 0
for month in range(10, 0, -1): 
    url = "https://www.straitstimes.com/sitemap/2025/{}/feeds.xml".format(month)
    soup = BeautifulSoup(requests.get(url).content, "xml")
    for loc in soup.find_all("loc"):
        if "/business/" in loc.text:
            last_modified = loc.find_next("lastmod")
            if not last_modified: 
                last_modified = "N/A"
                print("No last modified date found for article: ", loc.text)
                break
            else: 
                last_modified = last_modified.text
            business_articles.append((loc.text, last_modified))
            count += 1
            print("Fetched {} business articles".format(count))

for article in business_articles:
    print(article)

print("Total business articles fetched: ", len(business_articles))