import feedparser, requests
from bs4 import BeautifulSoup
from newspaper import Article
# # RSS validity check
# feed = feedparser.parse("https://www.straitstimes.com/news/business/rss.xml")

# for entry in feed.entries: 
#     print(entry.title)
import nltk
nltk.download('punkt_tab')

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

article = Article(business_articles[0][0])

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

article.download() 
article.parse()
article.nlp()
# Create parser and summarizer
parser = PlaintextParser.from_string(article.text, Tokenizer("english"))
summarizer = LsaSummarizer()

# Generate summary (3 sentences)
summary = summarizer(parser.document, 3)
summary_text = ' '.join([str(sentence) for sentence in summary])
dummy_summary = article.summary
print("Better Summary:", summary_text)
print("Newspaper3k Summary:", dummy_summary)