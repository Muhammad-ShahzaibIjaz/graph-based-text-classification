import requests
from bs4 import BeautifulSoup
import json

def scrape_article(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        paragraphs = soup.find_all('p')
        
        article_text = "\n".join([p.get_text() for p in paragraphs])
        
        return article_text
    else:
        print("Error: Unable to fetch the URL.")

def save(data):
    with open("scraped_articles.json", "a", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)



urls = ["https://fashionmagazine.com/style/celebrity-style/julia-fox-fashion-april-cover-star/",
        "https://fashionmagazine.com/beauty-grooming/hair/dyson-supersonic-nural/",
        "https://fashionmagazine.com/style/challengers-movie-fashion/",
        "https://fashionmagazine.com/flare/celebrity/kaia-gerber-acting-palm-royale/",
        "https://fashionmagazine.com/style/celebrity-style/met-gala-2024-sleeping-beauties/",
        "https://fashionmagazine.com/beauty-grooming/texture-talk/code-my-crown-hair-representation/",
        "https://fashionmagazine.com/style/celebrity-style/anne-hathaway-met-gala-looks/",
        "https://fashionmagazine.com/flare/tv-movies/anne-hathaway-the-idea-of-you-movie/",
        "https://fashionmagazine.com/style/shopping/best-sneakers-for-women/",
        "https://fashionmagazine.com/style/shopping/eco-friendly-denim/",
        "https://fashionmagazine.com/beauty-grooming/celebrity-beauty/met-gala-beauty-best-looks/",
        "https://fashionmagazine.com/beauty-grooming/aveda-be-curly-advanced/",
        "https://fashionmagazine.com/style/ysl-canada-toronto-store/",
        "https://fashionmagazine.com/beauty-grooming/best-new-sunscreens/",
        "https://fashionmagazine.com/style/scandi-mens-style-outfits/",
        "https://fashionmagazine.com/style/booty-shorts/",
        "https://fashionmagazine.com/style/nicola-coughlan-style/"
]

articles = []
for url in urls:
    article_text = scrape_article(url)
    if article_text:
        articles.append({"category": "fashion", "Article": article_text})


save(articles)
