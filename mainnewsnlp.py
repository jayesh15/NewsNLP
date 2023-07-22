# Team 1: Web Scraping
# TODO: Import necessary libraries for web scraping

import requests
from bs4 import BeautifulSoup as soup

# Task 1: Identify a suitable website for web scraping
# Sports news URLs
# Sports news URLs
sports_url1 = "https://indianexpress.com/article/sports/tennis/novak-djokovic-ties-roger-federer-wimbledon-8828916/"
sports_url2 = "https://www.nydailynews.com/sports/more-sports/ny-novak-djokovic-roger-federer-grand-slam-wimbledon-20230711-eyeiq4rnazc5lg2hkd5ykoi46u-story.html"

# Technology news URLs
tech_url1 = "https://indianexpress.com/article/technology/tech-news-technology/amazon-makes-first-big-tech-challenge-to-eu-online-content-rules-8829113/"
tech_url2 = "https://www.reuters.com/technology/amazon-challenges-eu-online-content-rules-says-unfairly-singled-out-2023-07-11/"

# Education news URLs
edu_url1 = "https://indianexpress.com/article/education/aiims-proposes-to-quash-interview-for-phd-selection-8829283/"
edu_url2 = "https://theprint.in/india/aiims-proposes-to-quash-interviews-in-phd-selection-process-for-greater-transparency/1665108/#google_vignette"

# Political news URLs
politics_url1 = "https://indianexpress.com/article/political-pulse/sc-prepares-article-370-pleas-look-major-parties-stand-8829676/"
politics_url2 = "https://www.livemint.com/news/india/jammu-and-kashmir-sc-to-hear-batch-of-pleas-challenging-the-abrogation-of-article-370-from-august-2-11689053035450.html"

# Global news URLs
global_url1 = "https://indianexpress.com/article/explained/explained-global/swedens-rocky-road-from-neutrality-toward-nato-membership-8827291/"
global_url2 = "https://www.theweek.in/wire-updates/international/2023/07/11/fgn19-sweden-nato-explainer.html"


corpus=[]
output=[]

# Task 2: Research and select appropriate web scraping tools and libraries
# TODO: Import web scraping libraries (e.g., BeautifulSoup, Scrapy)
import requests
from bs4 import BeautifulSoup as soup

# Task 3: Develop a web scraping script
# TODO: Write a function to scrape data from the chosen website
corpus = []

def scrape_website(url, headline_class, article_class, ads):
    html = requests.get(url)
    bsobj = soup(html.content, 'lxml')

    headline = bsobj.find('h1', {'class': headline_class})
    if headline:
        corpus.append("Headline : {}".format(headline.text.strip()))

    for article in bsobj.findAll('article', {'class': article_class}):
        corpus.append(article.text.strip())

    article_content = bsobj.find('div', {'class': article_class})
    if article_content:
        for paragraph in article_content.find_all('p'):
            if ads in paragraph.text:
                continue
            corpus.append(paragraph.text.strip())


# Task 4: Handle authentication or access restrictions
# TODO: If required, handle authentication or access restrictions here
import requests

# Task 5: Test and validate the web scraping script
# TODO: Test the web scraping function and validate the extracted data
# Sports news URLs
sports_url1 = "https://indianexpress.com/article/sports/tennis/novak-djokovic-ties-roger-federer-wimbledon-8828916/"
sports_url2 = "https://www.nydailynews.com/sports/more-sports/ny-novak-djokovic-roger-federer-grand-slam-wimbledon-20230711-eyeiq4rnazc5lg2hkd5ykoi46u-story.html"

# Technology news URLs
tech_url1 = "https://indianexpress.com/article/technology/tech-news-technology/amazon-makes-first-big-tech-challenge-to-eu-online-content-rules-8829113/"
tech_url2 = "https://www.reuters.com/technology/amazon-challenges-eu-online-content-rules-says-unfairly-singled-out-2023-07-11/"

# Education news URLs
edu_url1 = "https://indianexpress.com/article/education/aiims-proposes-to-quash-interview-for-phd-selection-8829283/"
edu_url2 = "https://theprint.in/india/aiims-proposes-to-quash-interviews-in-phd-selection-process-for-greater-transparency/1665108/#google_vignette"

# Political news URLs
politics_url1 = "https://indianexpress.com/article/political-pulse/sc-prepares-article-370-pleas-look-major-parties-stand-8829676/"
politics_url2 = "https://www.livemint.com/news/india/jammu-and-kashmir-sc-to-hear-batch-of-pleas-challenging-the-abrogation-of-article-370-from-august-2-11689053035450.html"

# Global news URLs
global_url1 = "https://indianexpress.com/article/explained/explained-global/swedens-rocky-road-from-neutrality-toward-nato-membership-8827291/"
global_url2 = "https://www.theweek.in/wire-updates/international/2023/07/11/fgn19-sweden-nato-explainer.html"

# Scrape and print sports news
print("----- Sports News -----")
scrape_website(sports_url1, 'native_story_title', 'full-details', "advertisement")
scrape_website(sports_url2, 'primary-font__PrimaryFontStyles-o56yd5-0 gVBMpi headline', 'default__ArticleBody-sc-1wxyvyl-2 hEvcgL article-body-wrapper-custom', "Advertisement")

# Scrape and print technology news
print("----- Technology News -----")
scrape_website(tech_url1, 'native_story_title', 'full-details', "Advertisement")
scrape_website(tech_url2, 'text__text__1FZLe text__dark-grey__3Ml43 text__medium__1kbOh text__heading_3__1kDhc heading__base__2T28j heading__heading_3__3aL54 article-header__title__3Y2hh', 'article-body__content__17Yit', "Advertisement")

# Scrape and print education news
print("----- Education News -----")
scrape_website(edu_url1, 'native_story_title', 'full-details', "Advertisement")
scrape_website(edu_url2, 'tdb-title-text', 'td-post-content', "Advertisement")

# Scrape and print political news
print("----- Political News -----")
scrape_website(politics_url1, 'native_story_title', 'full-details', "Advertisement")
scrape_website(politics_url2, 'headline', 'contentSec', "Advertisement")

# Scrape and print global news
print("----- Global News -----")
scrape_website(global_url1, 'native_story_title', 'full-details', "Advertisement")
scrape_website(global_url2, 'article-title', 'article', "")

print(corpus)

# Task 6: Document the web scraping process
# TODO: Write a detailed documentation of the web scraping process and challenges faced


# Team 2: NLP Preprocessing
# TODO: Import necessary libraries for NLP preprocessing

# Task 1: Implement data preprocessing pipeline
# TODO: Write functions to perform each preprocessing step (lowercasing, punctuation removal, etc.)

# Task 2: Apply preprocessing pipeline to scraped data
# TODO: Apply the preprocessing functions to the scraped data obtained by Team 1

# Task 3: Validate the effectiveness of preprocessing steps
# TODO: Analyze sample data before and after preprocessing to validate the effectiveness

# Task 4: Document the preprocessing pipeline
# TODO: Write a documentation explaining each step of the preprocessing pipeline and its purpose

# Task 5: Explore different preprocessing techniques
# TODO: Experiment with different preprocessing techniques and evaluate their impact on NLP tasks


# Team 3: NLP Morphological Analysis and Tokenization
# TODO: Import necessary libraries for morphological analysis and tokenization

# Task 1: Implement word tokenization techniques
# TODO: Write functions to perform different word tokenization techniques (whitespace, regex, etc.)

# Task 2: Apply sentence tokenization techniques
# TODO: Write a function to apply sentence tokenization to segment text into sentences

# Task 3: Experiment with morphological analysis methods
# TODO: Implement and evaluate different morphological analysis methods (stemming, lemmatization, etc.)

# Task 4: Evaluate the performance of tokenization and morphological analysis techniques
# TODO: Compare and document the performance and effectiveness of the implemented techniques


# Team 4: NLP Part of Speech Tagging and WordNet Analysis
# TODO: Import necessary libraries for part of speech tagging and WordNet analysis

# Task 1: Implement part of speech tagging
# TODO: Write a function to perform part of speech tagging using NLTK, Spacy, or other libraries

# Task 2: Apply POS tagging to preprocessed data
# TODO: Apply the POS tagging function to the preprocessed data

# Task 3: Perform WordNet analysis
# TODO: Utilize WordNet or similar resources for semantic analysis, synonym identification, etc.

# Task 4: Explore additional NLP libraries for POS tagging and WordNet analysis
# TODO: Experiment with additional libraries (e.g., TextBlob, Pattern) and compare the results

# Task 5: Document the findings of POS tagging and WordNet analysis
# TODO: Write a documentation summarizing the accuracy of POS tagging and the usefulness of WordNet


# Team 5: NLP Data Visualization and Analysis
# TODO: Import necessary libraries for data visualization and analysis

# Task 1: Generate word clouds
# TODO: Create word clouds based on the preprocessed data

# Task 2: Create frequency distributions
# TODO: Generate frequency distributions of words to gain insights

# Task 3: Explore visualization techniques
# TODO: Experiment with different visualization techniques (bar charts, scatter plots, heatmaps)

# Task 4: Evaluate effectiveness of visualization techniques
# TODO: Assess the effectiveness of different visualization techniques in conveying information

# Task 5: Prepare a comprehensive report
# TODO: Write a report summarizing the findings, insights, and visualizations generated

# Main function to orchestrate the project tasks
def main():
    # TODO: Call the functions for each team's tasks
    pass

if __name__ == "__main__":
    main()
