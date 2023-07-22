# Team 1: Web Scraping
# TODO: Import necessary libraries for web scraping

# Task 1: Identify a suitable website for web scraping
# Sports news URLs
sports_url1 = "https://indianexpress.com/article/sports/tennis/novak-djokovic-ties-roger-federer-wimbledon-8828916/"
sports_url2 = "https://mcdowellnews.com/sports/professional/djokovic-ties-federer-with-46-slam-semifinals-meets-sinner-next/article_af9d586a-2030-11ee-82e4-0b2a92280f49.html"

# Technology news URLs
tech_url1 = "https://indianexpress.com/article/technology/tech-news-technology/amazon-makes-first-big-tech-challenge-to-eu-online-content-rules-8829113/"
tech_url2 = "https://economictimes.indiatimes.com/tech/technology/amazon-challenges-eu-online-content-rules-says-unfairly-singled-out/articleshow/101668958.cms"

# Education news URLs
edu_url1 = "https://indianexpress.com/article/education/aiims-proposes-to-quash-interview-for-phd-selection-8829283/"
edu_url2 = "https://theprint.in/india/aiims-proposes-to-quash-interviews-in-phd-selection-process-for-greater-transparency/1665108/#google_vignette"

# Political news URLs
politics_url1 = "https://indianexpress.com/article/political-pulse/sc-prepares-article-370-pleas-look-major-parties-stand-8829676/"
politics_url2 = "https://economictimes.indiatimes.com/news/india/sc-to-hear-pleas-challenging-article-370-abrogation-from-august-2/articleshow/101658010.cms"

# Global news URLs
url1 = "https://indianexpress.com/article/explained/explained-global/swedens-rocky-road-from-neutrality-toward-nato-membership-8827291/"
url2 = "https://www.theweek.in/wire-updates/international/2023/07/11/fgn19-sweden-nato-explainer.html"

corpus=[]
output=[]

# Task 2: Research and select appropriate web scraping tools and libraries
# TODO: Import web scraping libraries (e.g., BeautifulSoup, Scrapy)
import requests
from bs4 import BeautifulSoup as soup

# Task 3: Develop a web scraping script
# TODO: Write a function to scrape data from the chosen website
def scrape_website(url, headline_class, article_class):
    html = requests.get(url)
    bsobj = soup(html.content, 'lxml')

    for headline in bsobj.findAll('h1', {'class': headline_class}): 
      corpus.append("Headline : {}".format(headline.text))
      for article in bsobj.findAll('div', {'class': article_class}):
        corpus.append("Article : {}".format(article.text.strip()))

# Task 4: Handle authentication or access restrictions
# TODO: If required, handle authentication or access restrictions here
import requests

# Task 5: Test and validate the web scraping script
# TODO: Test the web scraping function and validate the extracted data
# Sports news URLs
sports_url1 = "https://indianexpress.com/article/sports/tennis/novak-djokovic-ties-roger-federer-wimbledon-8828916/"
sports_url2 = "https://mcdowellnews.com/sports/professional/djokovic-ties-federer-with-46-slam-semifinals-meets-sinner-next/article_af9d586a-2030-11ee-82e4-0b2a92280f49.html"

# Scrape sports news
scrape_website(sports_url1, 'native_story_title', 'story_details')
scrape_website(sports_url2, 'headline', 'lee-article-text')

# Technology news URLs
tech_url1 = "https://indianexpress.com/article/technology/tech-news-technology/amazon-makes-first-big-tech-challenge-to-eu-online-content-rules-8829113/"
tech_url2 = "https://economictimes.indiatimes.com/tech/technology/amazon-challenges-eu-online-content-rules-says-unfairly-singled-out/articleshow/101668958.cms"

# Scrape technology news
scrape_website(tech_url1, 'native_story_title', 'story_details')
scrape_website(tech_url2, 'artTitle font_faus', 'article_wrap')

# Education news URLs
edu_url1 = "https://indianexpress.com/article/education/aiims-proposes-to-quash-interview-for-phd-selection-8829283/"
edu_url2 = "https://theprint.in/india/aiims-proposes-to-quash-interviews-in-phd-selection-process-for-greater-transparency/1665108/#google_vignette"

# Scrape education news
scrape_website(edu_url1, 'native_story_title', 'story_details')
scrape_website(edu_url2, 'tdb-title-text', 'tdb-block-inner td-fix-index')

# Political news URLs
politics_url1 = "https://indianexpress.com/article/political-pulse/sc-prepares-article-370-pleas-look-major-parties-stand-8829676/"
politics_url2 = "https://economictimes.indiatimes.com/news/india/sc-to-hear-pleas-challenging-article-370-abrogation-from-august-2/articleshow/101658010.cms"

# Scrape political news
scrape_website(politics_url1, 'native_story_title', 'story_details')
scrape_website(politics_url2, 'artTitle font_faus', 'pageContent flt')

#Global news URLs
url1 = "https://indianexpress.com/article/explained/explained-global/swedens-rocky-road-from-neutrality-toward-nato-membership-8827291/"
url2 = "https://www.theweek.in/wire-updates/international/2023/07/11/fgn19-sweden-nato-explainer.html"

scrape_website(url1, 'article-title', 'story_details')
scrape_website(url2, 'article', 'pageContent flt')

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
