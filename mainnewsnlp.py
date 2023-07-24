#Team 1:

import requests
from bs4 import BeautifulSoup as soup

corpus = []

# Scraping Indian Express URL
html = requests.get(sports_url1)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'native_story_title'})
corpus.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisement
article = bsobj.find('div', {'class': 'full-details'})
for paragraph in article.find_all('p'):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    corpus.append(paragraph.text.strip())

corpus

sports_url2 = "https://www.nydailynews.com/sports/more-sports/ny-novak-djokovic-roger-federer-grand-slam-wimbledon-20230711-eyeiq4rnazc5lg2hkd5ykoi46u-story.html"

corpus = []

# Scraping NY Daily News URL
html = requests.get(sports_url2)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'primary-font__PrimaryFontStyles-o56yd5-0 gVBMpi headline'})
corpus.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisement and text below the image
article = bsobj.find('article', {'class': 'default__ArticleBody-sc-1wxyvyl-2 hEvcgL article-body-wrapper-custom'})
for element in article.find_all(['p', 'figure']):
    # Skip internal links
    if element.find('a', href=True):
        continue

    # Check if the element contains the advertisement phrase and skip it
    if "Get the latest odds on all the top sports" in element.text:
        continue

    corpus.append(element.text.strip())


corpus

# Technical News
 tech_url1 = "https://indianexpress.com/article/technology/tech-news-technology/amazon-makes-first-big-tech-challenge-to-eu-online-content-rules-8829113/"
tech_url2 = "https://www.reuters.com/technology/amazon-challenges-eu-online-content-rules-says-unfairly-singled-out-2023-07-11/"

corpus = []

# Scraping Reuters URL
html = requests.get(tech_url2)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'text__text__1FZLe text__dark-grey__3Ml43 text__medium__1kbOh text__heading_3__1kDhc heading__base__2T28j heading__heading_3__3aL54 article-header__title__3Y2hh'})
corpus.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisements
article = bsobj.find('div', {'class': 'article-body__content__17Yit'})
for paragraph in article.find_all(['p']):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    corpus.append(paragraph.text.strip())

corpus


# Education News
education_url1 = "https://indianexpress.com/article/education/aiims-proposes-to-quash-interview-for-phd-selection-8829283/"

corpus = []

# Scraping Indian Express URL
html = requests.get(education_url)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'native_story_title'})
corpus.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisement
article = bsobj.find('div', {'class': 'full-details'})
for paragraph in article.find_all('p'):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    corpus.append(paragraph.text.strip())

corpus


education_url2 = "https://theprint.in/india/aiims-proposes-to-quash-interviews-in-phd-selection-process-for-greater-transparency/1665108/#google_vignette"

corpus = []

# Scraping Theprint URL
html = requests.get(education_url2)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'tdb-title-text'})
corpus.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisement
article = bsobj.find('div', {'class': 'td-post-content'})
for paragraph in article.find_all('p'):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    corpus.append(paragraph.text.strip())

corpus


# Politics News
politics_url1 = "https://indianexpress.com/article/political-pulse/sc-prepares-article-370-pleas-look-major-parties-stand-8829676/"
politics_url2 = "https://www.livemint.com/news/india/jammu-and-kashmir-sc-to-hear-batch-of-pleas-challenging-the-abrogation-of-article-370-from-august-2-11689053035450.html"

corpus = []

# Scraping LiveMint URL
html = requests.get(politics_url2)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'headline'})
corpus.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisement
article = bsobj.find('div', {'class': 'contentSec'})
for paragraph in article.find_all('p'):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    corpus.append(paragraph.text.strip())


corpus


# Global News
global_url1 = "https://indianexpress.com/article/explained/explained-global/swedens-rocky-road-from-neutrality-toward-nato-membership-8827291/"

corpus = []

# Scraping Indian Express URL
html = requests.get(global_url1)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'native_story_title'})
corpus.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisement
article = bsobj.find('div', {'class': 'full-details'})
for paragraph in article.find_all('p'):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    corpus.append(paragraph.text.strip())

corpus

global_url2 = "https://hindustannewshub.com/world-news/swedens-rocky-road-from-neutrality-toward-nato-membership/"

corpus = []

# Scraping Hindustan News URL
html = requests.get(global_url2)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'post-title entry-title'})
corpus.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisements
article = bsobj.find('div', {'class': 'entry-content entry clearfix'})
for paragraph in article.find_all(['p']):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    corpus.append(paragraph.text.strip())


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
