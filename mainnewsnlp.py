#Team 1:
import requests
from lxml import html

def corp_build(x):
  text=""
  text = " ".join(i for i in x)
  return text


# SPORTS NEWS 

# Scraping News from Hindu URL

sports_url1 = "https://www.thehindu.com/sport/cricket/asia-cup-2023-super-4-match-sri-lanka-vs-pakistan-in-colombo-on-september-14-2023/article67306703.ece"
sports1 = []

# Send an HTTP GET request to the URL
response = requests.get(sports_url1)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the webpage
    tree = html.fromstring(response.text)

    headline_xpath = '/html/body/section[2]/div/div/div[1]/h1'

    article_xpath = '/html/body/section[2]/div/div/div[1]/div[6]/p'

    # Use XPath to extract the headline and article content
    headline_elements = tree.xpath(headline_xpath)
    article_elements = tree.xpath(article_xpath)

    print("--------SPORTS NEWS-------")

    # Print the extracted headline
    for element in headline_elements:
        print("Headline:", element.text_content())

    # Print the word "Article" to indicate the start of the article
    print("Article:")

    # Append the extracted article content to the sports1 list
    for element in article_elements:
        sports1.append(element.text_content())

sports1 = corp_build(sports1)

# Print the concatenated article content
print(sports1)

# Scraping News from freepressjournal URL

sports_url2 = "https://www.freepressjournal.in/sports/asia-cup-2023-kusal-mendis-charith-asalanka-shine-as-sri-lanka-beat-pakistan-in-last-ball-thriller-to-set-up-final-vs-india"
sports2 = []

# Send an HTTP GET request to the URL
response = requests.get(sports_url2)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the webpage
    tree = html.fromstring(response.text)

    headline_xpath = '/html/body/main/section/div/div[1]/h1'

    article_xpath = '/html/body/main/section/div/div[2]/article/p'

    # Use XPath to extract the headline and article content
    headline_elements = tree.xpath(headline_xpath)
    article_elements = tree.xpath(article_xpath)

    # Print the extracted headline
    for element in headline_elements:
        print("Headline:", element.text_content())

    # Print the word "Article" to indicate the start of the article
    print("Article:")

    # Append the extracted article content to the sports2 list
    for element in article_elements:
        sports2.append(element.text_content())

sports2 = corp_build(sports2)

# Print the concatenated article content
print(sports2)

# TECHNICAL NEWS

# Scraping News from Hans India URL

tech_url1 = "https://www.thehansindia.com/technology/tech-news/elon-musk-finds-usb-type-c-charging-on-the-apple-iphone-15-amazing-822919"
tech1 = []

# Send an HTTP GET request to the URL
response = requests.get(tech_url1)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the webpage
    tree = html.fromstring(response.text)

    headline_xpath = '/html/body/div[7]/div[4]/div[1]/div[4]/div[1]/div/section[1]/div/div/div[1]/div[1]/div[2]/h1'

    article_xpath = '/html/body/div[7]/div[4]/div[1]/div[4]/div[1]/div/section[1]/div/div/div[1]/div[4]/div/div[4]/div[1]/div/p'

    # Use XPath to extract the headline and article content
    headline_elements = tree.xpath(headline_xpath)
    article_elements = tree.xpath(article_xpath)

    print("--------TECHNICAL NEWS--------")

    # Print the extracted headline
    for element in headline_elements:
        print("Headline:", element.text_content())

    # Print the word "Article" to indicate the start of the article
    print("Article:")

    # Append the extracted article content to the tech1 list
    for element in article_elements:
        tech1.append(element.text_content())

tech1 = corp_build(tech1)

# Print the concatenated article content
print(tech1)

# Scraping News from Investing.com 

tech_url2 = "https://www.investing.com/news/stock-market-news/apple-adopts-usb-typec-for-iphone-15-series-elon-musk-lauds-decision-93CH-3175191"
tech2 = []

# Send an HTTP GET request to the URL
response = requests.get(tech_url2)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the webpage
    tree = html.fromstring(response.text)

    headline_xpath = '/html/body/div[6]/section/h1'

    article_xpath = '/html/body/div[6]/section/div[5]/p'

    # Use XPath to extract the headline and article content
    headline_elements = tree.xpath(headline_xpath)
    article_elements = tree.xpath(article_xpath)

    # Print the extracted headline
    for element in headline_elements:
        print("Headline:", element.text_content())

    # Print the word "Article" to indicate the start of the article
    print("Article:")

    # Append the extracted article content to the tech2 list
    for element in article_elements:
        tech2.append(element.text_content())

tech2 = corp_build(tech2)

# Print the concatenated article content
print(tech2)

# BUSINESS NEWS

# Scraping News from India Today URL

business_url1 = "https://www.indiatoday.in/business/story/gucci-louis-vuitton-to-expand-in-india-with-new-outlets-in-reliances-luxury-mall-2436090-2023-09-15"
business1 = []

# Send an HTTP GET request to the URL
response = requests.get(business_url1)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the webpage
    tree = html.fromstring(response.text)

    headline_xpath = '/html/body/div[1]/div[3]/div/div/div[2]/main/div/div[1]/h1'

    article_xpath = '/html/body/div[1]/div[3]/div/div/div[2]/main/div/div[1]/div[6]/div[1]'

    # Use XPath to extract the headline and article content
    headline_elements = tree.xpath(headline_xpath)
    article_elements = tree.xpath(article_xpath)

    print("--------BUSINESS NEWS--------")

    # Print the extracted headline
    for element in headline_elements:
        print("Headline:", element.text_content())

    # Print the word "Article" to indicate the start of the article
    print("Article:")

    # Append the extracted article content to the business1 list
    for element in article_elements:
        business1.append(element.text_content())

business1 = corp_build(business1)

# Print the concatenated article content
print(business1)

# Scraping News from Reuters URL 

business_url2 = "https://www.reuters.com/business/retail-consumer/lvmh-gucci-expand-india-with-new-outlets-reliances-luxury-mall-2023-09-15/"

business2 = []

# Send an HTTP GET request to the URL
response = requests.get(business_url2)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the webpage
    tree = html.fromstring(response.text)

    headline_xpath = '/html/body/div[1]/div[3]/div/main/article/div[1]/div/header/div/div/h1'

    article_xpath = '/html/body/div[1]/div[3]/div/main/article/div[1]/div/div/div/div[2]/p'

    # Use XPath to extract the headline and article content
    headline_elements = tree.xpath(headline_xpath)
    article_elements = tree.xpath(article_xpath)

    # Print the extracted headline
    for element in headline_elements:
        print("Headline:", element.text_content())

    # Print the word "Article" to indicate the start of the article
    print("Article:")

    # Append the extracted article content to the business2 list
    for element in article_elements:
        business2.append(element.text_content())

business2 = corp_build(business2)

# Print the concatenated article content
print(business2)


# HEALTH NEWS

# Scraping News from Reuters URL

health_url1 = "https://www.reuters.com/business/healthcare-pharmaceuticals/highly-mutated-covid-variant-found-new-countries-pandemic-a-different-phase-2023-08-24/"

health1 = []

# Send an HTTP GET request to the URL
response = requests.get(health_url1)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the webpage
    tree = html.fromstring(response.text)

    headline_xpath = '/html/body/div[1]/div[3]/div/main/article/div[1]/div/header/div/div/h1'

    article_xpath = '/html/body/div[1]/div[3]/div/main/article/div[1]/div/div/div/div/p'

    # Use XPath to extract the headline and article content
    headline_elements = tree.xpath(headline_xpath)
    article_elements = tree.xpath(article_xpath)

    print("--------HEALTH NEWS--------")

    # Print the extracted headline
    for element in headline_elements:
        print("Headline:", element.text_content())

    # Print the word "Article" to indicate the start of the article
    print("Article:")

    # Append the extracted article content to the health1 list
    for element in article_elements:
        health1.append(element.text_content())

health1 = corp_build(health1)

# Print the concatenated article content
print(health1)

# Scraping News from Business Today URL

health_url2 = "https://www.businesstoday.in/latest/world/story/highly-mutated-covid-variant-found-in-new-countries-but-pandemic-in-a-different-phase-395557-2023-08-25"

health2 = []

# Send an HTTP GET request to the URL
response = requests.get(health_url2)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the webpage
    tree = html.fromstring(response.text)

    headline_xpath = '/html/body/div[10]/div/div[1]/div[2]/h1'

    article_xpath = '/html/body/div[10]/div/div[1]/div[9]/div/div/p'

    # Use XPath to extract the headline and article content
    headline_elements = tree.xpath(headline_xpath)
    article_elements = tree.xpath(article_xpath)

    # Print the extracted headline
    for element in headline_elements:
        print("Headline:", element.text_content())

    # Print the word "Article" to indicate the start of the article
    print("Article:")

    # Append the extracted article content to the health2 list
    for element in article_elements:
        health2.append(element.text_content())

health2 = corp_build(health2)

# Print the concatenated article content
print(health2)

# GLOBAL NEWS

# Scraping News from NEWS DRUM URL

global_url1 = "https://www.newsdrum.in/international/libya-seals-off-flooded-city-so-searchers-can-look-for-10000-missing-after-death-toll-passes-11000-1346077"

global1 = []

# Send an HTTP GET request to the URL
response = requests.get(global_url1)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the webpage
    tree = html.fromstring(response.text)

    headline_xpath = '/html/body/div[1]/main/div/div[2]/article/div/div[1]/div/div[2]/section[1]'

    article_xpath = '/html/body/div[1]/main/div/div[2]/article/div/div[1]/div/div[2]/section[2]/div[2]/div[1]/p'

    # Use XPath to extract the headline and article content
    headline_elements = tree.xpath(headline_xpath)
    article_elements = tree.xpath(article_xpath)

    print("--------GLOBAL NEWS--------")

    # Print the extracted headline
    for element in headline_elements:
        print("Headline:", element.text_content())

    # Print the word "Article" to indicate the start of the article
    print("Article:")

    # Append the extracted article content to the global1 list
    for element in article_elements:
        global1.append(element.text_content())

global1 = corp_build(global1)

# Print the concatenated article content
print(global1)

# Scraping News from HINDU URL

global_url2 = "https://www.thehindu.com/news/international/libya-seals-off-flooded-city-so-searchers-can-look-for-10000-missing-after-death-toll-passes-11000/article67311339.ece"

global2 = []

# Send an HTTP GET request to the URL
response = requests.get(global_url2)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the webpage
    tree = html.fromstring(response.text)

    headline_xpath = '/html/body/section[2]/div/div/div[1]/h1'

    article_xpath = '/html/body/section[2]/div/div/div[1]/div[6]/p'

    # Use XPath to extract the headline and article content
    headline_elements = tree.xpath(headline_xpath)
    article_elements = tree.xpath(article_xpath)

    # Print the extracted headline
    for element in headline_elements:
        print("Headline:", element.text_content())

    # Print the word "Article" to indicate the start of the article
    print("Article:")

    # Append the extracted article content to the global2 list
    for element in article_elements:
        global2.append(element.text_content())

global2 = corp_build(global2)

# Print the concatenated article content
print(global2)


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
