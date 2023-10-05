#Team 1:
import requests
import nltk
import string
import emoji
import re
import sentencepiece as spm
import spacy
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,TweetTokenizer, TreebankWordTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from lxml import html
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import ByteLevelBPETokenizer, SentencePieceBPETokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet
from textblob import Word
from textblob import TextBlob
from nltk.tag import pos_tag
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from nltk.probability import FreqDist

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')



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

tech_url2 = "https://www.indiatoday.in/technology/news/story/elon-musk-reacts-to-usb-type-c-charging-in-apple-iphone-15-finds-it-amazing-2435701-2023-09-14"
tech2 = []

# Send an HTTP GET request to the URL
response = requests.get(tech_url2)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the webpage
    tree = html.fromstring(response.text)

    headline_xpath = '/html/body/div[1]/div[3]/div/div/div[2]/main/div/div[1]/h1'


    article_xpath ='/html/body/div[1]/div[3]/div/div/div[2]/main/div/div[1]/div[7]/div[1]/p'

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

# Scraping News from NEWS DRUM URL

global_url1 = "https://www.newsdrum.in/international/libya-seals-off-flooded-city-so-searchers-can-look-for-10000-missing-after-death-toll-passes-11000-1346077"

global1 = []

# Send an HTTP GET request to the URL
response = requests.get(global_url1)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the webpage
    tree = html.fromstring(response.text)

    headline_xpath = '/html/body/div[2]/main/div/div[2]/article/div/div/div/div[2]/section[1]/h1'

    article_xpath = '/html/body/div[2]/main/div/div[2]/article/div/div[1]/div/div[2]/section[2]/div[2]/div[1]/p[1]'

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
# Building corpora of article of similar genre
sports_corp=[sports1,sports2]
business_corp = [business1,business2]
health_corp = [health1,health2]
global_corp = [global1,global2]
tech_corp = [tech1,tech2]

# Preprocessing pipeline
def text_preprocessing(corp):
  processed_corp = []
  for text in corp:
    # Converts the text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("","",string.punctuation))

    #Removal of punctuation present with different ASCII Code left out of string.punctuation
    punc = ['“', "”", "’", "…", "‘", "—",'私たちの行動規範：トムソン・ロイター「信頼の原則」','\ufeff']
    for i in punc:
      text = text.replace(i,"")

    # Remove numbers
    text = "".join([i for i in text if not i.isdigit()])

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    token = word_tokenize(text)
    text = ' '.join([word for word in token if word not in stop_words])

    # Strip extra whitespaces
    text = ' '.join(text.split())

    # Emoji Removing
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", '', text)  # Remove URLs
    text = " ".join(text.split())
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'RT[\s]+', '', text)  # Remove RT

    # Remove emojis
    text = emoji.demojize(text)
    text = re.sub(r':[a-zA-Z_]+:', '', text)

    processed_corp.append(text)
  return processed_corp

# Processed corpus from similar genre
sports = text_preprocessing(sports_corp)
business = text_preprocessing(business_corp)
health = text_preprocessing(health_corp)
globe = text_preprocessing(global_corp)
technology = text_preprocessing(tech_corp)

#Validation of preprocessing function
sc= []
x = [sports,health,technology,globe,business]
for k in x:
  for j in range(len(k)):
    for i in k[j]:
      if i != ' ' and not i.isalpha():
        sc.append(i)
      else:
        pass
  if len(sc) >0:
    print(sc)
  else:
    print(f"No special characters found in coprus")
    print('--'*40)


# Team 3: NLP Morphological Analysis and Tokenization
# TODO: Import necessary libraries for morphological analysis and tokenization
#task 1
# Task 1: Implement word tokenization techniques
# TODO: Write functions to perform different word tokenization techniques (whitespace, regex, etc.)



# whitespace tokenization function
def whitespace_tokenize_corpus(corpus):
    return [sentence.split() for sentence in corpus]

# WordPunct tokenization function
def wordpunct_tokenize_corpus(corpus):
    return [nltk.wordpunct_tokenize(sentence) for sentence in corpus]

#  Penn Treebank tokenization function
def treebank_tokenize_corpus(corpus):
    treebank_tokenizer = TreebankWordTokenizer()
    return [treebank_tokenizer.tokenize(sentence) for sentence in corpus]

#  Tweet tokenization function
def tweet_tokenize_corpus(corpus):
    tweet_tokenizer = TweetTokenizer()
    return [tweet_tokenizer.tokenize(sentence) for sentence in corpus]

#whitespace subword tokenization function
def subword_tokenize_corpus(corpus):
    return [sentence.split() for sentence in corpus]

# corpora
business_corpus = business

technology_corpus = technology

globe_corpus = globe

health_corpus = health

sports_corpus = sports
def flatten_list(list_of_lists):
    return [token for sublist in list_of_lists for token in sublist]

#applying the tokenize techniques to corpora
whitespace_tokenized_business = flatten_list(whitespace_tokenize_corpus(business_corpus))
wordpunct_tokenized_business = flatten_list(wordpunct_tokenize_corpus(business_corpus))
treebank_tokenized_business= flatten_list(treebank_tokenize_corpus(business_corpus))
tweet_tokenized_business = flatten_list(tweet_tokenize_corpus(business_corpus))
subword_tokenized_business = flatten_list(subword_tokenize_corpus(business_corpus))

whitespace_tokenized_technology = flatten_list(whitespace_tokenize_corpus(technology_corpus))
wordpunct_tokenized_technology = flatten_list(wordpunct_tokenize_corpus(technology_corpus))
treebank_tokenized_technology = flatten_list(treebank_tokenize_corpus(technology_corpus))
tweet_tokenized_technology = flatten_list(tweet_tokenize_corpus(technology_corpus))
subword_tokenized_technology = flatten_list(subword_tokenize_corpus(technology_corpus))

whitespace_tokenized_globe = flatten_list(whitespace_tokenize_corpus(globe_corpus))
wordpunct_tokenized_globe = flatten_list(wordpunct_tokenize_corpus(globe_corpus))
treebank_tokenized_globe = flatten_list(treebank_tokenize_corpus(globe_corpus))
tweet_tokenized_globe = flatten_list(tweet_tokenize_corpus(globe_corpus))
subword_tokenized_globe = flatten_list(subword_tokenize_corpus(globe_corpus))


whitespace_tokenized_sports = flatten_list(whitespace_tokenize_corpus(sports_corpus))
wordpunct_tokenized_sports = flatten_list(wordpunct_tokenize_corpus(sports_corpus))
treebank_tokenized_sports = flatten_list(treebank_tokenize_corpus(sports_corpus))
tweet_tokenized_sports = flatten_list(tweet_tokenize_corpus(sports_corpus))
subword_tokenized_sports = flatten_list(subword_tokenize_corpus(sports_corpus))


whitespace_tokenized_health =flatten_list( whitespace_tokenize_corpus(health_corpus))
wordpunct_tokenized_health = flatten_list(wordpunct_tokenize_corpus(health_corpus))
treebank_tokenized_health  =flatten_list( treebank_tokenize_corpus(health_corpus))
tweet_tokenized_health  = flatten_list(tweet_tokenize_corpus(health_corpus))
subword_tokenized_health  = flatten_list(subword_tokenize_corpus(health_corpus))

# Print results for corpus
print("Whitespace Tokenization (Politics):", whitespace_tokenized_business)
print("WordPunct Tokenization (Politics):", wordpunct_tokenized_business)
print("Penn Treebank Tokenization (Politics):", treebank_tokenized_business)
print("Tweet Tokenization (Politics):", tweet_tokenized_business)
print("Whitespace Subword Tokenization (Politics):", subword_tokenized_business)

# Print results for technology corpus
print("Whitespace Tokenization (Technology):", whitespace_tokenized_technology)
print("WordPunct Tokenization (Technology):", wordpunct_tokenized_technology)
print("Penn Treebank Tokenization (Technology):", treebank_tokenized_technology)
print("Tweet Tokenization (Technology):", tweet_tokenized_technology)
print("Whitespace Subword Tokenization (Technology):", subword_tokenized_technology)

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


# Define a function for applying morphological analysis (stemming and lemmatization)
def apply_morphological_analysis(corpus):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    stemmed_corpus = []
    lemmatized_corpus = []

    for text in corpus:
        sentences = sent_tokenize(text)  # Sentence tokenization

        # Initialize lists for stemmed and lemmatized sentences
        stemmed_sentences = []
        lemmatized_sentences = []

        for sentence in sentences:
            words = word_tokenize(sentence)  # Word tokenization
            stemmed_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]

            stemmed_sentences.append(" ".join(stemmed_words))
            lemmatized_sentences.append(" ".join(lemmatized_words))

        stemmed_corpus.append(stemmed_sentences)
        lemmatized_corpus.append(lemmatized_sentences)

    return stemmed_corpus, lemmatized_corpus



corpora = {
    "technology": technology,
    "sports": sports,
    "business": business,
    "health": health,
    "globe": globe
}


stemmed_corpora = {}
lemmatized_corpora = {}

# Applying morphological analysis to each corpus
for corpus_name, corpus in corpora.items():
    stemmed_corpora[corpus_name], lemmatized_corpora[corpus_name] = apply_morphological_analysis(corpus)

# Printing the results for each corpus and each morphological analysis techniquue
for corpus_name in corpora.keys():
    print(f"Corpus: {corpus_name.capitalize()}")

    for i, text in enumerate(stemmed_corpora[corpus_name]):
        for j, sentence in enumerate(text):
            print(f"Stemmed Text {i + 1}, Sentence {i + 1}: {sentence}")

    for i, text in enumerate(lemmatized_corpora[corpus_name]):
        for j, sentence in enumerate(text):
            print(f"Lemmatized Text {i + 1}, Sentence {i + 1}: {sentence}")
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Function for stemming and lemmatization
def apply_morphological_analysis(corpus):
    porter_stemmer = PorterStemmer()
    snowball_stemmer = SnowballStemmer("english")
    lancaster_stemmer = LancasterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    stemmed_corpus_porter = []
    stemmed_corpus_snowball = []
    stemmed_corpus_lancaster = []
    lemmatized_corpus = []

    for text in corpus:
        words = nltk.word_tokenize(text)

        # Stemming using different stemmers
        stemmed_words_porter = [porter_stemmer.stem(word) for word in words if word.lower() not in stop_words]
        stemmed_words_snowball = [snowball_stemmer.stem(word) for word in words if word.lower() not in stop_words]
        stemmed_words_lancaster = [lancaster_stemmer.stem(word) for word in words if word.lower() not in stop_words]

        # Lemmatization
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]

        stemmed_corpus_porter.append(" ".join(stemmed_words_porter))
        stemmed_corpus_snowball.append(" ".join(stemmed_words_snowball))
        stemmed_corpus_lancaster.append(" ".join(stemmed_words_lancaster))
        lemmatized_corpus.append(" ".join(lemmatized_words))

    return stemmed_corpus_porter, stemmed_corpus_snowball, stemmed_corpus_lancaster, lemmatized_corpus


# Applying morphological analysis to each corpus
stemmed_corpora_porter = {}
stemmed_corpora_snowball = {}
stemmed_corpora_lancaster = {}
lemmatized_corpora = {}

for corpus_name, corpus in corpora.items():
    (stemmed_corpora_porter[corpus_name],
     stemmed_corpora_snowball[corpus_name],
     stemmed_corpora_lancaster[corpus_name],
     lemmatized_corpora[corpus_name]) = apply_morphological_analysis(corpus)

# Printing the results for each corpus
for corpus_name in corpora.keys():
    print(f"Stemmed Corpus (Porter) for {corpus_name.capitalize()}:")
    for i, text in enumerate(stemmed_corpora_porter[corpus_name]):
        print(f"Text {i + 1}: {text}")

    print(f"\nStemmed Corpus (Snowball) for {corpus_name.capitalize()}:")
    for i, text in enumerate(stemmed_corpora_snowball[corpus_name]):
        print(f"Text {i + 1}: {text}")

    print(f"\nStemmed Corpus (Lancaster) for {corpus_name.capitalize()}:")
    for i, text in enumerate(stemmed_corpora_lancaster[corpus_name]):
        print(f"Text {i + 1}: {text}")

    print(f"\nLemmatized Corpus for {corpus_name.capitalize()}:")
    for i, text in enumerate(lemmatized_corpora[corpus_name]):
        print(f"Text {i + 1}: {text}")
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords



# Function for applying morphological analysis (stemming and lemmatization)
def apply_morphological_analysis(corpus, technique):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    stemmed_corpus = []
    lemmatized_corpus = []

    for text in corpus:
        words = nltk.word_tokenize(text)

        if technique == "stemming":
            # Stemming
            processed_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
        elif technique == "lemmatization":
            # Lemmatization
            processed_words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
        else:
            raise ValueError("Invalid technique. Use 'stemming' or 'lemmatization'.")

        processed_text = " ".join(processed_words)

        if technique == "stemming":
            stemmed_corpus.append(processed_text)
        elif technique == "lemmatization":
            lemmatized_corpus.append(processed_text)

    return stemmed_corpus, lemmatized_corpus

#corpora
corpora = {
    "technology": technology,
    "sports": sports,
    "business": business,
    "health": health,
    "globe": globe
}

techniques = ["stemming", "lemmatization"]

results = {}

for technique in techniques:
    technique_results = {}
    
    for corpus_name, corpus in corpora.items():
        technique_results[corpus_name] = apply_morphological_analysis(corpus, technique)

    results[technique] = technique_results

# Print the results for each technique and each corpus
for technique, technique_results in results.items():
    print(f"Technique: {technique.capitalize()}")
    
    for corpus_name, (stemmed_results, lemmatized_results) in technique_results.items():
        print(f"Corpus: {corpus_name.capitalize()}")
        
        for i, text in enumerate(stemmed_results):
            print(f"Stemmed Text {i + 1}, Sentence {i + 1}: {text}")

        for i, text in enumerate(lemmatized_results):
            print(f"Lemmatized Text {i + 1}, Sentence {i + 1}: {text}")
# Dedine corpora
corpus_business = business
corpus_technology = technology
corpus_globe = globe
corpus_health =health
corpus_sports = sports

#function to create a count matrix using Count Vectorizer
def create_count_matrix(documents):
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(documents)
    return count_matrix

#  count matrix for each corpus
count_matrix_politics = create_count_matrix(corpus_business)
count_matrix_technology = create_count_matrix(corpus_technology)
count_matrix_sports = create_count_matrix(corpus_sports)
count_matrix_globe = create_count_matrix(corpus_globe)
count_matrix_education = create_count_matrix(corpus_health)

# count matrices
print("Count Matrix (business):")
print(count_matrix_politics.toarray())

print("Count Matrix (Technology):")
print(count_matrix_technology.toarray())

print("Count Matrix (sports):")
print(count_matrix_sports.toarray())

print("Count Matrix (health):")
print(count_matrix_education.toarray())

print("Count Matrix (globe):")
print(count_matrix_globe.toarray())




# Team 4: NLP Part of Speech Tagging and WordNet Analysis
# TODO: Import necessary libraries for part of speech tagging and WordNet analysis

nlp_spacy = spacy.load('en_core_web_sm')

# Task 1: Implement part of speech tagging
# TODO: Write a function to perform part of speech tagging using NLTK, Spacy, or other libraries

def pos_tagging(texts):
    nltk_tags_list = []
    spacy_tags_list = []

    for text in texts:
        # NLTK
        words_nltk = nltk.word_tokenize(text)
        pos_tags_nltk = nltk.pos_tag(words_nltk)

        # spaCy
        doc_spacy = nlp_spacy(text)
        pos_tags_spacy = [(token.text, token.pos_) for token in doc_spacy]

        nltk_tags_list.append(pos_tags_nltk)
        spacy_tags_list.append(pos_tags_spacy)

    return nltk_tags_list, spacy_tags_list


# Task 2: Apply POS tagging to preprocessed data
# TODO: Apply the POS tagging function to the preprocessed data

for category, text in corpora.items():
    nltk_tag,spacy_tag=pos_tagging(text)
    print(f'NLTK Tag for {category}',nltk_tag)
    print(f'Spacy Tag for {category}',spacy_tag)
    print()  # Separating line between categories

# Task 3: Perform WordNet analysis
# TODO: Utilize WordNet or similar resources for semantic analysis, synonym identification, etc.
topics = lemmatized_corpora.keys()

set_of_nltk_synonyms,set_of_tb_synonyms,set_of_nltk_hypernyms,set_of_tb_hypernyms={},{},{},{}
for topic in topics:
    sentences = lemmatized_corpora[topic]
    for sentence in sentences:
        for word in sentence.split():
            try:
                #Getting the synonymns of the words
                set_of_nltk_synonyms[word+'_synonyms']=set([syn.name()[:-5] for syn in wordnet.synsets(word)]) # Using the nltk library
                set_of_tb_synonyms[word+'_synonyms'] = set([syn.name()[:-5] for syn in Word(word).get_synsets()]) # Using the textblob library
                #Getting the hypernymns
                set_of_nltk_hypernyms[word+'_hypernyms']= [[n.name()[:-5] for n in hyn.hypernyms()][0] for hyn in wordnet.synsets(word)] #Using the nltk library
                set_of_tb_hypernyms[word+'_hypernyms'] = [[n.name()[:-5] for n in hyn.hypernyms()][0] for hyn in Word(word).get_synsets()] #Using the textblob library
            except Exception as e:
                continue
print(set_of_nltk_synonyms)
print(set_of_tb_synonyms)
print(set_of_nltk_hypernyms)
print(set_of_tb_hypernyms)

# Task 4: Explore additional NLP libraries for POS tagging and WordNet analysis
# TODO: Experiment with additional libraries (e.g., TextBlob, Pattern) and compare the results


def pos_exp_tagging(texts):
    pos_tags_list = []

    for text in texts:
        # TextBlob
        blob = TextBlob(text)
        pos_tags_textblob = blob.tags
        pos_tags_list.append(pos_tags_textblob)

    return pos_tags_list

for category, text in corpora.items():
    sports_blob_tag=pos_exp_tagging(text)
    print(f'TextBlob Tag for {category}\n',sports_blob_tag)


#Initializing count vectorizer
vectorizer=CountVectorizer()

#Initializing tfidf vectorizer
tfidf=TfidfVectorizer()
#TDM, DTM, TF-IDF

DTM_sports=vectorizer.fit_transform(lemmatized_sports)
TDM_sports=DTM_sports.T
TFIDF_sports=tfidf.fit_transform(lemmatized_sports)
print('TDM for Sports\n',TDM_sports)
print('DTM for Sports\n',DTM_sports)
print('TF-IDF for Sports\n',TFIDF_sports)

DTM_health=vectorizer.fit_transform(lemmatized_health)
TDM_health=DTM_health.T
TFIDF_health=tfidf.fit_transform(lemmatized_health)
print('TDM for Health\n',TDM_health)
print('DTM for Health\n',DTM_health)
print('TF-IDF for Health\n',TFIDF_health)


DTM_business=vectorizer.fit_transform(lemmatized_business)
TDM_business=DTM_business.T
TFIDF_business=tfidf.fit_transform(lemmatized_business)
print('TDM for Business\n',TDM_business)
print('DTM for Business\n',DTM_business)
print('TF-IDF for Business\n',TFIDF_business)


DTM_globe=vectorizer.fit_transform(lemmatized_globe)
TDM_globe=DTM_globe.T
TFIDF_globe=tfidf.fit_transform(lemmatized_globe)
print('TDM for Globe\n',TDM_globe)
print('DTM for Globe\n',DTM_globe)
print('TF-IDF for Globe\n',TFIDF_globe)


DTM_technology=vectorizer.fit_transform(lemmatized_technology)
TDM_technology=DTM_technology.T
TFIDF_technology=tfidf.fit_transform(lemmatized_technology)
print('TDM for Technology\n',TDM_technology)
print('DTM for Technology\n',DTM_technology)
print('TF-IDF for Technology\n',TFIDF_technology)

# Task 5: Document the findings of POS tagging and WordNet analysis
# TODO: Write a documentation summarizing the accuracy of POS tagging and the usefulness of WordNet
#POS TAGGING TEST
text = "This is an example sentence for POS tagging."
#NLTK TAGGING
tokens = word_tokenize(text)
nltk_pos_tags = nltk.pos_tag(tokens)
print("NLTK POS Tags:", nltk_pos_tags)

#SPACY TAGGING
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
spacy_pos_tags = [(token.text, token.pos_) for token in doc]
print("spaCy POS Tags:", spacy_pos_tags)

#TEXTBLOB TAGGING
blob = TextBlob(text)
textblob_pos_tags = blob.tags
print("TextBlob POS Tags:", textblob_pos_tags)

correct_tags = [('This', 'DT'), ('is', 'VBZ'), ('an', 'DT'), ('example', 'NN'), ('sentence', 'NN'), ('for', 'IN'),
                ('POS', 'NN'), ('tagging', 'NN')]

#Calculate accuracy for NLTK
nltk_correct_count = sum(1 for tag1, tag2 in zip(correct_tags, nltk_pos_tags) if tag1[1] == tag2[1])
nltk_accuracy = nltk_correct_count / len(correct_tags)
print("NLTK Accuracy:", nltk_accuracy)

correct_tags=[('This', 'PRON'), ('is', 'AUX'), ('an', 'DET'), ('example', 'NOUN'), ('sentence', 'NOUN'),
              ('for', 'ADP'), ('POS', 'PROPN'),
              ('tagging', 'NOUN'), ('.', 'PUNCT')]

#Calculate accuracy for spaCy
spacy_correct_count = sum(1 for tag1, tag2 in zip(correct_tags, spacy_pos_tags) if tag1[1] == tag2[1])
spacy_accuracy = spacy_correct_count / len(correct_tags)
print("spaCy Accuracy:", spacy_accuracy)

correct_tags = [('This', 'DT'), ('is', 'VBZ'), ('an', 'DT'), ('example', 'NN'), ('sentence', 'NN'), ('for', 'IN'),
                ('POS', 'NN'), ('tagging', 'NN')]

#Calculate accuracy of TextBlob
textblob_correct_count = sum(1 for tag1, tag2 in zip(correct_tags, textblob_pos_tags) if tag1[1] == tag2[1])
textblob_accuracy = textblob_correct_count / len(correct_tags)
print("TextBlob Accuracy:", textblob_accuracy)


#Cosine Similarity

similarity = cosine_similarity(TFIDF_sports)
print("Cosine Similarity between sports documents is:",similarity[0][1])

similarity = cosine_similarity(TFIDF_health)
print("Cosine Similarity between health documents is:",similarity[0][1])

similarity = cosine_similarity(TFIDF_business)
print("Cosine Similarity between business documents is:",similarity[0][1])

similarity = cosine_similarity(TFIDF_globe)
print("Cosine Similarity between globe documents is:",similarity[0][1])

similarity = cosine_similarity(TFIDF_technology)
print("Cosine Similarity between technology documents is:",similarity[0][1])


# Team 5: NLP Data Visualization and Analysis

# Task 1: Generate word clouds
# TODO: Create word clouds based on the preprocessed data

b_corpus = business
t_corpus = technology
s_corpus = sports
h_corpus = health
g_corpus = globe

# Create WordCloud objects

wc_business = WordCloud(width=800, height=400, background_color='black').generate(" ".join(b_corpus))
wc_technology = WordCloud(width=800, height=400, background_color='skyblue').generate(" ".join(t_corpus))
wc_sports = WordCloud(width=800, height=400, background_color='lavender').generate(" ".join(s_corpus))
wc_health = WordCloud(width=800, height=400, background_color='crimson').generate(" ".join(h_corpus))
wc_globe = WordCloud(width=800, height=400, background_color='mediumorchid').generate(" ".join(g_corpus))

# Create a list of WordCloud objects and corresponding titles
wordclouds = [(wc_business, "Business Word Cloud"),
             (wc_technology, "Technology Word Cloud"),
             (wc_sports, "Sports Word Cloud"),
             (wc_health, "Health Word Cloud"),
             (wc_globe, "Globe Word Cloud")]

# Iterate through the list and display WordClouds with titles
for wordcloud, title in wordclouds:
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()
  
# Task 2: Create frequency distributions
# TODO: Generate frequency distributions of words to gain insights

# Define the text corpora using your variables
b_tokens = word_tokenize(" ".join(b_corpus))
s_tokens = word_tokenize(" ".join(s_corpus))
h_tokens = word_tokenize(" ".join(h_corpus))
g_tokens = word_tokenize(" ".join(g_corpus))
t_tokens = word_tokenize(" ".join(t_corpus))

# Function to remove punctuation and white spaces from a list of tokens
def preprocess_tokens(tokens):
    translator = str.maketrans('', '', string.punctuation)
    tokens = [token.translate(translator).strip() for token in tokens]
    return tokens

# Token lists without punctuation and white spaces
b_tokens = preprocess_tokens(b_tokens)
s_tokens = preprocess_tokens(s_tokens)
h_tokens = preprocess_tokens(h_tokens)
g_tokens = preprocess_tokens(g_tokens)
t_tokens = preprocess_tokens(t_tokens)

# Create a list of token lists and corresponding titles
token_lists = [b_tokens, s_tokens, h_tokens, g_tokens, t_tokens]
titles = ["Business Word Count", "Technology Word Count", "Sports Word Count", "Health Word Count", "Globe Word Count"]

# Initialize a list to store filtered token lists
filtered_token_lists = []

# Remove stopwords from each token list
for tokens in token_lists:
    stop_words = set(stopwords.words("english"))

    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    filtered_token_lists.append(filtered_tokens)

# Calculate the word frequency distribution for each token list
freq_dists = [FreqDist(tokens) for tokens in filtered_token_lists]

# Print the word count for each word in each distribution
for title, freq_dist in zip(titles, freq_dists):
    print(title)
    for word, count in freq_dist.items():
        print(f"{word}: {count}")
    print("\n")

# Task 3: Explore visualization techniques
# TODO: Experiment with different visualization techniques (bar charts, scatter plots, heatmaps)

# Bar Charts for Word Frequencies:

categories = ["Business", "Technology", "Sports", "Health", "Globe"]

# Define categories and corresponding filtered token lists
token_lists = [filtered_token_lists[0], filtered_token_lists[1], filtered_token_lists[2], filtered_token_lists[3], filtered_token_lists[4]]

# Create subplots for word frequency bar charts
plt.figure(figsize=(20,10))
for i in range(len(categories)):
    plt.subplot(2, 3, i + 1)

    # Get the frequency distribution for the current category
    freq_dist = FreqDist(token_lists[i])

    # Plot the top N words by frequency (e.g., top 20)
    top_words = freq_dist.most_common(20)
    words, counts = zip(*top_words)

    plt.bar(words, counts)
    plt.title(f"Top 20 Words in {categories[i]} Category")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Bar stacked Chart for Word Frequency Comparison:

# Define the labels
labels = list(freq_dists[0].keys())[:10]

# Create a list of word frequencies for each category, excluding corresponding labels
word_frequencies = [list(freq_dist.values())[:10] for freq_dist in freq_dists]

# Filter out frequencies for corresponding labels
filtered_word_frequencies = []
for frequencies in word_frequencies:
    filtered_frequencies = []
    for label, freq in zip(labels, frequencies):
        filtered_frequencies.append(freq)
    filtered_word_frequencies.append(filtered_frequencies)

# Create a stacked bar chart with filtered data
plt.figure(figsize=(8, 5))

# Initialize the bottom for each category
bottom = np.zeros(len(labels))

# Create a bar for each category and stack them with filtered data
for i, (category, frequencies) in enumerate(zip(categories, filtered_word_frequencies)):
    plt.bar(labels, frequencies, label=category, alpha=0.7, bottom=bottom)
    bottom += frequencies

plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 10 Word Frequencies Comparison")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# # Heatmap for TF-IDF Matrices:

# Loop through each category

for category in categories:
    # Get the top 10 terms by TF-IDF score for the current category
    top_terms = tfidf.get_feature_names_out()[:10]

    # Extract the TF-IDF values for the top 10 terms in the current category
    if category == "Business":
        TFIDF_category = TFIDF_business
    elif category == "Technology":
        TFIDF_category = TFIDF_technology
    elif category == "Sports":
        TFIDF_category = TFIDF_sports
    elif category == "Health":
        TFIDF_category = TFIDF_health
    elif category == "Globe":
        TFIDF_category = TFIDF_globe

    top_terms_tfidf = TFIDF_category.toarray()[:10, :10]

    # Plot a heatmap for TF-IDF values of the top 10 terms in the current category
    plt.figure(figsize=(10, 5))
    sns.heatmap(top_terms_tfidf, cmap="YlGnBu", xticklabels = top_terms, annot=True)
    plt.title(f"TF-IDF Matrix for Top 10 Terms in {category}")
    plt.xlabel("Terms")
    plt.ylabel("Documents")
    plt.tight_layout()
    plt.show()

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
