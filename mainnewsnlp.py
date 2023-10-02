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

#applying the tokenize techniques to corpora
whitespace_tokenized_business = whitespace_tokenize_corpus(business_corpus)
wordpunct_tokenized_business = wordpunct_tokenize_corpus(business_corpus)
treebank_tokenized_business= treebank_tokenize_corpus(business_corpus)
tweet_tokenized_business = tweet_tokenize_corpus(business_corpus)
subword_tokenized_business = subword_tokenize_corpus(business_corpus)

whitespace_tokenized_technology = whitespace_tokenize_corpus(technology_corpus)
wordpunct_tokenized_technology = wordpunct_tokenize_corpus(technology_corpus)
treebank_tokenized_technology = treebank_tokenize_corpus(technology_corpus)
tweet_tokenized_technology = tweet_tokenize_corpus(technology_corpus)
subword_tokenized_technology = subword_tokenize_corpus(technology_corpus)

whitespace_tokenized_globe = whitespace_tokenize_corpus(globe_corpus)
wordpunct_tokenized_globe = wordpunct_tokenize_corpus(globe_corpus)
treebank_tokenized_globe = treebank_tokenize_corpus(globe_corpus)
tweet_tokenized_globe = tweet_tokenize_corpus(globe_corpus)
subword_tokenized_globe = subword_tokenize_corpus(globe_corpus)


whitespace_tokenized_sports = whitespace_tokenize_corpus(sports_corpus)
wordpunct_tokenized_sports = wordpunct_tokenize_corpus(sports_corpus)
treebank_tokenized_sports = treebank_tokenize_corpus(sports_corpus)
tweet_tokenized_sports = tweet_tokenize_corpus(sports_corpus)
subword_tokenized_sports = subword_tokenize_corpus(sports_corpus)


whitespace_tokenized_health = whitespace_tokenize_corpus(health_corpus)
wordpunct_tokenized_health = wordpunct_tokenize_corpus(health_corpus)
treebank_tokenized_health  = treebank_tokenize_corpus(health_corpus)
tweet_tokenized_health  = tweet_tokenize_corpus(health_corpus)
subword_tokenized_health  = subword_tokenize_corpus(health_corpus)

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


# Task 2: Apply sentence tokenization techniques
# TODO: Write a function to apply sentence tokenization to segment text into sentences
# the sentence tokenization task because it doesn't make sense to perform sentence tokenization when there are no full stops to split the sentences in your text data.

# Task 3: Experiment with morphological analysis methods
# TODO: Implement and evaluate different morphological analysis methods (stemming, lemmatization, etc.)
# function for stemming and lemmatization

def apply_morphological_analysis(corpus):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    stemmed_corpus = []
    lemmatized_corpus = []

    for text in corpus:
        words = nltk.word_tokenize(text)
        stemmed_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]

        stemmed_corpus.append(" ".join(stemmed_words))
        lemmatized_corpus.append(" ".join(lemmatized_words))

    return stemmed_corpus, lemmatized_corpus

# corpora
corpora = {
    "technology": technology,
    "sports": sports,
    "business": business,
    "health": health,
    "globe" : globe
}

# Applying morphological analysis to each corpus
stemmed_corpora = {}
lemmatized_corpora = {}

for corpus_name, corpus in corpora.items():
    stemmed_corpora[corpus_name], lemmatized_corpora[corpus_name] = apply_morphological_analysis(corpus)

# Printed the stemmed and lemmatized results for each corpus
for corpus_name in corpora.keys():
    print(f"Stemmed Corpus for {corpus_name.capitalize()}:")
    for i, text in enumerate(stemmed_corpora[corpus_name]):
        print(f"Text {i + 1}: {text}")

    print(f"\nLemmatized Corpus for {corpus_name.capitalize()}:")
    for i, text in enumerate(lemmatized_corpora[corpus_name]):
        print(f"Text {i + 1}: {text}")


stemmed_technology = stemmed_corpora["technology"]
lemmatized_technology = lemmatized_corpora["technology"]

stemmed_sports = stemmed_corpora["sports"]
lemmatized_sports = lemmatized_corpora["sports"]

stemmed_business = stemmed_corpora["business"]
lemmatized_business = lemmatized_corpora["business"]

stemmed_health = stemmed_corpora["health"]
lemmatized_health = lemmatized_corpora["health"]

stemmed_globe = stemmed_corpora["globe"]
lemmatized_globe = lemmatized_corpora["globe"]



# Task 4: Evaluate the performance of tokenization and morphological analysis techniques
# TODO: Compare and document the performance and effectiveness of the implemented techniques
# Ensure you have downloaded NLTK data
nltk.download('punkt')

# document for different corpora
corpus_business = business
corpus_sports = sports
corpus_health = health
corpus_globe = globe
corpus_technology = technology


# Define your documents for different corpora
corpus_business = business
corpus_technology = technology
corpus_globe = globe
corpus_health = health
corpus_sports = sports


# Define your documents for different corpora
corpus_business = business
corpus_technology = technology
corpus_globe = globe
corpus_health =health
corpus_sports = sports



# Define a function to create a count matrix using Count Vectorizer
def create_count_matrix(documents):
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(documents)
    return count_matrix

# Create a count matrix for each corpus
count_matrix_politics = create_count_matrix(corpus_business)
count_matrix_technology = create_count_matrix(corpus_technology)
count_matrix_sports = create_count_matrix(corpus_sports)
count_matrix_globe = create_count_matrix(corpus_globe)
count_matrix_education = create_count_matrix(corpus_health)

# Print the count matrices
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
