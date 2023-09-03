#Team 1:

import requests
from bs4 import BeautifulSoup as soup
import emoji
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import emoji

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def corp_build(x):
  text=""
  text = " ".join(i for i in x)
  return text


#Sport News
sports_url1 = "https://indianexpress.com/article/sports/tennis/novak-djokovic-ties-roger-federer-wimbledon-8828916/"
sports1 = []

# Scraping Indian Express URL
html = requests.get(sports_url1)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'native_story_title'})
sports1.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisement
article = bsobj.find('div', {'class': 'full-details'})
for paragraph in article.find_all('p'):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    sports1.append(paragraph.text)
sports1 = corp_build(sports1)

sports_url2 = "https://www.nydailynews.com/sports/more-sports/ny-novak-djokovic-roger-federer-grand-slam-wimbledon-20230711-eyeiq4rnazc5lg2hkd5ykoi46u-story.html"

sports2 = []

# Scraping NY Daily News URL
html = requests.get(sports_url2)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'primary-font__PrimaryFontStyles-o56yd5-0 gVBMpi headline'})
sports2.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisement and text below the image
article = bsobj.find('article', {'class': 'default__ArticleBody-sc-1wxyvyl-2 hEvcgL article-body-wrapper-custom'})
for element in article.find_all(['p', 'figure']):
    # Skip internal links
    if element.find('a', href=True):
        continue

    # Check if the element contains the advertisement phrase and skip it
    if "Get the latest odds on all the top sports" in element.text:
        continue

    sports2.append(element.text.strip())
sports2 = corp_build(sports2)

# Technical News
tech_url1 = "https://indianexpress.com/article/technology/tech-news-technology/amazon-makes-first-big-tech-challenge-to-eu-online-content-rules-8829113/"

tech1 = []

# Scraping Indian Express URL
html = requests.get(tech_url1)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'native_story_title'})
tech1.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisement
article = bsobj.find('div', {'class': 'full-details'})
for paragraph in article.find_all('p'):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    tech1.append(paragraph.text.strip())
tech1 = corp_build(tech1)

tech_url2 = "https://www.reuters.com/technology/amazon-challenges-eu-online-content-rules-says-unfairly-singled-out-2023-07-11/"

tech2 = []

# Scraping Reuters URL
html = requests.get(tech_url2)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'text__text__1FZLe text__dark-grey__3Ml43 text__medium__1kbOh text__heading_3__1kDhc heading__base__2T28j heading__heading_3__3aL54 article-header__title__3Y2hh'})
tech2.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisements
article = bsobj.find('div', {'class': 'article-body__content__17Yit'})
for paragraph in article.find_all(['p']):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    tech2.append(paragraph.text.strip())
tech2 = corp_build(tech2)


# Education News
education_url1 = "https://indianexpress.com/article/education/aiims-proposes-to-quash-interview-for-phd-selection-8829283/"

edu1 = []

# Scraping Indian Express URL
html = requests.get(education_url1)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'native_story_title'})
edu1.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisement
article = bsobj.find('div', {'class': 'full-details'})
for paragraph in article.find_all('p'):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    edu1.append(paragraph.text.strip())
edu1 = corp_build(edu1)

education_url2 = "https://theprint.in/india/aiims-proposes-to-quash-interviews-in-phd-selection-process-for-greater-transparency/1665108/#google_vignette"

edu2 = []

# Scraping Theprint URL
html = requests.get(education_url2)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'tdb-title-text'})
edu2.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisement
article = bsobj.find('div', {'class': 'td-post-content'})
for paragraph in article.find_all('p'):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    edu2.append(paragraph.text.strip())
edu2 = corp_build(edu2)


# Politics News
politics_url1 = "https://indianexpress.com/article/political-pulse/sc-prepares-article-370-pleas-look-major-parties-stand-8829676/"
pol1 = []

# Scraping Indian Express URL
html = requests.get(politics_url1)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'native_story_title'})
pol1.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisement
article = bsobj.find('div', {'class': 'full-details'})
for paragraph in article.find_all('p'):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    pol1.append(paragraph.text.strip())
pol1=corp_build(pol1)

politics_url2 = "https://www.livemint.com/news/india/jammu-and-kashmir-sc-to-hear-batch-of-pleas-challenging-the-abrogation-of-article-370-from-august-2-11689053035450.html"

pol2 = []

# Scraping LiveMint URL
html = requests.get(politics_url2)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'headline'})
pol2.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisement
article = bsobj.find('div', {'class': 'contentSec'})
for paragraph in article.find_all('p'):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    pol2.append(paragraph.text.strip())
pol2= corp_build(pol2)

# Global News
global_url1 = "https://indianexpress.com/article/explained/explained-global/swedens-rocky-road-from-neutrality-toward-nato-membership-8827291/"

glo1 = []

# Scraping Indian Express URL
html = requests.get(global_url1)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'native_story_title'})
glo1.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisement
article = bsobj.find('div', {'class': 'full-details'})
for paragraph in article.find_all('p'):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    glo1.append(paragraph.text.strip())
glo1 = corp_build(glo1)

global_url2 = "https://hindustannewshub.com/world-news/swedens-rocky-road-from-neutrality-toward-nato-membership/"

glo2 = []

# Scraping Hindustan News URL
html = requests.get(global_url2)
bsobj = soup(html.content, 'lxml')

# Find and print the headline
headline = bsobj.find('h1', {'class': 'post-title entry-title'})
glo2.append("Headline: {}".format(headline.text.strip()))

# Find and print the article content, excluding the advertisements
article = bsobj.find('div', {'class': 'entry-content entry clearfix'})
for paragraph in article.find_all(['p']):
    # Check for the advertisement phrase and skip the paragraph if found
    if "advertisement" in paragraph.text:
        continue
    glo2.append(paragraph.text.strip())
glo2 = corp_build(glo2)

# Task 6: Document the web scraping process
# TODO: Write a detailed documentation of the web scraping process and challenges faced


# Team 2: NLP Preprocessing
# Building corpora of article of similar genre 
sports_corp=[sports1,sports2]
politics_corp = [pol1,pol2]
education_corp = [edu1,edu2]
global_corp = [glo1,glo2]
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
    punc = ['“', "”", "’", "…", "‘", "—"]
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
sports = [text_preprocessing(sports_corp)]
politics = [text_preprocessing(politics_corp)]
education = [text_preprocessing(education_corp)]
globe = [text_preprocessing(global_corp)]
technology = [text_preprocessing(tech_corp)]



# Team 3: NLP Morphological Analysis and Tokenization
# Team 3: NLP Morphological Analysis and Tokenization
# TODO: Import necessary libraries for morphological analysis and tokenization 
#task 1
# Task 1: Implement word tokenization techniques
# TODO: Write functions to perform different word tokenization techniques (whitespace, regex, etc.)


import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer, TreebankWordTokenizer
import sentencepiece as spm
from tokenizers import ByteLevelBPETokenizer, SentencePieceBPETokenizer
import re

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
politics_corpus = politics

technology_corpus = technology

globe_corpus = globe

education_corpus = education

sports_corpus = sports

#applying the tokenize techniques to corpora
whitespace_tokenized_politics = whitespace_tokenize_corpus(politics_corpus)
wordpunct_tokenized_politics = wordpunct_tokenize_corpus(politics_corpus)
treebank_tokenized_politics = treebank_tokenize_corpus(politics_corpus)
tweet_tokenized_politics = tweet_tokenize_corpus(politics_corpus)
subword_tokenized_politics = subword_tokenize_corpus(politics_corpus)

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


whitespace_tokenized_education = whitespace_tokenize_corpus(education_corpus)
wordpunct_tokenized_education = wordpunct_tokenize_corpus(education_corpus)
treebank_tokenized_education = treebank_tokenize_corpus(education_corpus)
tweet_tokenized_education = tweet_tokenize_corpus(education_corpus)
subword_tokenized_education = subword_tokenize_corpus(education_corpus)

# Print results for corpus
print("Whitespace Tokenization (Politics):", whitespace_tokenized_politics)
print("WordPunct Tokenization (Politics):", wordpunct_tokenized_politics)
print("Penn Treebank Tokenization (Politics):", treebank_tokenized_politics)
print("Tweet Tokenization (Politics):", tweet_tokenized_politics)
print("Whitespace Subword Tokenization (Politics):", subword_tokenized_politics)

# Print results for technology corpus
print("Whitespace Tokenization (Technology):", whitespace_tokenized_technology)
print("WordPunct Tokenization (Technology):", wordpunct_tokenized_technology)
print("Penn Treebank Tokenization (Technology):", treebank_tokenized_technology)
print("Tweet Tokenization (Technology):", tweet_tokenized_technology)
print("Whitespace Subword Tokenization (Technology):", subword_tokenized_technology)


# Task 2: Apply sentence tokenization techniques
# TODO: Write a function to apply sentence tokenization to segment text into sentences

# task 2
import nltk
nltk.download('punkt')

# Define a function
def apply_sentence_tokenization_list(texts):
    tokenized_sentences = []
    for text in texts:
        sentences = nltk.sent_tokenize(text)
        tokenized_sentences.append(sentences)
    return tokenized_sentences
#corpora
corpora = {
    "technology": technology,
    "sports": sports,
    "politics": politics,
    "education": education,
    "globe": globe
}

# applying tokenize techniques
tokenized_corpora = {}
for corpus_name, corpus in corpora.items():
    tokenized_sentences = apply_sentence_tokenization_list(corpus)
    tokenized_corpora[corpus_name] = tokenized_sentences

# Printing for each corpus 
for corpus_name, sentences_list in tokenized_corpora.items():
    print(f"Tokenized Sentences for {corpus_name.capitalize()} Corpus:")
    for i, sentences in enumerate(sentences_list):
        print(f"Text {i + 1}:")
        for j, sentence in enumerate(sentences):
            print(f"Sentence {j + 1}: {sentence}")
x_technology = tokenized_corpora["technology"]
x_sports = tokenized_corpora["sports"]
x_politics = tokenized_corpora["politics"]
print("x_politics",x_politics)
x_education = tokenized_corpora["education"]
x_globe = tokenized_corpora["globe"]

# Task 3: Experiment with morphological analysis methods
# TODO: Implement and evaluate different morphological analysis methods (stemming, lemmatization, etc.)
# function for stemming and lemmatization
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
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
    "politics": politics,
    "education": education,
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

stemmed_politics = stemmed_corpora["politics"]
lemmatized_politics = lemmatized_corpora["politics"]

stemmed_education = stemmed_corpora["education"]
lemmatized_education = lemmatized_corpora["education"]

stemmed_globe = stemmed_corpora["globe"]
lemmatized_globe = lemmatized_corpora["globe"]





# Task 4: Evaluate the performance of tokenization and morphological analysis techniques
# TODO: Compare and document the performance and effectiveness of the implemented techniques
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Creating an instance of CountVectorizer
vectorizer = CountVectorizer()

#preprocessed data for each corpus and technique

corpora = {
    "politics": {
        "original": " ".join(politics),
        "tokenized": " ".join(" ".join(doc) for doc in x_politics),
        "stemmed": " ".join(stemmed_politics),
        "lemmatized": " ".join(lemmatized_politics)
    },
    "sports": {
        "original": " ".join(sports),
        "tokenized": " ".join(" ".join(doc) for doc in x_sports),
        "stemmed": " ".join(stemmed_sports),
        "lemmatized": " ".join(lemmatized_sports)
    },
    "education": {
        "original": " ".join(education),
        "tokenized": " ".join(" ".join(doc) for doc in x_education),
        "stemmed": " ".join(stemmed_education),
        "lemmatized": " ".join(lemmatized_education)
    },
    "globe": {
        "original": " ".join(globe),
        "tokenized": " ".join(" ".join(doc) for doc in x_globe),
        "stemmed": " ".join(stemmed_globe),
        "lemmatized": " ".join(lemmatized_globe)
    },
    "technology": {
        "original": " ".join(technology),
        "tokenized": " ".join(" ".join(doc) for doc in x_technology),
        "stemmed": " ".join(stemmed_technology),
        "lemmatized": " ".join(lemmatized_technology)
    }
}

def compare_bow_similarity(corpus_name, corpus_data):
    print(f"Comparing bag-of-words representations for corpus: {corpus_name}")

    # original and different technique data
    original_corpus = corpus_data["original"]
    tokenized_corpus = corpus_data["tokenized"]
    stemmed_corpus = corpus_data["stemmed"]
    lemmatized_corpus = corpus_data["lemmatized"]

    # Combining all documents for fitting the vectorizer
    all_documents = [original_corpus, tokenized_corpus , stemmed_corpus , lemmatized_corpus]

    # Fiting the vectorizer on all documents
    vectorizer.fit(all_documents)

    # Transforming documents into bag-of-words representations
    original_bow = vectorizer.transform([original_corpus])
    tokenized_bow = vectorizer.transform([tokenized_corpus])
    stemmed_bow = vectorizer.transform([stemmed_corpus])
    lemmatized_bow = vectorizer.transform([lemmatized_corpus])

    # Calculating cosine similarity between different techniques and the original data
    print("Cosine Similarity Matrix for Tokenized:")
    print(cosine_similarity(original_bow, tokenized_bow))

    print("Cosine Similarity Matrix for Stemmed:")
    print(cosine_similarity(original_bow, stemmed_bow))

    print("Cosine Similarity Matrix for Lemmatized:")
    print(cosine_similarity(original_bow, lemmatized_bow))

    print("\n")

# Comparing bag-of-words representations for each corpus and technique
for corpus_name, corpus_data in corpora.items():
    compare_bow_similarity(corpus_name, corpus_data)


# Team 4: NLP Part of Speech Tagging and WordNet Analysis
# TODO: Import necessary libraries for part of speech tagging and WordNet analysis
from nltk.tag import pos_tag
import spacy
nlp_spacy = spacy.load('en_core_web_sm')
from nltk.corpus import wordnet
from textblob import Word
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
# Task 1: Implement part of speech tagging
# TODO: Write a function to perform part of speech tagging using NLTK, Spacy, or other libraries

def pos_tagging(texts):
    # NLTK
    words_nltk = nltk.word_tokenize(text)
    pos_tags_nltk = nltk.pos_tag(words_nltk)
    # spaCy
    doc_spacy = nlp_spacy(text)
    pos_tags_spacy = [(token.text, token.pos_) for token in doc_spacy]
    return pos_tags_nltk,pos_tags_spacy

# Task 2: Apply POS tagging to preprocessed data
# TODO: Apply the POS tagging function to the preprocessed data

for category, data in corpora.items():
    preprocessed_text = data["original"]
    nltk_tag,spacy_tag=pos_tagging(preprocessed_text)
    print(f'NLTK Tag for {category}',nltk_tag)
    print(f'Spacy Tag for {category}',spacy_tag)
    print()  # Separating line between categories

# Task 3: Perform WordNet analysis
# TODO: Utilize WordNet or similar resources for semantic analysis, synonym identification, etc.
topics = corpora.keys()

set_of_nltk_synonyms,set_of_tb_synonyms,set_of_nltk_hypernyms,set_of_tb_hypernyms={},{},{},{}
for topic in topics:
    sentence = corpora[topic]['lemmatized']
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
from textblob import TextBlob

def pos_exp_tagging(text):
    # TextBlob
    blob = TextBlob(text)
    pos_tags_textblob = blob.tags    
    return pos_tags_textblob

for category, data in corpora.items():
    preprocessed_text = data["original"]
    sports_blob_tag=pos_exp_tagging(preprocessed_text)
    print(f'TextBlob Tag for {category}\n',sports_blob_tag)

#Pattern cannot be used because 

#TDM, DTM, TF-IDF
tfidf=TfidfVectorizer()

DTM_sports=vectorizer.fit_transform(lemmatized_sports)
TDM_sports=DTM_sports.T
TFIDF_sports=tfidf.fit_transform(lemmatized_sports)
print('TDM for Sports\n',TDM_sports)
print('DTM for Sports\n',DTM_sports)
print('TF-IDF for Sports\n',TFIDF_sports)

DTM_politics=vectorizer.fit_transform(lemmatized_politics)
TDM_politics=DTM_politics.T
TFIDF_politics=tfidf.fit_transform(lemmatized_politics)
print('TDM for Politics\n',TDM_politics)
print('DTM for Politics\n',DTM_politics)
print('TF-IDF for Politics\n',TFIDF_politics)


DTM_education=vectorizer.fit_transform(lemmatized_education)
TDM_education=DTM_education.T
TFIDF_education=tfidf.fit_transform(lemmatized_education)
print('TDM for Education\n',TDM_education)
print('DTM for Education\n',DTM_education)
print('TF-IDF for Education\n',TFIDF_education)


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
import nltk
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
nltk_pos_tags = nltk.pos_tag(tokens)
print("NLTK POS Tags:", nltk_pos_tags)

#SPACY TAGGING
import spacy
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
