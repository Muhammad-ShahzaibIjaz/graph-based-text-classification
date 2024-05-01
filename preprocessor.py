from spellchecker import SpellChecker
from collections import Counter
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk.tokenize import word_tokenize

PUNCT_TO_REMOVE = string.punctuation
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()


def tokenize(text):
    return word_tokenize(text)


def lowercase_text(text):
    return text.lower()


def removeHtmlTags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def removeNumbers(text):
    return re.sub(r'\d+', '', text)


def removeUrls(text):
    urlPattern = re.compile(r'https?://\S+|www\.\S+')
    cleanText = urlPattern.sub('', text)
    return cleanText


def removePunctuation(text):
    return ' '.join(text.translate(str.maketrans('', '', PUNCT_TO_REMOVE)).split())


def removeWhitespace(text):
    return ' '.join(text.split())


def removeStopwords(text):
    return ' '.join([word for word in str(text).split() if word not in STOPWORDS])


def lemmatizeWords(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def removeEmoji(text):
    emojiPattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emojiPattern.sub(r'', text)


def stemWords(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


def posTagging(text):
    result = TextBlob(text)
    return result.tags


def remove_words_above_threshold(text, threshold=5):
    tokens = word_tokenize(text)
    wordFreq = Counter(tokens)

    filteredWords = [word for word in tokens if wordFreq[word] >= threshold]

    return ' '.join(filteredWords)


def removeFrequentWords(tokens, top_n=10):
    wordFreq = Counter(tokens)
    frequentWords = set([word for word, _ in wordFreq.most_common(top_n)])
    filteredTokens = [word for word in tokens if word not in frequentWords]
    return filteredTokens


def removeEmoticons(text):
    emoticonPattern = r'(?::|;|=)(?:-)?(?:\)|\(|D|P)'
    text_without_emoticons = re.sub(emoticonPattern, '', text)
    return text_without_emoticons

def removeSpecialSymbols(text):
    return re.sub(r'[^A-Za-z0-9\s]+', '', text)

def preprocessorPerformer(txt):
    txt = removeUrls(txt)
    txt = removeHtmlTags(txt)
    txt = removeWhitespace(txt)
    txt = removePunctuation(txt)
    txt = lowercase_text(txt)
    txt = removeEmoticons(txt)
    txt = removeSpecialSymbols(txt)
    txt = removeEmoji(txt)
    txt = removeNumbers(txt)
    txt = removeStopwords(txt)
    txt = lemmatizeWords(txt)
    txt = stemWords(txt)
    return txt