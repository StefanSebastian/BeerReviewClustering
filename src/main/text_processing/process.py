import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from src.main.algorithms.utils import serialize

from src.main.config import processed_path, features_path


def do_vectorization(file):
    beerdf = pd.read_csv(processed_path + "\\" + file)

    punctuation = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "%"]
    stop_words = text.ENGLISH_STOP_WORDS.union(punctuation)
    desc = beerdf['review_text'].values.astype('U')
    stemmer = SnowballStemmer('english')
    tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

    def tokenize(text):
        return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]

    vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize, max_features=1000)
    features = vectorizer.fit_transform(desc)

    serialize(features, features_path + "\\" + file + '.features')

do_vectorization('small.csv')
do_vectorization('train.csv')
