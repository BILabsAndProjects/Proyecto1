import pandas as pd
import nltk
import re
nltk.download("stopwords", quiet=True)

from nltk.stem.snowball import SpanishStemmer

class WordTokenizerTransformer:
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.stem = False
        self.wpt = nltk.WordPunctTokenizer()
        self.stop_words = set(nltk.corpus.stopwords.words("spanish"))
        self.spanish_stemmer = SpanishStemmer()
        return self

    def transform(self, X):

        tokenized_text = []
        
        for text in X:
            doc = re.sub(r"[^A-Za-zÁÉÍÓÚáéíóúÜüÑñ\s]", "", text, flags=re.U)  
            doc = doc.lower()
            doc = doc.strip()
            tokens = self.wpt.tokenize(doc)
            # Filtrar palabras
            filtered_tokens = [
                self.spanish_stemmer.stem(token) if self.stem else token
                for token in tokens
                if token not in self.stop_words
            ]
            # Recrear documento de texto
            doc = " ".join(filtered_tokens)
            tokenized_text.append(doc)
        return tokenized_text


    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

