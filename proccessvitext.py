import string
import unicodedata
from underthesea import text_normalize
from bs4 import BeautifulSoup
from vncorenlp import VnCoreNLP

def remove_html_tags(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_punctuation(text: str) -> str:
    result = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    return ' '.join(result.split())

def to_lower(text: str) -> str:
    return text.lower()

def standardize_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)

def normalize_text(text: str) -> str:
    text = text_normalize(text)
    return text

def stopword_removal(text: str, stopwords: set) -> str:
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

def preprocess_text(text: str, stopwords: set) -> str:
    text = remove_html_tags(text)
    text = to_lower(text)
    text = remove_punctuation(text)
    text = standardize_unicode(text)
    text = normalize_text(text)
    
    text = stopword_removal(text, stopwords)
    return text

if __name__ == "__main__":
    annotator = VnCoreNLP("vncorenlp/VnCoreNLP-1.2.jar", annotators="wseg")

    words = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
    result = annotator.tokenize(words)
    print(result)

    annotator.close()