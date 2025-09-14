import re
import string
import spacy
import nltk
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def full_preprocess(text): #for fake news model
    # Lowercase
    text = text.lower()
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # Remove brackets and content within
    text = re.sub('\[.*\]','', text)
    # Remove words with digits
    text = re.sub('\S*\d\S*\s*','', text)
    # Remove URLs
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    # Remove single characters surrounded by whitespace
    text = re.sub(r'\s+[a-zA-Z]\s+', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Lemmatize and remove stop words and non-alpha tokens
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)


def preprocess_text(text): #for category model
    """
    Complete preprocessing pipeline:
    - lowercase
    - remove HTML tags
    - remove URLs
    - remove slashes
    - replace line breaks with space
    - remove escaped quotes
    - remove punctuation
    - remove stopwords
    """
    if not isinstance(text, str):
        return ''

    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'http\S+', '', text)  # extra catch

    # Remove slashes
    text = re.sub(r'[\\/]', '', text)

    # Replace line breaks with space
    text = re.sub(r'\n', ' ', text)

    # Replace escaped quotes
    text = re.sub(r"\\'", "'", text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]

    return " ".join(tokens)

# Category mapping
category_map = {
    0: "ARTS & CULTURE",
    1: "BUSINESS",
    2: "COMEDY",
    3: "CRIME",
    4: "EDUCATION",
    5: "ENTERTAINMENT",
    6: "ENVIRONMENT",
    7: "MEDIA",
    8: "POLITICS",
    9: "RELIGION",
    10: "SCIENCE",
    11: "SPORTS",
    12: "TECH",
    13: "WOMEN"
}