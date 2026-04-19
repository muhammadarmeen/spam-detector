import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

def clean_text(text):
    """Remove special characters and convert to lowercase"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, digits, and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords_and_lemmatize(text):
    """Remove stopwords and perform lemmatization"""
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)

def preprocess_pipeline(text):
    """Complete preprocessing pipeline"""
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Clean text
    text = clean_text(text)
    
    # Step 2: Remove stopwords and lemmatize
    text = remove_stopwords_and_lemmatize(text)
    
    return text
