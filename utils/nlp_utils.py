# utils/nlp_utils.py
import os
import spacy
import nltk

def setup_nlp():
    """Download required NLP resources and load SpaCy models."""
    # NLTK
    nltk.download('punkt', quiet=True)

    # SpaCy English
    try:
        spacy_en = spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    # SpaCy German
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except OSError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    return spacy_en, spacy_de
