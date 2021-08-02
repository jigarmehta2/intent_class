import pandas as pd
import numpy as np
import argparse
import os
import warnings
import pickle
import logging
import argparse
import re
import warnings
warnings.filterwarnings("ignore")

import nltk # important
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
lemmatizer = nltk.stem.WordNetLemmatizer()
#import spell
#from spell import SpellingAutocorrecter
#sc=SpellingAutocorrecter()

import spacy  # important  -- okay to replace with large version
nlp=spacy.load('en')

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

brackets_re = re.compile(r"[(\[].*?[)\]]")
replace_by_space_re = re.compile(r"[{}|@,;]")
non_alphanum_re = re.compile(r"[^0-9a-z#+_]")

#train_file_name="NES_training_data.csv"

# utterance cleaning  function
def clean_text(text, spell_check=False):
        text = str(text).lower()  # lowercase text
        # regex clean operations
        text = re.sub(brackets_re, "", text)  # remove [] & () brackets
        text = re.sub(replace_by_space_re, " ", text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = re.sub(non_alphanum_re, " ", text)  # delete symbols which are not alphanumeric numbers from text
        #text = re.sub(r"\s{2,}", " ", text)
        # in-house financial spell checker
        if spell_check:
            text = " ".join(sc.correction(sc.tokenize(text,pos="v")))
        #lemmatization
        text = " ".join([lemmatizer.lemmatize(w) for w in text.split(" ")])
        return text  # return cleaned text

# spacy nlp features function -- pos, dependency tags, tags, NER, stop word flag
def spacy_features(text):
    doc = nlp(text)  # spacy object <-- nlp is spacy built in function
    ftrs = []
    for token in doc:
        ftrs.extend([ent.label_ for ent in doc.ents if ent.text == token.text])
        ftrs.extend([token.lemma_, token.pos_, token.tag_, token.dep_, str(token.is_stop)])

    ftrs = " ".join(ftrs)
    return ftrs  # return features

    #clean intents function
def run_preprocess(raw_data=None,spacy='yes'):
    
    print("\nRunning preprocessing ...")
    raw_data['word_len']=raw_data.Sentence.str.split(" ").map(len)
    raw_data=raw_data[raw_data.word_len>1]
    del raw_data['word_len']
    
    try:
        raw_data['sent']=raw_data.Sentence.apply(clean_text) #clean data
    except:
         print("\nError occured in clean text function of preprcoessing stage")
            
    if(spacy=="yes"):
        try:
            raw_data['feat']=raw_data.sent.apply(spacy_features)  #Running Spacy NLP Features
            
       
        except:
            print("\nError occured in Spacy features function of preprcoessing stage")
    print("Preprocessing completed successfully")
    return raw_data