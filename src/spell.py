import re
from collections import Counter
from sys import argv
import pandas as pd
import pickle
import math
import itertools
import time
import csv
import os

class SpellingAutocorrecter():
    """
    Tool to identify spelling mistakes and automatically correct them.
    Tailored to Fidelity-specific data.
    """
    
    def __init__(self, training_data_filepath="../data/"):
        """
        Initializes attributes and builds language model
        """
        self.vocab = Counter()
        self._vocab_set = set()
        self.bigrams = Counter()
        self._sum_of_vocab_counts = 0
        self._build_vocab(training_data_filepath)
        
    def _build_vocab(self, training_data_filepath):
        """
        Creates language model for selecting among candidate corrections.
        Imports binarized language model if possible; builds and stores as binary if not.
        """
        try:
            with open(os.path.join(training_data_filepath,'vocab.pickle'), 'rb') as f:
                self.vocab = pickle.load(f)
                
            with open(os.path.join(training_data_filepath,'bigrams.pickle'), 'rb') as f:
                self.bigrams = pickle.load(f)
        except FileNotFoundError:
            with open(os.path.join(training_data_filepath,'vocab.txt'), 'r', encoding='latin-1') as f:
                self._vocab_set = set(f.read().split('\n'))
            for word in self._vocab_set:
                self.vocab[word] += 1
            data = self._read_csv(os.path.join(training_data_filepath,'vocab_sentences.csv'))[['Sentence','Count']]
            data.apply(self._parse_row, axis=1)
            with open(os.path.join(training_data_filepath,'vocab.pickle'),'wb') as f:
                pickle.dump(self.vocab, f)
            with open(os.path.join(training_data_filepath,'bigrams.pickle'),'wb') as f:
                pickle.dump(self.bigrams, f)
        self._sum_of_vocab_counts = sum(self.vocab.values())
        
    def _candidates(self, word):
        """
        Identifies misspelled words and returns a list of possible candidates to be corrected to.
        """
        # Ignore non-letters (numbers, punctuation) and urls
        if re.fullmatch('[^a-z\-]+',word) is not None or '.com' in word or '.org' in word or '.net' in word or '.gov' in word or '.co.' in word:
            return [word]
        # For short words only look at candidates 1 edit distance away to same computation time.
        if len(word) < 6:
            return (self._known([word]) or self._known(self._edits1(word)) or [word])
        # Otherwise generate all candidates up to 2 edit distance away.
        return (self._known([word]) or self._known(self._edits1(word)).union(self._known(self._edits2(word))) or [word])
        
        #return (self._known([word]) or self._alt_edits(word) or [word])
        
    def _edits1(self, word):
        """
        Generate a list of candidate strings 1 edit distance away from passed word.
        Uses damerau-levenshtein distance (additions, deletions, insertions, transpositions)
        """
        letters = 'abcdefghijklmnopqrstuvwxyz\'-1234567890'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L,R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
        

    def _edits2(self, word):
        """Uses _edits1() to get all strings edit distance 2 from word"""
        return (e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))
        
    def _get_candidate_sentences(self, sentence):
        """
        Generates all permutations of a sentence based on candidates for each misspelled word.
        @pre Tokenized list of strings
        """
        new_sent = [[x] for x in sentence]
        for i in range(len(new_sent)):
            new_sent[i] = list(self._candidates(new_sent[i][0]))
            
        return [['<S>'] + list(s)  + ['</S>'] for s in itertools.product(*new_sent)]
        
    def _known(self, words):
        """Return all words in a passed sequence that are in-vocab"""
        return set(w for w in words if w in self.vocab)

    def _P(self, word):
        """Returns probability of word based on stored vocab"""
        return self.vocab[word] / self._sum_of_vocab_counts
        
    def _parse_row(self, row):
        """
        Reads a row of training data and populates vocab and bigram probabilities.
        Input is a pandas Series object.
        """
        # Nuance data is pre-tokenized so just split on white space
        sentence = row['Sentence'].split(' ')
        for i in range(len(sentence)):
            word = sentence[i]
            # Skip rows that have misspellings to avoid confusing the model
            if re.fullmatch('[A-Za-z0-9_\-]+',word) is not None and word not in self._vocab_set:
                return
            # Experimented with creating placeholder tags for numbers. Left out for now but 
            # looking at for MVP-2
            #if re.fullmatch('(\$?)([0-9]+)(\.?)([0-9])*', word) is not None and word not in self._vocab_set:
            #    sentence[i] = "<NUM>"
        # Use the count in NES to weight some utterances more than others
        for count in range(row['Count']):
            self.bigrams[('<S>',sentence[0])] += 1
            self.vocab['<S>'] += 1
            self.vocab[sentence[0]] += 1
            for i in range(len(sentence)-1):
                if sentence[i] in self._vocab_set:
                    self.vocab[sentence[i]] += 1
                self.bigrams[(sentence[i],sentence[i+1])] += 1
            self.bigrams[(sentence[-1], '</S>')] += 1
            self.vocab['</S>'] += 1        
        
    def _read_csv(self, filename):
        """Import data for training language model"""
        data = pd.read_csv(filename)
        return data
        
    def correct_file(self,input_filepath,output_filepath=""):
        """
        Reads text from file, and returns each line corrected.
        Handles .txt and .csv. For csv, the sentence to be 
        corrected is assumed to be the first column.
        """
        sents = []
        
        with open(input_filepath, 'r') as f:
            if input_filepath[-4:] == ".csv":
                sents = [x[0] for x in csv.reader(f)][1:]
            else:
                sents = f.read().split('\n')
        #print(sents)
        corrected_sents = []
        for sent in sents:
            corrected_sents.append(self.correction(self.tokenize(sent)))
        if output_filepath != "":
            with open(output_filepath, 'w') as f:
                for sent in corrected_sents:
                    f.write(' '.join(sent) + '\n')
        return corrected_sents
        
        
    def correction(self, sentence):
        """
        Gets list of candidate replacement sentences and returns the one with 
        the highest score based on the language model.
        @pre Tokenized list of strings
        """
        candidates = self._get_candidate_sentences(sentence)
        scored_candidates = []
        for candidate in candidates:
            word_scores = 0
            for i in range(1,len(candidate)):
                # Gets bigram language model score. Uses backoff to unigram for unknown bigram pairs
                try:
                    score = self.bigrams[(candidate[i-1], candidate[i])] / self.vocab[candidate[i-1]]
                except ZeroDivisionError:
                    score = self._P(candidate[i])
                score = math.log2(score + (1 / self._sum_of_vocab_counts))
                word_scores += score
            scored_candidates.append((candidate,word_scores))
        #for cand in scored_candidates:
        #    print(cand)
        best_candidate = max(scored_candidates,key=lambda x:x[1])[0][1:-1]
        return best_candidate
        
    def tokenize(self, input):
        """
        Tokenizes a string as close to the way NES does as possible.
        Returns a list of strings representing identified individual tokens.
        """
        # Set of all contractions to exclude when parsing words with "'". 
        #Probably a cleaner way to do this but it works for now 
        contractions = {"can't", "won't", "didn't", "shouldn't", "aren't", "could've", "couldn't've", "doesn't", "don't", "everbody's", "everyone's",
                                "hadn't", "hasn't", "haven't", "haven't", "he'd", "he'll", "he's", "how'd", "how'll", "how're", "how's", "i'd", "i'll", "i'm",
                                "i've", "isn't", "it'd", "it'll", "it's", "let's", "ma'am", "might've", "mustn't", "must've", "needn't", "o'clock", "she'd", 
                                "she'll", "she's", "should've", "shouldn't've", "somebody's", "someone's", "something's", "so're", "that'll", "that're", "that's",
                                "that'd", "there'd", "there'll", "there're", "there's", "they'd", "they'll", "they're", "they've", "this's", "those're", "wasn't",
                                "we'd", "we'll", "we're", "we've", "weren't", "what'd", "what'll", "what're", "what's", "what've", "when's", "where'd", "where'll",
                                "where's", "where've", "which'll", "which're", "who'd", "who'll", "who's", "why'd", "why're", "why's", "would've", "wouldn't", "you'd",
                                "you'll", "you're", "you've"}
        #Initially tokenize on white space
        sentence = input.lower().split()
        tokenized = []
        # Additional tokenization rules
        for word in sentence:
            # Tokenize simple punctuation
            if "'" not in word and '.' not in word and '$' not in word:
                tokenized.extend(re.split('([^a-zA-Z0-9\-])', word))
            # Don't break up urls
            elif '.com' in word or '.org' in word or '.net' in word or '.gov' in word or '.co.' in word:
                tokenized.append(word)
            # Don't break up contractions, make "'s" one token and split other uses of apostrophes
            elif "'" in word:
                tokens = re.split('(\$?[0-9]*,?[0-9]*\.?[0-9]+|\.|[^a-zA-Z0-9\-\'])', word)
                for i in range(len(tokens)):
                    if tokens[i] in contractions:
                        tokens[i] = [tokens[i]]
                    else:
                        tokens[i] = re.split('(\'s|\$?[0-9]*,?[0-9]*\.?[0-9]+|\.|\'|[^a-zA-Z0-9\-])', tokens[i])
                tokenized.extend([token for sublist in tokens for token in sublist])
            else:
                # Tokenize other cases, keeping numbers with decimals and/or dollar signs as single tokens
                tokenized.extend(re.split('(\$?[0-9]*,?[0-9]*\.?[0-9]+|\.|\'|[^a-zA-Z0-9\-])', word))
        tokenized = [x for x in tokenized if x != '']
        return tokenized
        

if __name__ == '__main__':
    sc = SpellingAutocorrecter()
    #print(sc.tokenize("Is this, a test sen-tence?? $4.00 55 $100 3.14159 ... can't, won't, fidelity's 444'r"))
    #sc.correct_file('TruthFile_beta.csv', 'tf_sc_output.txt')
    #print(sc.bigrams)
    t1 = time.time()
    
    candidates = sc.correction(argv[1:])
    to_print = ' '.join(candidates)
    t1 = (time.time() - t1) * 1000
    print(to_print)
    print('Completed in ' + '{:0.2f}'.format(t1) + ' ms')
    