from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import convokit
from convokit import Corpus, Speaker, Utterance, TextParser, PolitenessStrategies, download, Classifier
from pandas import DataFrame
from typing import List, Dict, Set
import random
from sklearn import svm
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report
import spacy
import time
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
import torch
import time
#######################################################################
# LOAD AND TRAIN POLITE CLASSIFIER
#######################################################################

wiki_corpus = Corpus(download("wikipedia-politeness-corpus"))
parser = TextParser(verbosity=1000)
wiki_corpus = parser.transform(wiki_corpus)
ps = PolitenessStrategies()
wiki_corpus = ps.transform(wiki_corpus, markers=True)
binary_corpus = Corpus(utterances=[utt for utt in wiki_corpus.iter_utterances() if utt.meta["Binary"] != 0])
test_ids = binary_corpus.get_utterance_ids()[-435:]
train_corpus = Corpus(utterances=[utt for utt in binary_corpus.iter_utterances() if utt.id not in test_ids])
test_corpus = Corpus(utterances=[utt for utt in binary_corpus.iter_utterances() if utt.id in test_ids])
print("train size = {}, test size = {}".format(len(train_corpus.get_utterance_ids()),len(test_corpus.get_utterance_ids())))
clf = Classifier(obj_type="utterance",pred_feats=["politeness_strategies"], labeller=lambda utt: utt.meta['Binary'] == 1)
clf.fit(train_corpus)
test_pred = clf.transform(test_corpus)
sp = spacy.load('en', disable=['ner'])
pred2label = {1: "polite", 0: "impolite"}
clf.accuracy(test_corpus)

#######################################################################
# LOAD GPT2 FOR PERPLEXITY CALCULATION
#######################################################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_id = 'gpt2-medium'
model = GPT2LMHeadModel.from_pretrained(model_id, return_dict=True).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

#######################################################################
# CALCULATION FUNCTIONS
#######################################################################

def pairwiseSimilarity(original, output):
	vect = TfidfVectorizer(min_df=1, stop_words="english")   
	tfidf = vect.fit_transform([original, output])
	pairwise_similarity = tfidf * tfidf.T 
	pairwise_similarity = pairwise_similarity.A
	return pairwise_similarity[0,1]


def predictText(txt, model, politenessStrategies, spacyL):
	utt = Utterance(text=txt)
	utt = politenessStrategies.transform_utterance(utt, spacy_nlp = spacyL, markers=True)
	pred = model.transform_objs([utt])
	return [pred[0].meta['prediction'], pred[0].meta['pred_score']]


def perplexityCalc(current):
	test = current
	encodings = tokenizer(test, return_tensors='pt')

	input_ids = encodings.input_ids.to(device)
	target_ids = input_ids.clone()

	with torch.no_grad():
		outputs = model(input_ids, labels=target_ids)
		log_likelihood = outputs[0]
	ppl = torch.exp(log_likelihood)
	return ppl.item()


#######################################################################
# RL REWARD FUNCTION
#######################################################################

def rlScore(original, current):
	w_pol = 1000
	w_similarity = 100
	w_lm = .1
	s = time.time()
	polite = predictText(current, clf, ps, sp)
	p = time.time()
	print('politeness classifier', time.time() - s)
	similarity = pairwiseSimilarity(original, current)
	sim = time.time()
	print('sim time', time.time() - p)
	perplexity = perplexityCalc(current)
	print('perplexity time', time.time() - sim)
	return w_pol*polite + w_similarity*similarity + w_lm*perplexity*-1
