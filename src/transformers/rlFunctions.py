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

#######################################################################
# LOAD AND TRAIN POLITE CLASSIFIER
#######################################################################

WIKI_CORPUS = Corpus(download("wikipedia-politeness-corpus"))
parser = TextParser(verbosity=1000)
WIKI_CORPUS = parser.transform(WIKI_CORPUS)
PS = PolitenessStrategies()
WIKI_CORPUS = PS.transform(WIKI_CORPUS, markers=True)
BINARY_CORPUS = Corpus(utterances=[utt for utt in WIKI_CORPUS.iter_utterances() if utt.meta["Binary"] != 0])
TEST_IDS = BINARY_CORPUS.get_utterance_ids()[-435:]
TRAIN_CORPUS = Corpus(utterances=[utt for utt in BINARY_CORPUS.iter_utterances() if utt.id not in TEST_IDS])
TEST_CORPUS = Corpus(utterances=[utt for utt in BINARY_CORPUS.iter_utterances() if utt.id in TEST_IDS])
CLF = Classifier(obj_type="utterance",pred_feats=["politeness_strategies"], labeller=lambda utt: utt.meta['Binary'] == 1)
CLF.fit(TRAIN_CORPUS)
TEST_PRED = clf.transform(TEST_CORPUS)
SP = spacy.load('en', disable=['ner'])
PRED2LABEL = {1: "polite", 0: "impolite"}
CLF.accuracy(TEST_CORPUS)

#######################################################################
# LOAD GPT2 FOR PERPLEXITY CALCULATION
#######################################################################

device = 'cpu'
model_id = 'gpt2-medium'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
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
	polite = predictText(current, clf, ps, SP)
	similarity = pairwiseSimilarity(original, current)
	perplexity = perplexityCalc(current)
	return [polite[1], similarity, perplexity]

while True:
	o = input("The data was")
	over = input("The data was")
	print(rlScore(o, over))


