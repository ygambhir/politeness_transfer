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
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from tqdm import tqdm
import torch
from fuzzywuzzy import fuzz
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

device = 'cpu'
model_id = 'gpt2-medium'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
tokenizer.pad_token=tokenizer.eos_token

#######################################################################
# CALCULATION FUNCTIONS
#######################################################################

def pairwiseSimilarity(original, output):
	vect = TfidfVectorizer(min_df=1)   
	tfidf = vect.fit_transform([original, output])
	pairwise_similarity = tfidf * tfidf.T 
	pairwise_similarity = pairwise_similarity.A
	return pairwise_similarity[0,1]

def pairwiseSimilarityBatch(original, output):
	q = np.array([original, output])
	ret = []
	for i in range(len(original)):
		ret.append(fuzz.ratio(q[0,i], q[1,i]))
	return np.array(ret)


def predictText(txt, model, politenessStrategies, spacyL):
	utt = Utterance(text=txt)
	utt = politenessStrategies.transform_utterance(utt, spacy_nlp = spacyL, markers=True)
	pred = model.transform_objs([utt])
	return [pred[0].meta['prediction'], pred[0].meta['pred_score']]

def predictTextBatch(txt, model, politenessStrategies, spacyL):
	utt = []
	for i in range(len(txt)):
		utt.append(Utterance(text=txt[i],id=str(i), speaker=Speaker()))
	utt2 = parser.transform(Corpus(utterances=utt))
	utt3 = politenessStrategies.transform(utt2, markers=True)
	pred = model.transform(utt3)
	predScore = [x.meta['pred_score'] for x in pred.iter_utterances()]

	return np.array(predScore)


def perplexityCalc(current):
	test = current
	encodings = tokenizer(test.tolist(), return_tensors='pt', padding=True)
	input_ids = encodings.input_ids.to(device)
	mask = encodings.attention_mask.to(device)

	with torch.no_grad():
		outputs = model(input_ids, labels=input_ids, attention_mask = mask)
		log_likelihood = outputs[0]
	ppl = torch.exp(log_likelihood)

	return ppl.item()

def perplexityCalcBatch(current):
	test = current
	results = []
	encodings = tokenizer(test, return_tensors='pt', padding=True)
	for i in range(len(current)):
		input_ids = encodings.input_ids[i].to(device)
		mask = encodings.attention_mask[i].to(device)

		with torch.no_grad():
			outputs = model(input_ids, labels=input_ids, attention_mask = mask)
			log_likelihood = outputs[0]
		ppl = torch.exp(log_likelihood)
		results.append(ppl.item())

	return np.array(results)


#######################################################################
# RL REWARD FUNCTION
#######################################################################

def rlScore(original, current):
	w_pol = 1000
	w_similarity = 1/100
	w_lm = .1
	t0=time.time()
	polite = predictTextBatch(current.tolist(), clf, ps, sp)
	similarity = pairwiseSimilarityBatch(original, current)
	perplexity = perplexityCalcBatch(current.tolist())
	print(time.time()-t0)
	return polite + w_similarity*similarity + 100/perplexity

def test():
	one = np.array(['Hello from my side I must have called two thousand times','Pee pee oop lol poo poo','I just want to sip till the pain wears off', 'Hello world hello world hello world hello world'])	
	two = np.array(['Hello from the other side I must have called one thousand times.','Pee pee oop lol poo poo','My name is Jim Harrison and welcome to my shop', 'Thank you'])
	print(rlScore(one, two))

test()