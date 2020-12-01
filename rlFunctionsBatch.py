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
print("about to gpt2")
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
print('done')
from tqdm import tqdm
import torch
from fuzzywuzzy import fuzz
import time
from torch.nn import CrossEntropyLoss


#######################################################################
# LOAD AND TRAIN POLITE CLASSIFIER
#######################################################################
print('Imports done')
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
sp = spacy.load('en_core_web_lg', disable=['ner'])
pred2label = {1: "polite", 0: "impolite"}
clf.accuracy(test_corpus)

#######################################################################
# LOAD GPT2 FOR PERPLEXITY CALCULATION
#######################################################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_id = 'gpt2-medium'
model = GPT2LMHeadModel.from_pretrained(model_id, return_dict=True).to(device)
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
	encodings = tokenizer(test, return_tensors='pt', padding=True)
	input_ids = encodings.input_ids.to(device)
	mask = encodings.attention_mask.to(device)

	# with torch.no_grad():
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

def perplexityCalcBatchTest(current):
	test = current#['Hello from my side I must have called two thousand times','Pee pee oop lol poo poo','I just want to sip till the pain wears off', 'Hello world hello world hello world hello world','Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.']
	results = []
	encodings = tokenizer(test, return_tensors='pt', padding=True)
	input_ids = encodings.input_ids.to(device)
	mask = encodings.attention_mask.to(device)
	with torch.no_grad():
		loss_fct = CrossEntropyLoss(reduction='none')
		outputs = model(input_ids, attention_mask = mask)
		loss= loss_fct(outputs.logits.view(-1, outputs.logits.size(-1), outputs.logits.size(-2)), input_ids)
		loss2 = loss.mean(axis=1)

	return loss2.to(torch.device('cpu')).numpy()


#######################################################################
# RL REWARD FUNCTION
#######################################################################

def rlScore(original, current):
	w_pol = 1000
	w_similarity = 1/100
	w_lm = .1
	polite = predictTextBatch(current.tolist(), clf, ps, sp)
	similarity = pairwiseSimilarityBatch(original, current)
	perplexity = perplexityCalc(current.tolist())
	# print(f'Reward polite: {polite}, sim: {similarity}, perplexity:{perplexity}, total: {polite + w_similarity*similarity + 100/perplexity}')
	return 1/(polite) + 1/(w_similarity*similarity) + 1/perplexity

def rlScoreTest(original, current):
	w_pol = 1000
	w_similarity = 1/100
	w_lm = .1
	t0=time.time()
	polite = predictTextBatch(current.tolist(), clf, ps, sp)
	print(time.time()-t0)
	t1=time.time()
	similarity = pairwiseSimilarityBatch(original, current)
	print(time.time()-t1)
	t2=time.time()
	perplexity = perplexityCalcBatch(current.tolist())
	print(time.time()-t2)
	t3=time.time()
	perplexity = perplexityCalcBatchTest(current.tolist())
	print(time.time()-t3)
	print(f'Reward {polite + w_similarity*similarity + 100/perplexity}')
	return polite + w_similarity*similarity + 100/perplexity

def test():
	one = np.array(['Hello from my side I must have called two thousand times','Pee pee oop lol poo poo','I just want to sip till the pain wears off', 'Hello world hello world hello world hello world'])	
	two = np.array(['Hello from the other side I must have called one thousand times.','Pee pee oop lol poo poo','My name is Jim Harrison and welcome to my shop', 'Thank you'])
	print(rlScoreTest(one, two))

# test()