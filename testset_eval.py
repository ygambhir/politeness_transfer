import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from src.transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


from rlFunctionsBatch import *

def compare_gpt2(model, pre_trainedGPT2, input_ids=None, input_strings=None, verbose=False, mask=None):
	'''
	model: Our pre-trained model
	pre_trainedGPT2: Pretrained GPT2 model
	input_data: List of input_ids that we want to compare
	input_ids: Ids of the input--provide for testing batches from test set, not original string
	input_strings: List of strings for testing with our input or decoded strings
	verbose: If False, simply returns two lists of politeness scores (our's compared to GPT2)
	'''
	if input_strings is not None:
		encodings = tokenizer(input_strings, return_tensors='pt', padding=True)
		input_ids = encodings.input_ids
		mask = encodings.attention_mask
	else:
		input_strings = []
		for i in range(len(input_ids)):
			input_strings.append(tokenizer.decode(input_ids[i], clean_up_tokenization_spaces=True, skip_special_tokens=True))
	beginning = input_ids[0].size()[0]
	our_output = model.generate(input_ids, attention_mask=mask, max_length=beginning+30, do_sample=True, topk=10)
	gpt2_output = pre_trainedGPT2.generate(input_ids, attention_mask=mask, max_length=beginning+30, do_sample=True, topk=10)
	decoded = []
	gpt2_decoded = []
	similarity = []
	for i in range(input_ids.shape[0]):
		our_string = tokenizer.decode(our_output[i][beginning:], clean_up_tokenization_spaces=True, skip_special_tokens=True)
		gpt_string = tokenizer.decode(gpt2_output[i], clean_up_tokenization_spaces=True, skip_special_tokens=True)
		decoded.append(our_string)
		gpt2_decoded.append(gpt_string)


	our_politeness = predictTextBatch(decoded, clf, ps, sp)
	gpt2_politeness = predictTextBatch(gpt2_decoded, clf, ps, sp)
	similarity = pairwiseSimilarityBatch(input_strings, decoded)

	perp = perplexityCalcBatchTest(decoded)

	if verbose is False:
		print({"average_delta_politeness" : (our_politeness-gpt2_politeness).mean(),
				"average_similarity" : similarity.mean(),
				"average_perplexity" : perp.mean()
		})
		return (our_politeness-gpt2_politeness).sum(), similarity.sum(), perp.sum()

	for i in range(input_ids.shape[0]):
		print('Our output: ', decoded[i])
		print('Politeness: ', our_politeness[i])
		print('GPT2 output: ', gpt2_decoded[i])
		print('Politeness: ', gpt2_politeness[i])



if __name__ == '__main__':
	ex_string = 'Do not do that'
	input_ids = tokenizer(ex_string, return_tensors='pt', padding=True)
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	config = GPT2Config.from_pretrained('gpt2')
	model = GPT2LMHeadModel.from_pretrained('gpt2')

	class PoliteDataset(torch.utils.data.Dataset):
	    def __init__(self, encodings):
	        self.encodings = encodings

	    def __getitem__(self, idx):
	        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
	        return item

	    def __len__(self):
	        return len(self.encodings.input_ids)

	###########################################################
	tokenizer.pad_token = tokenizer.eos_token
	# f = open('open_subtitles_small_clean.txt', 'r')
	f = open('rude_data.txt', 'r', encoding='UTF-8')
	lines = f.readlines()
	l = [x.replace('\n','') for x in lines]

	totalSize = len(l)
	testSize = int(totalSize*.05)
	train_dataset, test_dataset = train_test_split(l, test_size=testSize, random_state=42)

	tokens_tensor_train = tokenizer(train_dataset, add_special_tokens=False, padding=True, return_tensors='pt')
	tokens_tensor_test = tokenizer(test_dataset,  add_special_tokens=False, padding=True, return_tensors='pt')

	trainloader = DataLoader(PoliteDataset(tokens_tensor_train), batch_size=20, shuffle=True)
	testloader = DataLoader(PoliteDataset(tokens_tensor_test), batch_size=20, shuffle=True)
	sim = 0
	pol_delta = 0
	perp = 0
	for batch in testloader:

		input_ids = batch['input_ids'].to(device)
		mask = batch['attention_mask'].to(device)

		pol, s, p = compare_gpt2(model, model, input_ids=input_ids, mask=mask, verbose=False)
		perp += p
		pol_delta += pol
		sim += s