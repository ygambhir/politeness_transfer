from transformers import BertGenerationTokenizer, BertLMHeadModel, BertGenerationConfig, EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder, BertTokenizer
# from rlFunctionsBatch import rlScore
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from torch import optim
import time
from playsound import playsound
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")


class PoliteDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


train_dataset = torch.load('open_subtitles_small_encoded_train_generation.pt')
test_dataset = torch.load('open_subtitles_small_encoded_test_generation.pt')

trainloader = DataLoader(PoliteDataset(train_dataset), batch_size=40, shuffle=True)
testloader = DataLoader(PoliteDataset(test_dataset), batch_size=40, shuffle=True)

def hybridTry(ids, orig):
	current = np.array([])
	original = np.array([])
	p,q=ids.size()
	for t in range(p):
		np.append(current, tokenizer.decode(ids[t], skip_special_tokens=True))
		np.append(original, tokenizer.decode(orig[t], skip_special_tokens=True))

def classicTry(ids, orig):
	current = []
	original = []
	p,q=ids.size()
	for t in range(p):
		current.append(tokenizer.decode(ids[t], skip_special_tokens=True))
		original.append(tokenizer.decode(orig[t], skip_special_tokens=True))

def classicTry(ids, orig):
	current = []
	original = []
	p,q=ids.size()
	for t in range(p):
		current.append(tokenizer.decode(ids[t], skip_special_tokens=True))
		original.append(tokenizer.decode(orig[t], skip_special_tokens=True))

for batch in trainloader:
	input_ids = batch['input_ids'].to(device)

	t0=time.time()
	hybridTry(input_ids, input_ids)
	print(time.time()-t0)

	t1=time.time()
	classicTry(input_ids, input_ids)
	print(time.time()-t1)
	
	break
