import numpy as np
import torch
from torch import optim
from rlFunctionsBatch import rlScore
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import GPT2Config,GPT2LMHeadModel,GPT2Tokenizer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)

for param in model.base_model.parameters():
	param.requires_grad = False
optimizer = optim.AdamW(model.parameters(), lr=3e-4)



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

###########################################################

def decodeBatch(ids, orig, beginning):
	current = np.array([])
	original = np.array([])
	p,q=ids.size()
	for t in range(p):
		c = tokenizer.decode(ids[t][beginning:], clean_up_tokenization_spaces=True, skip_special_tokens=True)
		original = np.append(original, tokenizer.decode(orig[t], clean_up_tokenization_spaces=True, skip_special_tokens=True))
		if c[0:2] == '\n\n':
			c = c[2:]
			idx = c.find('\n\n')
			if idx!=-1:
				c=c[:idx]
		else:
			idx = c.find('\n\n')
			if idx!=-1:
				c=c[:idx]
		current = np.append(current, c)
	return (current,original)

def getScore(model, tokenizer, input_ids, mask, pr):
	beginning = input_ids[0].size()[0]
	sample = model.generate(input_ids, attention_mask=mask, max_length=beginning+50)
	current_text, original_text = decodeBatch(sample, input_ids, beginning)
	if pr:
		print(f'{original_text[0]}:\n {current_text[0]}')
	reward = rlScore(original_text, current_text).mean()
	reward = torch.tensor(reward).to(device)
	return reward

###########################################################
file = open('logFirstPunt.txt', 'w')
for epoch in range(15):
	avgLoss = 0
	c=0
	pr=True
	for batch in trainloader:
		optimizer.zero_grad()
		input_ids = batch['input_ids'].to(device)
		mask = batch['attention_mask'].to(device)
		output = model(input_ids, attention_mask=mask, labels=input_ids)
		loss=output[0]
		reward = getScore(model, tokenizer, input_ids, mask, pr)
		pr=False
		# print(loss)
		# print(reward)
		loss = loss+reward
		loss = Variable(loss, requires_grad = True)
		loss.backward()
		optimizer.step()
		avgLoss+=loss
		c+=1
		# print(f'loss {loss}, avg loss {avgLoss/c}, reward {reward}')
	print(f'Epoch {epoch} loss {loss}, avg loss {avgLoss/c}, reward {reward}')
	file.write(f'Epoch {epoch} loss {loss}, avg loss {avgLoss/c}, reward {reward}\n')
model.save_pretrained('H:/School/fall2020/nlpdeeplearning/project/projHub/politeness_transfer/modelPuntOne')