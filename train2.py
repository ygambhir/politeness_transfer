from src.transformers import EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder, BertGenerationTokenizer
from rlFunctionsBatch import rlScore
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from torch import optim
import time
from playsound import playsound
import numpy as np
torch.cuda.empty_cache()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

encoder = BertGenerationEncoder.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
decoder = BertGenerationDecoder.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder", add_cross_attention=True, is_decoder=True)
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
# create tokenizer...
tokenizer = BertGenerationTokenizer.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")

model = model.to(device)
model.train()

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

def decodeBatch(ids, orig):
	current = np.array([])
	original = np.array([])
	p,q=ids.size()
	for t in range(p):
		np.append(current, tokenizer.decode(ids[t], skip_special_tokens=True))
		np.append(original, tokenizer.decode(orig[t], skip_special_tokens=True))
	return [current,original]

train_dataset = torch.load('open_subtitles_small_encoded_train_try_format.pt')
test_dataset = torch.load('open_subtitles_small_encoded_test_try_format.pt')

trainloader = DataLoader(PoliteDataset(train_dataset), batch_size=1, shuffle=True)
testloader = DataLoader(PoliteDataset(test_dataset), batch_size=1, shuffle=True)

def decodeBatchTestMap(ids, orig):
	current = np.array(list(map(lambda x: tokenizer.decode(torch.from_numpy(x), skip_special_tokens=True), ids.cpu().numpy())))
	original = np.array(list(map(lambda x: tokenizer.decode(torch.from_numpy(x), skip_special_tokens=True), orig.cpu().numpy())))
	return [current,original]

def train_model(model, train_loader, optimizer, epochs):
	file = open('logFirst.txt', 'w')
	for epoch in range(epochs):
		avgLoss = 0
		c=0
		for batch in trainloader:
			optimizer.zero_grad()
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			current, probs = model.generate(input_ids, do_sample=True,  max_length=50, decoder_start_token_id=model.config.decoder.bos_token_id, attention_mask=attention_mask)
			text = decodeBatchTestMap(current, input_ids)
			originalText=text[1]
			currentText = text[0]
			reward = rlScore(originalText, currentText)
			reward = torch.tensor(reward).to(device)
			loss = -torch.mean(torch.sum(torch.log(probs), dim=1))
			loss = Variable(loss, requires_grad = True)
			loss.backward()
			optimizer.step()
			avgLoss+=loss
			c+=1
			print(f'loss {loss}, avg loss {avgLoss/c}, exText: {currentText[0]}')
		print(f'Epoch {epoch} loss {loss}, avg loss {avgLoss/c}')
	    #playsound('epochComplete.mp3')
		file.write(f'Epoch {epoch} loss {loss}, avg loss {avgLoss/c}\n')
	model.save_pretrained('H:/School/fall2020/nlpdeeplearning/project/projHub/politeness_transfer/modelFinalTry')


def greedySample(input_ids, model, tokenizer):
	o = model.generate(input_ids, decoder_start_token_id=model.config.decoder.bos_token_id)
	print(tokenizer.decode(o[0], skip_special_tokens=True))

def testTest():
	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
	config = BertConfig.from_pretrained("bert-base-uncased")
	config.is_decoder = True
	model2 = EncoderDecoderModel.from_pretrained('H:/School/fall2020/nlpdeeplearning/project/projHub/politeness_transfer/model2')
	model2.to(device)
	input_ids = torch.tensor(tokenizer.encode("What is all the name of the players, I'm still unsure. Who knows? ", add_special_tokens=True)).unsqueeze(0) 
	input_ids = input_ids.to(device)
	model.generate(input_ids, decoder_start_token_id=model.config.decoder.bos_token)

def beamSample(input_ids, model, tokenizer):
	beam_output = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True,decoder_start_token_id=model.config.decoder.bos_token_id,no_repeat_ngram_size=2)
	print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

def nucleusSample(input_ids, model, tokenizer):
	sample_outputs = model.generate(input_ids,do_sample=True, max_length=50, top_k=50, top_p=0.95, num_return_sequences=3,decoder_start_token_id=model.config.decoder.bos_token_id)
	for i, sample_output in enumerate(sample_outputs):
		print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))



train_model(model, trainloader, optimizer, 15)

input_ids = torch.tensor(tokenizer.encode("What is all the name of the players, I'm still unsure. Who knows? ", add_special_tokens=True)).unsqueeze(0) 
input_ids = input_ids.to(device)
greedySample(input_ids, model, tokenizer)
beamSample(input_ids, model, tokenizer)
nucleusSample(input_ids, model, tokenizer)





