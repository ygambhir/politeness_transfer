from transformers import BertGenerationTokenizer, BertLMHeadModel, BertGenerationConfig, EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder, BertTokenizer
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from torch import optim
import time
from playsound import playsound

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

encoder = BertGenerationEncoder.from_pretrained("bert-large-uncased", bos_token_id=101, eos_token_id=102)
decoder = BertGenerationDecoder.from_pretrained("bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)
# create tokenizer...
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

input_ids = tokenizer('This is a long article to summarize', add_special_tokens=True, return_tensors="pt").input_ids

tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
config = BertGenerationConfig.from_pretrained("bert-base-uncased")
config.is_decoder = True
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints
model = model.to(device)
model.train()
# model.config.encoder.pad_token_id=-100
# model.config.decoder.pad_token_id=-100

for param in model.base_model.parameters():
	param.requires_grad = False
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# strings = ["Man you're mean and awful! I can't wait to get rid of this plan", "Who is the"]

# encoded = tokenizer.batch_encode_plus(strings, pad_to_max_length=True)

# optimizer = optim.AdamW(model.parameters(), lr=5e-5)
# input_ids = torch.tensor(tokenizer.encode("What is all the name of the players, I'm still unsure. Who knows? ", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# i_ids = torch.tensor(tokenizer.encode("What is all the name of the players, I'm still not sure ", add_special_tokens=True)).unsqueeze(0) 
# batch_input = torch.cat([input_ids, input_ids, input_ids])

# optimizer.zero_grad()
# label_outputs = torch.tensor(tokenizer.encode("name ", add_special_tokens=True)).unsqueeze(0)
# outputs = model(input_ids=torch.tensor(encoded['input_ids']), attention_mask=torch.tensor(encoded['attention_mask']), decoder_input_ids=torch.tensor(encoded['input_ids']), decoder_attention_mask=torch.tensor(encoded['attention_mask']), labels=label_outputs, tokenizer=tokenizer)
# loss = outputs[0]
# print(loss)

# loss = Variable(loss, requires_grad = True)
# loss_s = time.time()
# loss.backward()
# optimizer.step()
# print('loss and gradient update', time.time() - loss_s)

class PoliteDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


train_dataset = torch.load('open_subtitles_small_encoded_train.pt')
test_dataset = torch.load('open_subtitles_small_encoded_test.pt')

trainloader = DataLoader(PoliteDataset(train_dataset), batch_size=40, shuffle=True)
testloader = DataLoader(PoliteDataset(test_dataset), batch_size=40, shuffle=True)

def train_model(model, train_loader, optimizer, epochs):
	file = open('logFirst.txt', 'w')
	for epoch in range(epochs):
	    for batch in trainloader:
	        optimizer.zero_grad()
	        input_ids = batch['input_ids'].to(device)
	        attention_mask = batch['attention_mask'].to(device)
	        labels = input_ids.clone()
	        labels[labels==0]=-100
	        labels[labels!=-100]=1
	        outputs = model(input_ids, decoder_input_ids=input_ids, decoder_attention_mask=attention_mask, tokenizer=tokenizer, attention_mask=attention_mask, labels=input_ids)
	        curent, probs = model.generate(input_ids, decoder_start_token_id=model.config.decoder.bos_token)
	        original = tokenizer.decode(o[0], skip_special_tokens=True)
	        loss2 = rlReward(current, original)
	        loss = outputs[0]*loss2
	        loss = Variable(loss, requires_grad = True)
	        loss.backward()
	        optimizer.step()
	    print(f'Loss: {loss}')
	    file.write(f'Epoch {epoch} loss {loss}')
	    #playsound('epochComplete.mp3')
	model.save_pretrained('H:/School/fall2020/nlpdeeplearning/project/projHub/politeness_transfer/model2')

# def test_model(model, train_loader, optimizer, epochs):
# 	for epoch in range(epochs):
# 	    for batch in train_loader:
# 	        optimizer.zero_grad()
# 	        input_ids = batch['input_ids'].to(device)
# 	        attention_mask = batch['attention_mask'].to(device)
# 	        labels = 1
# 	        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
# 	        loss = outputs[0]
def greedySample(input_ids, model, tokenizer):
	o = model.generate(input_ids, decoder_start_token_id=model.config.decoder.bos_token)
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
	beam_output = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True,decoder_start_token_id=model.config.decoder.pad_token_id,no_repeat_ngram_size=2)
	print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

def nucleusSample(input_ids, model, tokenizer):
	sample_outputs = model.generate(input_ids,do_sample=True, max_length=50, top_k=50, top_p=0.95, num_return_sequences=3,decoder_start_token_id=model.config.decoder.bos_token)
	for i, sample_output in enumerate(sample_outputs):
		print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))



train_model(model, trainloader, optimizer, 1)

input_ids = torch.tensor(tokenizer.encode("What is all the name of the players, I'm still unsure. Who knows? ", add_special_tokens=True)).unsqueeze(0) 
input_ids = input_ids.to(device)
greedySample(input_ids, model, tokenizer)
beamSample(input_ids, model, tokenizer)
nucleusSample(input_ids, model, tokenizer)





