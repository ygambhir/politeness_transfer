from src.transformers import BertTokenizerFast, BertLMHeadModel, BertConfig, EncoderDecoderModel
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from torch import optim
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("bert-base-uncased")
config.is_decoder = True
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints
model.to(device)
for param in model.base_model.parameters():
	param.requires_grad = False
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# input_ids = torch.tensor(tokenizer.encode("What is all the name of the players, I'm still unsure. Who knows? ", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# i_ids = torch.tensor(tokenizer.encode("What is all the name of the players, I'm still not sure ", add_special_tokens=True)).unsqueeze(0) 
# batch_input = torch.cat([input_ids, input_ids, input_ids])
# print(batch_input.shape)
# optimizer.zero_grad()
# label_outputs = torch.tensor(tokenizer.encode("name ", add_special_tokens=True)).unsqueeze(0)
# outputs = model(input_ids=batch_input, decoder_input_ids=batch_input, labels=label_outputs, tokenizer=tokenizer)
# loss = outputs[0]
# print(outputs[0])
# loss = Variable(loss, requires_grad = True)
# loss_s = time.time()
# loss.backward()
# optimizer.step()
# print('loss and gradient update', time.time() - loss_s)


train_dataset = torch.load('open_subtitles_small_encoded.pt')
tloader = DataLoader(train_dataset, batch_size=128, shuffle=True)


def train_model(model, train_loader, optimizer, epochs):
	for epoch in range(epochs):
	    for batch in train_loader:
	        optimizer.zero_grad()
	        input_ids = batch['input_ids'].to(device)
	        attention_mask = batch['attention_mask'].to(device)
	        labels = 1
	        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
	        loss = outputs[0]
	        loss.backward()
	        optim.step()

train_model(model, tloader, optimizer, 3)