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
model.train()

for param in model.base_model.parameters():
	param.requires_grad = False
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

strings = ["Man you're mean and awful! I can't wait to get rid of this plan", "Who is the"]

encoded = tokenizer.batch_encode_plus(strings, pad_to_max_length=True)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)
input_ids = torch.tensor(tokenizer.encode("What is all the name of the players, I'm still unsure. Who knows? ", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
i_ids = torch.tensor(tokenizer.encode("What is all the name of the players, I'm still not sure ", add_special_tokens=True)).unsqueeze(0) 
batch_input = torch.cat([input_ids, input_ids, input_ids])

optimizer.zero_grad()
label_outputs = torch.tensor(tokenizer.encode("name ", add_special_tokens=True)).unsqueeze(0)
outputs = model(input_ids=torch.tensor(encoded['input_ids']), attention_mask=torch.tensor(encoded['attention_mask']), decoder_input_ids=torch.tensor(encoded['input_ids']), decoder_attention_mask=torch.tensor(encoded['attention_mask']), labels=label_outputs, tokenizer=tokenizer)
loss = outputs[0]
print(loss)

loss = Variable(loss, requires_grad = True)
loss_s = time.time()
loss.backward()
optimizer.step()
print('loss and gradient update', time.time() - loss_s)

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
	    for batch in train_loader:
	        optimizer.zero_grad()
	        input_ids = batch['input_ids'].to(device)
	        attention_mask = batch['attention_mask'].to(device)
	        labels = 1
	        outputs = model(input_ids, decoder_input_ids=input_ids, decoder_attention_mask=attention_mask, tokenizer=tokenizer, attention_mask=attention_mask, labels=labels)
	        loss = outputs[0]
	        print(f'Loss: {loss}')
	        loss = Variable(loss, requires_grad = True)
	        loss.backward()
	        optimizer.step()
	    file.write(f'Epoch {epoch} loss {loss}')

# def test_model(model, train_loader, optimizer, epochs):
# 	for epoch in range(epochs):
# 	    for batch in train_loader:
# 	        optimizer.zero_grad()
# 	        input_ids = batch['input_ids'].to(device)
# 	        attention_mask = batch['attention_mask'].to(device)
# 	        labels = 1
# 	        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
# 	        loss = outputs[0]

train_model(model, trainloader, optimizer, 10)
# test_model()