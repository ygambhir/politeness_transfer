from src.transformers import BertTokenizer, BertLMHeadModel, BertConfig, EncoderDecoderModel

import torch
from torch.autograd import Variable

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("bert-base-uncased")
config.is_decoder = True
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints
for param in model.base_model.parameters():
	param.requires_grad = False

input_ids = torch.tensor(tokenizer.encode("gender? male or ", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
batch_input = torch.cat([input_ids, input_ids])
print(batch_input.shape)
label_outputs = torch.tensor(tokenizer.encode("name ", add_special_tokens=True)).unsqueeze(0)
outputs = model(input_ids=batch_input, decoder_input_ids=input_ids, labels=label_outputs, tokenizer=tokenizer)
loss = outputs[0]
print(outputs[0])
loss = Variable(loss, requires_grad = True)
loss.backward()
print(loss)
