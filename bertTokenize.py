from transformers import BertTokenizer
import torch
import pickle

device = 'cuda'
model_id = 'bert_uncased'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

f = open('open_subtitles_small_clean.txt', 'r')
lines = f.readlines()
l = [x.replace('\n','') for x in lines]

#tokenized_text = tokenizer.tokenize(text)
# tokenized_list = [tokenizer.tokenize(x) for x in l]

# #indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# indexed_list = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_list]

# #tokens_tensor = torch.tensor([indexed_tokens])
# tokens_tensor = [torch.tensor(x) for x in indexed_list]

tokens_tensor = tokenizer(l,padding=True)

torch.save(tokens_tensor, 'open_subtitles_small_encoded.pt')

#torch.load('open_subtitles_small_encoded.pt')