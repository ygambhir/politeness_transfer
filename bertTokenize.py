from transformers import BertTokenizer
import torch
import pickle
from sklearn.model_selection import train_test_split

device = 'cuda'

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

f = open('open_subtitles_small_clean.txt', 'r')
lines = f.readlines()
l = [x.replace('\n','') for x in lines]

totalSize = len(l)
testSize = int(totalSize*.05)
train_dataset, test_dataset = train_test_split(l, test_size=testSize, random_state=42)

tokens_tensor_train = tokenizer(train_dataset, add_special_tokens=True, padding=True, return_tensors='pt')
tokens_tensor_test = tokenizer(test_dataset,  add_special_tokens=True, padding=True, return_tensors='pt')

torch.save(tokens_tensor_train, 'open_subtitles_small_encoded_train_generation_2.pt')
torch.save(tokens_tensor_test, 'open_subtitles_small_encoded_test_generation_2.pt')
print(tokens_tensor_train.input_ids)
#tokenized_text = tokenizer.tokenize(text)


# tokens_tensor_train = tokenizer(train_dataset,padding=True, truncation=True)
# tokens_tensor_test = tokenizer(test_dataset,padding=True, truncation=True)

# torch.save(tokens_tensor_train, 'open_subtitles_small_encoded_train.pt')
# torch.save(tokens_tensor_test, 'open_subtitles_small_encoded_test.pt')
# #tokenized_text = tokenizer.tokenize(text)
# tokenized_list = [tokenizer.tokenize(x) for x in l]

# #indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# indexed_list = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_list]

# #tokens_tensor = torch.tensor([indexed_tokens])
# tokens_tensor = [torch.tensor(x) for x in indexed_list]

# tokens_tensor = tokenizer(l,padding=True)

# torch.save(tokens_tensor, 'open_subtitles_small_encoded.pt')

#torch.load('open_subtitles_small_encoded.pt')