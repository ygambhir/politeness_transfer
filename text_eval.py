from src.transformers import GPT2LMHeadModel, GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')


from rlFunctionsBatch import *

def compare_gpt2(model, pre_trainedGPT2, input_ids=None, input_strings=None, verbose=False):
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

	our_output = model.generate(input_ids, max_length=30, do_sample=True, topk=10)
	gpt2_output = pre_trainedGPT2.generate(input_ids, max_length=30, do_sample=True, topk=10)
	decoded = []
	gpt2_decoded = []
	for i in range(input_ids.shape[0]):
		decoded.append(tokenizer.decode(list(our_output[i]), skip_special_tokens=True))
		gpt2_decoded.append(tokenizer.decode(list(gpt2_output[i]), skip_special_tokens=True))

	our_politeness = predictTextBatch(decoded, clf, ps, sp)
	gpt2_politeness = predictTextBatch(gpt2_decoded, clf, ps, sp)

	if verbose is False:
		return our_politeness, gpt2_politeness

	for i in range(input_ids.shape[0]):
		print('Our output: ', decoded[i])
		print('Politeness: ', our_politeness[i])
		print('GPT2 output: ', gpt2_decoded[i])
		print('Politeness: ', gpt2_politeness[i])



if __name__ == '__main__':
	ex_string = ['Do not do that']
	model_id = 'gpt2'
	model = GPT2LMHeadModel.from_pretrained(model_id)
	compare_gpt2(model, model, input_strings=ex_string, verbose=True)

