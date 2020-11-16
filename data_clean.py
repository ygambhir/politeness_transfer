from better_profanity import profanity
import time


from convokit import Corpus, download


SPECIAL_CHARACTERS = ["-", "~", '/', '...']

def data_clean(file):
	f_out = open('open_subtitles_small_clean.txt', 'w+')
	with open(file, 'r') as f:
		count = 0
		lines = f.readlines()
		start = time.time()
		for line in lines:
			if profanity.contains_profanity(line):
				count += 1
			else:
				for char in SPECIAL_CHARACTERS:
					line = line.replace(char, '')
				tokens = line.split(' ')
				if len(tokens) < 2:
					count += 1
					continue
				f_out.write(line.lower())
		print(time.time() - start)
		print(count)


data_clean('open_subtitles_small.txt')



def process_reddit_corpus():
	corpus = Corpus(filename=download("reddit-corpus-small"))
	with open('reddit-corpus-small.txt', 'w') as f:
		for utt in corpus.iter_utterances():
			f.write(utt.text)


#process_reddit_corpus()
