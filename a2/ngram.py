# CS421: Natural Language Processing
# University of Illinois Chicago
# Fall 2025
# Assignment 2
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================

import re



# Function to split a piece of text into sentences
# Arguments:
# text: A string containing input text
# Returns: sents (list)
# Where, sents (list) is a list of sentences that the text is split into
# Note that while this function works for most cases, it does not work in all scenarios;
# Can you think of any exceptions?
def get_sents(text):
	sents = re.split("[\n\.!?]", text)
	sents = list(filter(len, [s.strip() for s in sents]))
	return sents


# Function to split a sentence into a list of words
# Arguments:
# sent: A string containing input sentence
# Returns: words (list)
# Where, words (list) is a list of words that the sentence is split into
# Note that while this function works for most cases, it does not work in all scenarios;
# Can you think of any exceptions?
def get_words(sent):
	words = re.split(r"[^A-Za-z0-9-]", sent)
	words = list(filter(len, words))
	return words


# Function to get unigram counts
# Arguments:
# text: A string containing input text (may have multiple sentences)
# Returns: unigrams (dict)
# Where, unigrams (dict) is a python dictionary countaining lower case unigrams (words) as keys and counts as values
# Make sure that you convert all unigrams to lower case while counting e.g. "Police" and "police" are counted as the same unigram "police"
# You should use get_sents and get_words function to get sentences and words respectively, do not use any other tokenization mechanisms
def get_unigram_counts(text):
	counts = dict()

	# [WRITE YOUR CODE HERE]
	for sent in get_sents(text):
		for word in get_words(sent):
			word = word.lower()
			if word in counts:
				counts[word] += 1
			else:
				counts[word] = 1
	

	return counts


# Function to get bigram counts
# Arguments:
# text: A string containing input text (may have multiple sentences)
# Returns: bigrams (dict)
# Where, unigrams (dict) is a python dictionary countaining lower case bigrams as keys and counts as values.
# Bigram keys must be formatted as two words separated by an underscore character
# For example, the bigram "Red car" is represented as "red_car" and the "Dark Horse" as "dark_horse"
# You should also respect the sentence boundaries, for example in the following text:
# "Can't repeat the past?... Why of course you can!", past_why should not be a bigram since there is
# a sentence boundary between the two words.
# Make sure that you convert all bigrams to lower case while counting e.g. "RED Car" and "red car" are counted as the same bigram "red_car"
# You should use get_sents and get_words function to get sentences and words respectively, DO NOT use any other tokenization mechanisms
def get_bigram_counts(text):
	counts = dict()

	# [WRITE YOUR CODE HERE]
	for sent in get_sents(text):
		words = [w.lower() for w in get_words(sent)]
		for i in range(len(words) - 1):
			bigram = f"{words[i]}_{words[i+1]}"
			if bigram in counts:
				counts[bigram] += 1
			else:
				counts[bigram] = 1
	return counts


# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python ngram.py
# It should produce the following output (with correct solution):
#
# $ python3 ngram.py
# First sentence: Gatsby believed in the green light, the orgastic future that year by year recedes before us
# Second sentence: It eluded us then, but that’s no matter-tomorrow we will run faster, stretch out our arms farther
# Words in first sentence: ['Gatsby', 'believed', 'in', 'the', 'green', 'light', 'the', 'orgastic', 'future', 'that', 'year', 'by', 'year', 'recedes', 'before', 'us']
# Unigram counts: {'gatsby': 1, 'believed': 1, 'in': 1, 'the': 4, 'green': 1, 'light': 1, 'orgastic': 1, 'future': 1, 'that': 2, 'year': 2, 'by': 1, 'recedes': 1, 'before': 1, 'us': 2, 'it': 1, 'eluded': 1, 'then': 2, 'but': 1, 's': 1, 'no': 1, 'matter-tomorrow': 1, 'we': 2, 'will': 1, 'run': 1, 'faster': 1, 'stretch': 1, 'out': 1, 'our': 1, 'arms': 1, 'farther': 1, 'and': 1, 'one': 1, 'fine': 1, 'morning': 1, 'so': 1, 'beat': 1, 'on': 1, 'boats': 1, 'against': 1, 'current': 1, 'borne': 1, 'back': 1, 'ceaselessly': 1, 'into': 1, 'past': 1}
# Bigram counts: {'gatsby_believed': 1, 'believed_in': 1, 'in_the': 1, 'the_green': 1, 'green_light': 1, 'light_the': 1, 'the_orgastic': 1, 'orgastic_future': 1, 'future_that': 1, 'that_year': 1, 'year_by': 1, 'by_year': 1, 'year_recedes': 1, 'recedes_before': 1, 'before_us': 1, 'it_eluded': 1, 'eluded_us': 1, 'us_then': 1, 'then_but': 1, 'but_that': 1, 'that_s': 1, 's_no': 1, 'no_matter-tomorrow': 1, 'matter-tomorrow_we': 1, 'we_will': 1, 'will_run': 1, 'run_faster': 1, 'faster_stretch': 1, 'stretch_out': 1, 'out_our': 1, 'our_arms': 1, 'arms_farther': 1, 'and_then': 1, 'then_one': 1, 'one_fine': 1, 'fine_morning': 1, 'so_we': 1, 'we_beat': 1, 'beat_on': 1, 'on_boats': 1, 'boats_against': 1, 'against_the': 1, 'the_current': 1, 'current_borne': 1, 'borne_back': 1, 'back_ceaselessly': 1, 'ceaselessly_into': 1, 'into_the': 1, 'the_past': 1}

def main():
	# Given an input text
	text = """  Gatsby believed in the green light, the orgastic future that year by year recedes before us.
                It eluded us then, but that’s no matter-tomorrow we will run faster, stretch out our arms farther...
                And then one fine morning... So we beat on, boats against the current, borne back ceaselessly into the past."""
	
	# We can convert it into a list of sentences using get_sents function
	sents = get_sents(text)
	print("First sentence: {0}".format(sents[0]))
	print("Second sentence: {0}".format(sents[1]))

	# Any sentence can then be converted into a list of words using get_words function
	words = get_words(sents[0])
	print("Words in first sentence: {0}".format(words))

	# Get unigram counts as
	counts = get_unigram_counts(text)
	print("Unigram counts: {0}".format(counts))


	counts = get_bigram_counts(text)
	print("Bigram counts: {0}".format(counts))


	"""with open("callofthewild.txt", "r", encoding="utf-8") as f:
		COTW = f.read()
	with open("theodyssey.txt", "r", encoding="utf-8") as f:
		ODYSSEY = f.read()

	CotwUnigrams = get_unigram_counts(COTW)
	CotBigrams = get_bigram_counts(COTW)

	OdysseyUngrams = get_unigram_counts(ODYSSEY)
	OdysseyBigrams = get_bigram_counts(ODYSSEY)

	print(sorted(CotwUnigrams.items(), key=lambda x: -x[1])[:10])
	print(sorted(CotBigrams.items(), key=lambda x: -x[1])[:10])
	print(sorted(OdysseyUngrams.items(), key=lambda x: -x[1])[:10])
	print(sorted(OdysseyBigrams.items(), key=lambda x: -x[1])[:10])"""



	return 0


################ Do not make any changes below this line ################
if __name__ == '__main__':
	exit(main())
