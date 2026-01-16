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


# Function to calculate perplexity
# Arguments:
# probs: A dictionary holding your toy language model, with keys (tokens) and values (probabilities)
# sentence: A string containing tokens from your language model's vocabulary
# Returns: perplexity (float)
# Where perplexity is the floating point perplexity for the sentence given the language model
def perplexity(probs, sentence):
	perplexity = 0.0

	# [WRITE YOUR CODE HERE]
	if not isinstance(sentence, str) or len(sentence) == 0:
		raise ValueError('Empty sentence')
	
	tokens = sentence.split()
	T = len(tokens)

	logSum = 1.0  
	#Calculate probabilities product
	for token in tokens:
		if token not in probs:
			raise ValueError('Token not in vocab')
		p = probs[token]
		if p <= 0.0:
			raise ValueError('Zero probability')
		logSum *= float(p)

	perplexity = float(logSum ** (-1.0 / T))


	return perplexity



# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python perplexity.py
# It should produce the following output (with correct solution) given the example input:
#
# $ python3 perplexity.py
# Perplexity: 4.52

def main():
	# Given a dictionary holding the language model probabilities and a test sentence:
	probs = {"cs": 0.4, "421": 0.3, "is": 0.2, "fun": 0.1}
	sentence = "cs 421 is fun"
	
	# We can calculate the perplexity
	pp = perplexity(probs, sentence)
	print("Perplexity: {0}".format(pp))

	return 0


################ Do not make any changes below this line ################
if __name__ == '__main__':
	exit(main())
