# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2025
# Assignment 3
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Function: get_pos_tags(user_input)
# user_input: A string of arbitrary length
# Returns: A list of (token, POS) tuples
#
# This function tags each token in the user_input with a Part of Speech (POS) tag from Penn Treebank.
def get_pos_tags(user_input):
    tagged_input = []

    # [WRITE YOUR CODE HERE]
    tokens = nltk.word_tokenize(user_input)
    tagged_input = nltk.pos_tag(tokens)
    
    return tagged_input


# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python pos_tagger.py
# It should produce the following output (with correct solution):
#
# $ python3 pos_tagger.py
# Input: Time flies like an arrow; fruit flies like a banana.
# POS Tags: [('Time', 'NNP'), ('flies', 'NNS'), ('like', 'IN'), ('an', 'DT'), ('arrow', 'NN'), (';', ':'),
# ('fruit', 'CC'), ('flies', 'NNS'), ('like', 'IN'), ('a', 'DT'), ('banana', 'NN'), ('.', '.')]
def main():
    input = "I like pizza but not before I go for a swim"
    pos_tags = get_pos_tags(input)
    print("Input: {0}\nPOS Tags: {1}".format(input, pos_tags))


    tests = [
       "I enjoy swimming more than running.",  
       "Please rake the leaves before I come from work.", 
       "They will play soccer after lunch.",
         "The tree leaves have fallen so fall is here.",
         "Skyfall is much better than Spectre."
    ]

    for i, sent in enumerate(tests, 1):
        print(f"Input {i}:", sent)
        print("NLTK:", get_pos_tags(sent), "\n")

################ Do not make any changes below this line ################
if __name__ == '__main__':
	exit(main())

