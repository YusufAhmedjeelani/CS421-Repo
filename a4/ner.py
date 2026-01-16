# CS421: Natural Language Processing
# University of Illinois Chicago
# Fall 2025
# Assignment 4
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

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

#from piazza
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')

# Function: extract_entities(sentence)
# sentence: A string of arbitrary length
# Returns: A list of strings (entities)
#
# This function returns a list of the named entities found in the input sentence.  Only entities with labels
# "PERSON", "GPE", and "ORGANIZATION" are returned, and no duplicate entities are returned.
def extract_entities(sentence):
    seen = set()
    entities = []

    # [WRITE YOUR CODE HERE]
    # Hint: You will likely want to use the functions nltk.word_tokenize, nltk.pos_tag, and nltk.ne_chunk, and then
    # traverse the returned tree to find the allowed entities.
    AllowedLabel = {"PERSON", "GPE", "ORGANIZATION"}

    tokens   = nltk.word_tokenize(sentence)
    POStags = nltk.pos_tag(tokens)
    tree  = nltk.ne_chunk(POStags, binary=False)

    nodes = list(tree)
    i = 0
    
    # Traverse tree
    while i < len(nodes):
        node = nodes[i]
        if isinstance(node, nltk.Tree) and node.label() in AllowedLabel:
            current = " ".join(tok for tok, _ in node.leaves())
            j = i + 1
            while j < len(nodes):
                next = nodes[j]
                # Add all same tags
                if isinstance(next, nltk.Tree) and next.label() == node.label():
                    current += " " + " ".join(tok for tok, _ in next.leaves())
                    j += 1
                else:
                    break
            if current not in seen:
                entities.append(current)
                seen.add(current)
            i = j
        else:
            i += 1

    return entities


# Function: entity_cooccurrences(sentences)
# sentences: A list of strings of arbitrary length
# Returns: A list of tuples of strings
#
# This function forms, for each sentence in the list of sentences, all of the unique unordered pairs of the sentence's
# entities.  It returns the alphabetically sorted pairs (duplicate pairs across different sentences are allowed).
def entity_cooccurrences(sentences):
    pairs = []
    
    # [WRITE YOUR CODE HERE]
    # Hint: You will want to use the extract_entities() function you wrote earlier to extract the entities for a given
    # sentence.
    
    for sentance in sentences:
        entities = extract_entities(sentance)

        seen = set()
        ordered = []
        for ent in entities:
            #Add to list
            if ent not in seen:
                ordered.append(ent)
                seen.add(ent)

        n = len(ordered)
        sentancePairs = []
        # Add pairs in list
        for i in range(n):
            for j in range(i + 1, n):
                a, b = ordered[i], ordered[j]
                pair = tuple(sorted((a, b)))
                sentancePairs.append(pair)

        sentancePairs.sort()
        pairs.extend(sentancePairs)
    return pairs


# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python ner.py
# It should produce the following output (with correct solution):
#
# $ python3 ner.py
# Input: "Taylor Swift met Travis Kelce in Kansas City.",
#         "Selena Gomez visited Chicago with Taylor Swift."
# Output:
# Entities in Taylor Swift met Travis Kelce in Kansas City.:
# ['Taylor Swift', 'Travis Kelce', 'Kansas City']
#
# Entities in "Selena Gomez visited Chicago with Taylor Swift.":
# ['Selena Gomez', 'Chicago', 'Taylor Swift']
#
# Entity Co-Occurrence Network:
# [
#     ('Kansas City', 'Taylor Swift'),
#     ('Kansas City', 'Travis Kelce'),
#     ('Taylor Swift', 'Travis Kelce'),
#     ('Chicago', 'Selena Gomez'),
#     ('Chicago', 'Taylor Swift'),
#     ('Selena Gomez', 'Taylor Swift')
# ]
def main():
    sentences = ["Taylor Swift met Travis Kelce in Kansas City.", "Selena Gomez visited Chicago with Taylor Swift."]
    entities = extract_entities(sentences[0])
    print("Entities in {0}:\n{1}".format(sentences[0], entities))

    entities = extract_entities(sentences[1])
    print("Entities in {0}:\n{1}".format(sentences[1], entities))

    pairs = entity_cooccurrences(sentences)
    print("Entity Co-Occurrence Network:\n{0}".format(pairs))


################ Do not make any changes below this line ################
if __name__ == '__main__':
	exit(main())