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


import numpy as np
from sentence_transformers import SentenceTransformer


# Function to get sentence representations for a list of input documents
# Arguments:
# docs: A list of strings, with each string representing a document
# Returns: mat (numpy.ndarray) of size (len(docs), dim)
# mat is a two-dimensional numpy array containing a vector representation for
# the ith document (in input list docs) in the ith row, and dim represents the
# dimensions of the output vectors (here, dim = 384 for the
# paraphrase-MiniLM-L3-v2 model)
def sentence_rep(docs: list[str]) -> np.ndarray:
    dim = 384
    mat = np.zeros((len(docs), dim))

    # [Your code here!]
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

    embeddings = model.encode(docs, convert_to_numpy=True)
    mat = np.array(embeddings)

    return mat


# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python sentence_rep.py
#
# Running the sample code with a correctly-implemented sentence_rep function should
# produce the following output:
# [[ 0.04179132 -0.01482642  0.02694485 ... -0.2956554  -0.27379724
#    0.54425013]
#  [ 0.18813246  0.01910055 -0.29848018 ... -0.02525909 -0.0477315
#    0.10944465]
#  [-0.02668928  0.07221615  0.0118683  ... -0.38545278  0.2251407
#    0.11856089]
#  [-0.08885137 -0.00196246  0.17794801 ... -0.10296345 -0.04274498
#    0.09362762]
#  [ 0.27780864  0.14017436  0.14890964 ... -0.00681332 -0.1736005
#    0.09357634]]
def main():
    # Initialize the corpus
    sample_corpus = ['Many buildings at UIC are designed in the brutalist style.',
                     'Brutalist buildings are generally characterized by stark geometric lines and exposed concrete.',
                     'One famous proponent of brutalism was a Chicago architect named Walter Netsch.',
                     'Walter Netsch designed the entire UIC campus in the early 1960s.',
                     'When strolling the campus and admiring the brutalism, remember to think of Walter Netsch!']

    sample_reps = sentence_rep(sample_corpus)
    print(sample_reps)

    sentences = [("She bought a cup of coffee on her way to work.", "On her commute, she picked up a coffee to go."),
            ("Budapest has so much to do, even when it’s chilly!",
            "Bangkok is a great escape during the colder months!"),
            ("I love going to CS 421 on Tuesday mornings.",
            "Tuesdays are my favorite day of the week."),
            ("The banks in Michigan close at 5 p.m.",
            "She sat on the banks of Lake Michigan."),
            ("The end of the semester is approaching too quickly.",
            "They dressed up as a language model for Halloween."),
            ("The new album was fantastic and I loved it.",
            "The new album was terrible and I hated it."),
            ("The students wrote messages on Piazza to the professor.",
            "The professor wrote messages on Piazza to the students.")
        ]
    output = [s for pair in sentences for s in pair]

    # compute 14x384 embedding matrix
    from csim import cosine_similarity

    output = sentence_rep(output)

    print("Q1C. outputs (embeddings shape):", output.shape)
    print()

    # unpack embeddings into labeled sentence vectors
    (sentences1A, sentences1B,
    sentences2A, sentences2B,
    sentences3A, sentences3B,
    sentences4A, sentences4B,
    sentences5A, sentences5B,
    sentences6A, sentences6B,
    sentences7A, sentences7B) = output

    # compute cosine similarity for each pair
    scores = [
        cosine_similarity(sentences1A, sentences1B),
        cosine_similarity(sentences2A, sentences2B),
        cosine_similarity(sentences3A, sentences3B),
        cosine_similarity(sentences4A, sentences4B),
        cosine_similarity(sentences5A, sentences5B),
        cosine_similarity(sentences6A, sentences6B),
        cosine_similarity(sentences7A, sentences7B)
    ]

    # print results nicely
    print("Q1C. cosine similarity results:\n")
    for i, s in enumerate(scores, 1):
        print(f"Sentence Pair {i}: {s:.4f}")

    new_sents = [
    "I love Chicago weather in the fall.",
    "The weather in Chicago makes me fall sick."
    ]

    embs = sentence_rep(new_sents)
    score = cosine_similarity(embs[0], embs[1])
    print(f"New Pair Similarity: {score:.4f}")
################ Do not make any changes below this line ################
if __name__ == '__main__':
    main()
