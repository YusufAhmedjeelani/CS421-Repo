# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2025
# Assignment 3
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in
# the assignment then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================

import numpy as np



# Function to get cosine similarity see Equation 6.10 (Jurafsky & Martin v3) for reference
# Arguments:
# a: A numpy vector of size (x, )
# b: A numpy vector of size (x, )
# Returns: sim (float)
# Where, sim (float) is the cosine similarity between vectors a and b. x is the size of the numpy vector. Assume that both vectors are of the same size.
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    csim = 0.0
    # [WRITE YOUR CODE HERE]
    numVector = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        csim = 0.0
    else:
        csim = numVector / denom
    return csim


# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python csim.py
# It should produce the following output (from the template):
#
# Vector a: [1 2 3]
# Vector b: [7 8 9]
# Vector a shape: (3,)
# Vector b shape: (3,)
# Cosine similarity: 0.96
def main():
    # Initialize the vectors
    a = np.array([1, 2, 3])
    b = np.array([7, 8, 9])

    # We can print the vectors as 
    print("Vector a: {0}".format(a))
    print("Vector b: {0}".format(b))

    # We can see shape (and dimensions) of vectors as
    print("Vector a shape: {0}".format(a.shape))
    print("Vector b shape: {0}".format(b.shape))

    # Compute cosine similarity
    sim = cosine_similarity(a, b)
    print("Cosine similarity: {0:.2}".format(sim))




################ Do not make any changes below this line ################
if __name__ == '__main__':
    main()
