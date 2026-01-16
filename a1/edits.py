# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2025
# Assignment 1
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages that were not specified
# in the assignment then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================

import re


# Function to compute the number of insertions, deletions, and substitutions between a given reference phrase
# and hypothesis
# Args:
#   ref: reference text, dtype: string
#   hyp: hypothesis text, dtype: string
# Returns: Number of insertions, deletions, and substitutions, dtype: tuple(integer, integer, integer)
#
def edits(initial, target):
    num_insertions = 0
    num_deletions = 0
    num_substitutions = 0

    # [YOUR CODE HERE]
    source_str = initial.lower()
    distance_str = target.lower()

    n = len(source_str)
    m = len(distance_str)

    # Initalize Matrix
    Dtable = [[0] * (m + 1) for _ in range(n + 1)]
    Ptable = [[None] * (m + 1) for _ in range(n + 1)]

    
    for j in range(1, m + 1):
        Dtable[0][j] = j
        Ptable[0][j] = "INS"  
    for i in range(1, n + 1):
        Dtable[i][0] = i
        Ptable[i][0] = "DEL"  

    # Recurrence Relation
    for i in range(1, n + 1):
        for j in range(1, m + 1):

            if source_str[i - 1] == distance_str[j - 1]:
                Dtable[i][j] = Dtable[i - 1][j - 1]
                Ptable[i][j] = "MATCH"

            else:
                sub = Dtable[i - 1][j - 1] + 1 
                ins = Dtable[i][j - 1] + 1      
                dele = Dtable[i - 1][j] + 1    
                #Backtracking step 
                cost, _, move = min((sub, 1, "SUB"),(ins, 2, "INS"),(dele, 3, "DEL"),)
                Dtable[i][j] = cost
                Ptable[i][j] = move


    i, j = n, m
    while i > 0 or j > 0:
        move = Ptable[i][j]
        if move == "MATCH":
            i -= 1
            j -= 1
        elif move == "SUB":
            num_substitutions += 1
            i -= 1
            j -= 1
        elif move == "INS":
            num_insertions += 1
            j -= 1
        else:  
            num_deletions += 1
            i -= 1

    
    return num_insertions, num_deletions, num_substitutions



# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python wer.py
# It should produce the following output (with correct solution):
#     $ python3 edits.py
#     Insertions:  7
#     Deletions:  0
#     Substitutions:  0
def main():
    # The reference and hypothesis strings
    initial_str = "UIC"
    target_str = "UIP"


    # Call edits function on reference and hypothesis strings to compute the number of
    # insertions, deletions, and substitutions
    # In this example, the correct number of insertions, deletions, and substitutions should
    # be:
    # Insertions:  7
    # Deletions:  0
    # Substitutions:  0
    insertions, deletions, substitutions = edits(initial_str, target_str)
    print("Insertions: ", insertions)
    print("Deletions: ", deletions)
    print("Substitutions: ", substitutions)


if __name__ == '__main__':
    main()