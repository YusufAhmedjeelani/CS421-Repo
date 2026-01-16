# CS421: Natural Language Processing
# University of Illinois Chicago
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


from collections import defaultdict
import nltk


# Function: load_grammar(fname)
# fname: A string indicating a filename
# Returns: A dictionary with two keys:
#          - "binary": list of (lhs, rhs1, rhs2, prob)
#          - "unary": list of (lhs, terminal, prob)
#
# This function loads a grammar from a text file and returns it in a dictionary.
def load_grammar(fname):
    grammar = {"binary": [], "unary": []}
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            lhs = parts[0]
            # binary rule: LHS -> RHS1 RHS2 prob
            if len(parts) == 5:
                _, rhs1, rhs2, prob = parts[1:]
                grammar["binary"].append((lhs, rhs1, rhs2, float(prob)))
            # unary rule: LHS -> terminal prob
            elif len(parts) == 4:
                _, terminal, prob = parts[1:]
                grammar["unary"].append((lhs, terminal, float(prob)))
            else:
                raise ValueError(f"Unrecognized rule format: {line}")
    return grammar


# Function: print_chart(chart)
# chart: A 2D list of dictionaries
# Returns: Nothing
#
# This function prints your completed CKY chart.
def print_chart(chart):
    n = len(chart)
    for i in range(n):
        for j in range(i + 1, n + 1):
            cell = chart[i][j]
            if cell:
                entries = ", ".join(f"{nt}:{p:.6f}" for nt, p in cell.items())
                print(f"[{i},{j}]: {entries}")
            else:
                print(f"[{i},{j}]: empty")

    return


# Function: pcky(sentence, grammar)
# Sentence: A string containing a sentence
# Grammar: A dictionary containing a grammar with keys "binary" and "unary"
# Returns: A 2D list of dictionaries
#          chart[i][j] is a dict mapping nonterminals -> max probability of deriving the span of tokens from i to j
#
# This function loads a grammar from a text file and returns it in a dictionary.
def pcky(sentence, grammar):
    tokens = nltk.word_tokenize(sentence)
    n = len(tokens)

    # Initialize empty chart
    chart = [[defaultdict(float) for i in range(n + 1)] for j in range(n)]

    # [WRITE YOUR CODE HERE]
    if "unary" not in grammar or "binary" not in grammar:
        raise ValueError("unary or binary error")

    #build rule lookup tables
    unaryTerms = defaultdict(list)
    for parentNonT, term, ruleProb in grammar["unary"]:
        unaryTerms[term].append((parentNonT, ruleProb))

    binaryTerms = defaultdict(list)
    for parentNonT, leftNonT, rightNonT, ruleProb in grammar["binary"]:
        binaryTerms[(leftNonT, rightNonT)].append((parentNonT, ruleProb))

    #fill chart
    for idx, word in enumerate(tokens):
        for parentNonT, ruleProb in unaryTerms.get(word, ()):
            if ruleProb > chart[idx][idx + 1][parentNonT]:
                chart[idx][idx + 1][parentNonT] = ruleProb
                
    for spanLen in range(2, n + 1):
        for start in range(0, n - spanLen + 1):
            end = start + spanLen
            destCell = chart[start][end]

            for split in range(start + 1, end):
                leftCell = chart[start][split]
                rightCell = chart[split][end]
                if not leftCell or not rightCell:
                    continue

                for leftNonT, pLeft in leftCell.items():
                    for rightNonT, pRight in rightCell.items():
                        for parentNonT, ruleProb in binaryTerms.get((leftNonT, rightNonT), ()):
                            newProb = pLeft * pRight * ruleProb
                            if newProb > destCell[parentNonT]:
                                destCell[parentNonT] = newProb

    return chart


# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python parse.py
# It should produce the following output (with correct solution) for the example input:
#
# $ python3 parse.py
# [0,1]: NP:0.200000
# [0,2]: empty
# [0,3]: empty
# [0,4]: S:0.004800
# [0,5]: empty
# [0,6]: empty
# [0,7]: S:0.000115
# [1,2]: V:1.000000
# [1,3]: empty
# [1,4]: VP:0.024000
# [1,5]: empty
# [1,6]: empty
# [1,2]: V:1.000000
# [1,3]: empty
# [1,4]: VP:0.024000
# [1,5]: empty
# [1,6]: empty
# [1,4]: VP:0.024000
# [1,5]: empty
# [1,6]: empty
# [1,6]: empty
# [1,7]: VP:0.000576
# [2,3]: Det:0.600000
# [2,4]: NP:0.060000
# [2,5]: empty
# [2,6]: empty
# [2,7]: NP:0.001080
# [2,4]: NP:0.060000
# [2,5]: empty
# [2,6]: empty
# [2,7]: NP:0.001080
# [2,5]: empty
# [2,6]: empty
# [2,7]: NP:0.001080
# [2,6]: empty
# [2,7]: NP:0.001080
# [2,7]: NP:0.001080
# [3,4]: N:0.500000
# [3,4]: N:0.500000
# [3,5]: empty
# [3,5]: empty
# [3,6]: empty
# [3,7]: NP:0.006000
# [4,5]: P:1.000000
# [4,6]: empty
# [4,7]: PP:0.040000
# [4,6]: empty
# [4,7]: PP:0.040000
# [5,6]: Det:0.400000
# [5,7]: NP:0.040000
# [6,7]: N:0.500000
def main():
    print("Loading grammar....")
    grammar = load_grammar("grammar.txt")

    sentence = "I shot an elephant in my pajamas"
    print("Calculating probabilistic CKY for: {0}".format(sentence))
    chart = pcky(sentence, grammar)

    print("Probabilistic CKY Chart:")
    print_chart(chart)


################ Do not make any changes below this line ################
if __name__ == '__main__':
	exit(main())

