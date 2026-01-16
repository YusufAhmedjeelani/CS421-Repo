# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2025
# Assignment 1
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


# Sample function which returns a regex to represent a set of strings containing only the capital letters
# Note that it returns a raw python string (preceded by r) which saves us from the backslash plague
# More details here https://docs.python.org/3/howto/regex.html#the-backslash-plague
def capital_letters():
    # ^ signifies the start of string and $ signifies end of string
    # [A-Z] indicates all capital letters
    # [A-Z]+ indicates one or more capital letters
    return r"^[A-Z]+$"


# Q1(a): The set of strings that contain only the letters c or s, or only digits, but not both (no strings
# may contain non-digits or letters other than c or s)
# Returns: regex as a valid python string
def cs_digits():
    # [YOUR CODE HERE]
    return r"^[cs]+$|^\d+$"


# Q1(b): The set of strings containing questions, with the following constraints: the string must contain
# one or more words, the first word must begin with a capital W, and the string must end with a question mark (?)
# Returns: regex as a valid python string
def questions():
    # [YOUR CODE HERE]
    return r"^W[A-Za-z]*\?*(\s+[A-Za-z]+\?*)*\s*\?+$"


# Q1(c): The set of all English future perfect tense phrases, under the simplifying assumption that all future
# perfect tense phrases are constructed using the phrase "will have" followed by a past participle (word) ending
# in "ed", with a single space between each word in the phrase
# Returns: regex as a valid python string
def future_perfect():
    # [YOUR CODE HERE]
    return r"^will have [A-Za-z]+ed$"


# Q1(d): The set of email addresses formatted as "netid@uic.edu", with the constraint that "netid" can be a string
# of at least one but no more than six lowercase letters optionally followed (but not preceded) by at most four digits
# Returns: regex as a valid python string
def email_address():
    # [YOUR CODE HERE]
    return r"^[a-z]{1,6}\d{0,4}@uic\.edu$"


# Q1(e): The set of strings containing mathematical functions, which may include numbers (including decimal values),
# operators (+, -, *, and /), and variables denoted by a single letter, optionally separated by spaces
# Returns: regex as a valid python string
def mathematical_functions():
    # [YOUR CODE HERE]
    return r"^([0-9]+(\.[0-9]+)?|[A-Za-z])( *[+\-*/] *([0-9]+(\.[0-9]+)?|[A-Za-z]))*$"


# Q1(f): The set of all strings matching the valid date "mm/dd/yyyy", where a valid month can be only those with 30
# days (e.g., 09/30/2010 is valid but 02/30/2010 is invalid) and a valid year can be in the range 2000-2025, including
# 2000 and 2025
# Return: regex as a valid python string
def valid_date():
    # [YOUR CODE HERE]
    return r"^(04|06|09|11)/(0[1-9]|[12][0-9]|30)/(200[0-9]|201[0-9]|202[0-5])$"


# Q1(g): The set of all strings that contain only English singulars (e.g., "class" or "lecture hall"), with the
# simplifying assumption that an English singular is a word that does not end with "s" or "es"
# Return: regex as a valid python string
def singulars():
    # [YOUR CODE HERE]
    return r"^[A-Za-z]*([A-RT-Za-rt-z]|[sS]{2,})(\s+[A-Za-z]*([A-RT-Za-rt-z]|[sS]{2,}))*$"


# Q1(h): The set of all strings that have the exact words "llama" and "gpt" in them, in any order, repeated any number
# of times
# Return: regex as a valid python string
def llama_gpt():
    # [YOUR CODE HERE]
    return r"^(llama( (llama|gpt))* gpt( (llama|gpt))*|gpt( (llama|gpt))* llama( (llama|gpt))*)$"


# Q1(i): The set of all websites, under the simplifying assumption that a website URL is formatted as "http" or "https",
# followed by "://www.", followed by a string of alphanumeric characters, followed by ".com", ".org", or ".edu"
# Return: regex as a valid python string
def websites():
    # [YOUR CODE HERE]
    return r"^(http://www\.[A-Za-z0-9]+\.(com|org|edu)|https://www\.[A-Za-z0-9]+\.(com|org|edu))$"


# Q1(j): The set of all strings that have an exclamation mark (!) in them
# Return: regex as a valid python string
def exclamation():
    # [YOUR CODE HERE]
    return r"^.*!.*$"




# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python regex.py
# It should produce the following output:
# 		$ python regex.py 
# 		Match: "HELLOWORLD"
#		No Match: "HELLOWORLD "

def main():
    # Get the regex from function
    regex = capital_letters()

    # Compile the regex
    p = re.compile(regex)

    # Let us test our regex with a valid string
    test = 'HELLOWORLD'
    match = p.match(test)
    if match is None:
        print(f'No Match: "{test}"')
    else:
        print(f'Match: "{test}"')
    
    # Let us test our regex with an invalid string.
    # Why is it invalid?
    test = 'HELLOWORLD '
    match = p.match(test)
    if match is None:
        print(f'No Match: "{test}"')
    else:
        print(f'Match: "{test}"')



################ Do not make any changes below this line ################
if __name__ == '__main__':
    main()
