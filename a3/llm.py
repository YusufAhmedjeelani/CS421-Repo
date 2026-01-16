# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2025
# Assignment 3
#
# Unlike other skeleton code in CS 421, this code will *not* be autograded!  It is for demonstration
# purposes, to assist with evaluating LLM output.
#
# Before running this code, make sure to install the HuggingFace transformers package and other associated
# libraries:
# pip install transformers bitsandbytes accelerate
#
# This demonstration code was adapted from the HuggingFace tutorial here:
# https://huggingface.co/docs/transformers/en/llm_tutorial.  You're encouraged to review the page for
# additional tips regarding LLM prompting.
# =========================================================================================================


from transformers import AutoTokenizer, AutoModelForCausalLM


# Function: prompt_llm(prompt)
# prompt: A text string used to extract output from the language model
# Returns: An output text string
#
# This function loads the specified large language model (LLM) and prompts it to produce output.
def prompt_llm(prompt):

    # The code below should not require a GPU, but it will install a very large file (2+ GB) and depending on
    # your computer's hardware it may take a very long time to load.  If you find that running it is not feasible
    # with your computer's hardware, you can also prompt numerous models using a GUI environment here if you create
    # an account: https://huggingface.co/models?pipeline_tag=text-generation&sort=trending (look for models that
    # have an available input box in the "Inference API" area on the righthand side of their landing page).
    #tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
   # model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Uncomment the lines below if you'd like to try prompting a smaller LLM (~0.5 GB download) locally:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # return_tensors can return PyTorch ("pt"), TensorFlow ("tf"), or Numpy ("np") objects
    model_inputs = tokenizer([prompt], return_tensors="pt")

    # max_length constrains the output to be no longer than the desired value; you may want to increase this or
    # remove that parameter (it's optional) entirely for some tasks
    generated_ids = model.generate(**model_inputs, max_length=25)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return output


# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python llm.py
# It should produce the following output (with correct solution):
#
# $ python3 llm.py
# Input: The first letter of UIC is
# Output: The first letter of UIC is U, which is the first letter of the word "university." The second letter
def main():
    prompt = "Generate Penn Treebank POS tags for 'Time flies like an arrow; fruit flies like a banana.'"
    output = prompt_llm(prompt)
    print("Input: {0}\nOutput: {1}".format(prompt, output))


################ Do not make any changes below this line ################
if __name__ == '__main__':
	exit(main())

