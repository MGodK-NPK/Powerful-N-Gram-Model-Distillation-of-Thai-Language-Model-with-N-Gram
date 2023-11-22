from perplexity import Perplexity

model = 'gpt2'
path_text = 'Powerful-N-Gram-Model-Distillation-of-Thai-Language-Model-with-N-Gram/Datasets/Test set/extracted_str_content.txt'

with open(path_text, 'r') as file:
    # Read the contents of the file into a string variable
    test_data = file.read()

Perplexity = Perplexity(model= model, data= test_data)

calculate = Perplexity.cal_perplexity(512)

print(f"Perplexity of {model} model is {calculate}") #OUTPUT : 5.07