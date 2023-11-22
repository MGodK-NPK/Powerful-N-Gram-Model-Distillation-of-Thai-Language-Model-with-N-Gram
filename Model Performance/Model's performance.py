from perplexity import Perplexity

models = ['gpt2', 'flax-community/gpt2-base-thai']
path_text = 'Powerful-N-Gram-Model-Distillation-of-Thai-Language-Model-with-N-Gram/Datasets/Test set/extracted_str_content.txt'

with open(path_text, 'r') as file:
    # Read the contents of the file into a string variable
    test_data = file.read()


# Perplexity = Perplexity(model= model, data= test_data).cal_perplexity(512)
print(f"Perplexity of {models[2]} model is {Perplexity(model= models[2], data= test_data).cal_perplexity(512)}")

### OUTPUT gpt2 model: 5.07
### OUTPUT pgt2-base-thai model: 26.066