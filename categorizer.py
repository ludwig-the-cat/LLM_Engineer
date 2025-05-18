import ollama
import os

model = 'llama3.2:1b'

input_file = "./data/grocery_list.txt"
output_file = "./data/categorized_grocery_list.txt"

if not os.path.exists(input_file):
    print("input file not found")
    exit(1)

with open(input_file, 'r') as i_f:
    items = i_f.read().strip()

prompt = f"""
    Your are an assistant that categorizes and sort grocery items.
    Here is a list of items:
    {items}
    Please: 
    1. Categorize there items into appropriate categories such as Produce, Dairy, Meat, Bakery, Beverage and etc.
    2. Sort the items alphabetically each category.
    3. Present the categorized list in a clear and organized manner, using bullet points or numbering
"""

try:
    response = ollama.generate(model=model, prompt=prompt)
    generated_text = response.get("response", "")
    print("==== Categorized List: ===== \n")
    print(generated_text)

    # Write the categorized list to the output file
    with open(output_file, "w") as f:
        f.write(generated_text.strip())

    print(f"Categorized grocery list has been saved to '{output_file}'.")
except Exception as e:
    print("An error occurred:", str(e))