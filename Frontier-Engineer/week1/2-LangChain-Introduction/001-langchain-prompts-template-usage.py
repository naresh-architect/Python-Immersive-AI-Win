"""
LangChain Prompt Templates
Greatly simplifies the task of creating maintainable prompts.
"""

from langchain.prompts import PromptTemplate

examples = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10"
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. #### 5"
    }
]

# define the prompt template without examples
example_template = "question: {question}\nanswer:{answer}"

# template for the example
example_prompt_template = PromptTemplate(
    template=example_template,
    input_variables=["question", "answer"]
)

# test the example prompt (commented out)
# print(example_template.format(**examples[0]))

# --- Prepare a template for the few shot prompt ---

from langchain.prompts import FewShotPromptTemplate

fewshot_template = FewShotPromptTemplate(
    # example prompt template
    example_prompt=example_prompt_template,
    # examples
    examples=examples,
    # A prompt template string to put after the examples
    suffix="question: {input}",
    input_variables=["input"]
)

input_text = "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?"

prompt = fewshot_template.format(input=input_text)
print("=== 3. FewShotPrompt ===")
print(prompt)
print()


# ============================================================
# 4. ExampleSelector
# https://python.langchain.com/docs/modules/model_io/prompts/few_shot_examples#using-an-example-selector
# https://api.python.langchain.com/en/stable/core_api_reference.html#module-langchain_core.example_selectors
# https://python.langchain.com/docs/modules/model_io/prompts/example_selector_types/length_based
# ============================================================

# Change the length to see the difference in the prompt.
# You will always see a full example!!

from langchain.prompts.example_selector import LengthBasedExampleSelector

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt_template,
    max_length=100,
    # you can provide your own function for getting the length of example
    # get_text_length = your func reference
)

fewshot_template = FewShotPromptTemplate(
    # example prompt template
    example_prompt=example_prompt_template,
    # examples - CANNOT provide examples if example_selector is provided
    # examples = examples,
    # examples selector
    example_selector=example_selector,
    # A prompt template string to put after the examples
    suffix="question: {input}",
    input_variables=["input"]
)

prompt = fewshot_template.format(input=input_text)
print("=== 4. ExampleSelector (LengthBased) ===")
print(prompt)