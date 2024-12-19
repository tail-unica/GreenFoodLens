prompt_templates = {
    "system": """
You are an expert in categorizing ingredients based on a structured list of predefined paths. Your task is to assign the correct category path to each ingredient by analyzing each option carefully
and adding just one category layer at a time, not the full path at once.
If the ingredient doesn't match any category from the next categorical candiates, categorize it based on the closest fit or answer I DON'T KNOW if you are not confident in your answer.
Do not generate any other text except for the category name or 'I DON'T KNOW'.

Consider that the upper level of the hierarchy is the most general category and the lower levels are more specific categories.
The upper level categories are:
1) AGRICULTURAL PROCESSED: it covers any kind of plant based processed food (includes ice-cream).
2) MEAT PRODUCTS: it covers products of terrestrial animal origin.
3) ANIMAL DERIVED: it covers products of animal origin. This group includes milk, eggs, honey, etc.
4) CROPS: it covers plant based product, not processed. This group includes fresh plant products, seeds, dry fruit.
5) FISH: it covers all the animals and weeds from fresh and salted waters.
""",
    "few_shot_examples": """
Few examples of the final paths that you should generate, but you should not generate the full path at once, just one category layer at a time.
Note: liquors that are not beers or wines must be categorized as 'AGRICULTURAL PROCESSED -> I DON'T KNOW'
Ingredient: CHICKEN BREAST
Output: `MEAT PRODUCTS -> POULTRY MEAT -> POULTRY BONE FREE MEAT -> CHICKEN BONE FREE MEAT`

Ingredient: COD
Output: `FISHING -> FISH -> COD`

Ingredient: MOOSE
Output: `MEAT PRODUCTS -> I DON'T KNOW

Ingredient: POMEGRANTE SCHNAPPS
Output: `AGRICULTURAL PROCESSED -> I DON'T KNOW`
""",
"bootstrap_description": """
Ingredient: {ingredient}

Write a brief description of the ingredient and its properties. This will help you to categorize the ingredient more accurately.
""",
    "partial_path_instruction": """
Select the appropriate category to append to the partial path below.
{path}
""",
    "instruction": """
Remember the ingredient is: {ingredient}
{partial_path_instruction}

Reply with just the category name or 'I DON'T KNOW' if you are not sure, do not generate any other text.

""",
    "instruction_with_candidates": """
Remember the ingredient is: {ingredient}
{partial_path_instruction}

Select the appropriate category at the current hierarchical level as only one of the candidates categories below separated by '|':
{candidates}

Reply with just the category name or 'I DON'T KNOW' if you are not sure, do not generate any other text.

"""
}