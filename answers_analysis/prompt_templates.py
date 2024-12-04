prompt_templates = {
    "system": """
        You are an expert in categorizing ingredients based on a structured list of predefined paths. Your task is to assign the correct category path to each ingredient by analyzing each option carefully.

            Important Instructions:
            1. Only use the category paths listed in the provided context. Do not create or infer new categories; each path you assign must exactly match one from the context.
            2. When categorizing each ingredient, follow a step-by-step reasoning process:
            - First, check if the ingredient exactly matches a category path in the context. If it does, use that path.
            - If there is no exact match, look for the closest match by considering related words or phrases in the available paths.
            - If no path closely matches the ingredient, or if you are uncertain about the match, select a path that ends with "I DON'T KNOW".
    """,
    "user": {
        "base_prompt": """
            Given the following categorization context:
            {context}

            Please categorize the following ingredient accordingly. If the ingredient doesn't match any category exactly, categorize it based on the closest fit.

            Ingredient:
            {ingredient}
        """,
        "chatgpt_prompt_engineering_prompt_v1": """
            Given the following categorization context:
            {context}

            Ingredient to Categorize:
            {ingredient}

            Example Process
            Use the following reasoning process and format:
            1. Ingredient: CHICKEN BREAST
            - Step 1: Check if "CHICKEN BREAST" exactly matches a path in the context.
            - Step 2: If not, search for any path in the context related to "CHICKEN" or "POULTRY."
            - Step 3: If no appropriate path exists, select "I DON'T KNOW" from the context if available.
            - Response: MEAT PRODUCTS -> POULTRY MEAT -> POULTRY BONE FREE MEAT -> CHICKEN BONE FREE MEAT
        """,
        "chatgpt_prompt_engineering_prompt_v2": """
            Few Examples to Guide You
            1. Ingredient: CHICKEN BREAST
            - Step-by-Step:
                - Check if "CHICKEN BREAST" has an exact match in the context. (None found)
                - Look for related terms like "CHICKEN" or "POULTRY." (If no path fits, choose "I DON'T KNOW.")
            - Output: `MEAT PRODUCTS -> POULTRY MEAT -> POULTRY BONE FREE MEAT -> CHICKEN BONE FREE MEAT`

            2. Ingredient: COD
            - Step-by-Step:
                - Check if "COD" has an exact match in the context. (Match found)
            - Output: `FISHING -> FISH -> COD`

            3. Ingredient: BEEF
            - Step-by-Step:
                - Check if "BEEF" has an exact match in the context. (If none exists, select "I DON'T KNOW.")
            - Output: `MEAT PRODUCTS -> BEEF MEAT`

            Categorization Context:
            {context}

            Ingredient to Categorize:
            {ingredient}
        """
    }
}