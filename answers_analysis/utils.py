import re
from typing import (
    Dict,
    Set,
    Literal,
    Union
)

import polars as pl
import pydantic


class Ingredient(pydantic.BaseModel):
    id: int
    level: int
    label: str
    parent: str


def ingredient_tree_from_json(json_ingredient_data):
    all_labels = [ingr["label"] for ingr in json_ingredient_data]
    assert len(all_labels) == len(set(all_labels)), "Duplicate labels found in json ingredient data"

    ingredient_tree = {'-': {"level": -1, "parent": None, "children": set()}}
    for ingredient_dict in json_ingredient_data:
        ingredient = Ingredient(
            id=ingredient_dict[""],
            level=ingredient_dict["level"],
            label=ingredient_dict["label"].replace('"', ''),
            parent=ingredient_dict["parent"]
        )

        if ingredient.label not in ingredient_tree:
            ingredient_tree[ingredient.label] = {"level": ingredient.level, "parent": ingredient.parent, "children": set()}
        
        if ingredient.parent in ingredient_tree:
            ingredient_tree[ingredient.parent]["children"].add(ingredient.label)
    
    return ingredient_tree


def traverse_ingredients(ingr_parent_child: Dict[str, Union[str, Set[str]]],
                         starting_nodes: Set[str] = None,
                         path_string: str = "") -> str:
    if starting_nodes is None:
        starting_nodes = ingr_parent_child["-"]["children"]

    for i, child in enumerate(starting_nodes):
        ingr_info = ingr_parent_child[child]
        current_path_string = path_string.split('\n')[-1]

        if i == 0:
            path_string += "I DON'T KNOW\n" + current_path_string

        path_string += child
        if not ingr_info["children"]:
            path_string += "\n"
        else:
            path_string += " -> "
            path_string = traverse_ingredients(ingr_parent_child, starting_nodes=ingr_info["children"], path_string=path_string)

        if i != len(starting_nodes) - 1:
            path_string += current_path_string

    return path_string


def clean_and_save_llm_output(llm_output_filepath, output_csv_filepath):
    with open(llm_output_filepath, 'r') as f:
        llm_output_ingredients = f.read().split('\n')

    cleaned_labeled_ingredients = []
    for ingr_label in llm_output_ingredients:
        ingr_label = ingr_label.strip()
        ingr_label = re.sub(r"\d+\. ", '', ingr_label)

        if '<sep>' in ingr_label:
            split_ingr_label = ingr_label.split(' <sep> ')
        else:
            split_ingr_label = ingr_label.split(' -> ', maxsplit=1)
        
        cleaned_labeled_ingredients.append(split_ingr_label)

    cleaned_labeled_ingredients_df = pl.DataFrame(cleaned_labeled_ingredients, schema=['ingredient', 'answer_path'])
    cleaned_labeled_ingredients_df.write_csv(output_csv_filepath, separator="\t")

    print(f"Labeled ingredients after cleaning LLM output saved to {output_csv_filepath}")


def tree_to_pydantic_schema(tree):
    i_dont_know = Literal["I DON'T KNOW"]

    def traverse_tree_to_pydantic(starting_nodes: Set[str]):
        children_schemas = []
        for node_name in starting_nodes:
            node_info = tree[node_name]

            leaf_children = [leaf for leaf in node_info["children"] if not tree[leaf]["children"]]
            non_leaf_children = [leaf for leaf in node_info["children"] if tree[leaf]["children"]]

            # leaf_enum = Enum(f"{node_name}-Items", {leaf: leaf for leaf in leaf_children})
            leaf_enum = (Literal[leaf] for leaf in leaf_children)
            non_leaf_schemas = traverse_tree_to_pydantic(starting_nodes=non_leaf_children)

            node_schema = pydantic.create_model(node_name, next_node=(Union[(*leaf_enum, i_dont_know, *non_leaf_schemas)], ...))
            children_schemas.append(node_schema)

        return children_schemas

    starting_nodes_schemas = traverse_tree_to_pydantic(starting_nodes=tree["-"]["children"])

    full_schema = pydantic.create_model("SU-EATABLE-Path", starting_node=(Union[(i_dont_know, *starting_nodes_schemas)], ...))

    return full_schema


def gbnf_grammar_choice(choices, split_chars=False, as_string=True):
    root = "root"
    assign_op = "::="
    choice_op = "|"

    grammar = [[root, assign_op]]
    for i, ch in enumerate(choices):
        opt = "option_" + str(i)
        grammar[0].append(opt)
        if split_chars:
            choice_chars = list(ch)
            grammar.append([
                opt,
                assign_op,
                *['"' + choice_chars[j] + '"' if (j % 2) == 0 else choice_op for j in range(len(choice_chars) * 2 - 1)]
            ])
        else:
            grammar.append([opt, assign_op, '"' + ch + '"'])

        if i < len(choices) - 1:
            grammar[0].append(choice_op)
    
    if as_string:
        return "\n".join([" ".join(line) for line in grammar])
    else:
        return grammar
