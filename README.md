# GreenFoodLens

This repository contains the material to reproduce the dataset of sustainability labels, the analysis of the generated labels, and the configuration files to run the recommender systems through Recbole.

The generated data is available on Zenodo at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14894799.svg)](https://doi.org/10.5281/zenodo.14894798).

[llama_cpp_grammar_ingredient_labeling.py](src/llama_cpp_grammar_ingredient_labeling.py) is the script that runs the LLM to generate the labels by loading the prompts pieces from [prompt_templates_guidance.py](src/prompt_templates_guidance.py) and leveraging constraint generation through Llama-cpp grammars.

[evaluate_llm_labeling.py](src/evaluate_llm_labeling.py) is the script that evaluates the generated labels by computing the accuracy of the labels for exact matches and at different levels of granularity of the taxonomy.

[json_melted.json](json_melted.json) is our revision of the food taxonomy that we used to generate the labels. It is loaded to build the taxonomy for the LLM labeling process to decide which labels are correct candidates at the corresponding level of the hierarchy.

[semantic_matching_eda.py](src/semantic_matching_eda.py) is the script that reproduces the paper proposing HeaSe to find semantic matches between HUMMUS recipe ingredients and the food taxonomy items (tree leaves).

[labeling_analysis.ipynb](src/labeling_analysis.ipynb) is the notebook that reproduces the analysis of the generated labels.

[test_model_sustainability.py](test_model_sustainability.py) is the script that reproduces the analysis of the recommender systems's performance on the test set by computing the CF and WF per serving size, retrieving the top-k recommendations, and explored the models' performance relationship with the CF and WF.

[experiment_config.yaml](experiment_config.yaml) is the configuration file for the recommender systems' training and evaluation (Novelty was added by us by extending the metrics script in the Recbole framework).

[revised_su_eatable_life.pdf](revised_su_eatable_life.pdf) is the revised food taxonomy in PDF for easier reading.
