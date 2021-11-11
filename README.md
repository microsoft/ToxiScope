# Toxic Language Detection in Workplace Communications

This is the repo for the dataset for detecting toxicity in workplace communications. This work has been published in EMNLP 2021.
The link to the paper can be found here: [Say ‘YES’ to Positivity: Detecting Toxic Language in Workplace Communications](https://aclanthology.org/2021.findings-emnlp.173.pdf)

## Using the code

### Installing the dependencies

To run the code, please install the packages from ```requirements.txt``` using the command: ```pip install -r requirements.txt```


To run linear models from related papers:
python main.py --model <name> --train_set <train_data> --test_set <test_data> --train_or_test <train/test> 

Currently supported model name arguments:
avocado, wiki, github, context_matters

data is present in data/<model_name> folder

