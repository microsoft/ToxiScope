## Toxic Language Detection in Workplace Communications

This is the repo for the dataset for detecting toxicity in workplace communications. This work has been published in Findings of EMNLP 2021.
The link to the paper can be found here: [Say ‘YES’ to Positivity: Detecting Toxic Language in Workplace Communications](https://aclanthology.org/2021.findings-emnlp.173.pdf)

### Using the code

#### Installing the dependencies

To run the code, please install the packages from ```requirements.txt``` using the command: ```pip install -r requirements.txt```

#### Dataset

We used Avocado corpus to label toxic instances in this work. Avocado corpus is distributed under [LDC license](https://www.ldc.upenn.edu/data-management/using/licensing). 
We have provided the email id, span of text (start and end index columns) from the email body and the corresponding label.
A ```tsv``` file created after parsing the corpus for provided email id body text in ```data/avacado``` folder can be used for training models. Please note that the text column of email body parsed from the corpus should be named as ```comment```. 

#### Running baselines
To run the baselines, run ```python main.py```

All the hyper-parameters and user options can be modified in ```args.json```.

### Citation
If you find our paper useful, please cite the following:
``` 
@inproceedings{bhat-etal-2021-say-yes,
    title = "Say {`}{YES}{'} to Positivity: Detecting Toxic Language in Workplace Communications",
    author = "Bhat, Meghana Moorthy  and
      Hosseini, Saghar  and
      Awadallah, Ahmed Hassan  and
      Bennett, Paul  and
      Li, Weisheng",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.173",
    pages = "2017--2029",
} 
```
### Contact

For questions, please contact [Meghana Bhat](https://meghu2791.github.io/).


