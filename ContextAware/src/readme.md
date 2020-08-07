#Code adopted from https://www.microsoft.com/en-us/research/uploads/prod/2019/05/Wang_SIGIR19.pdf to run WikiComments, GitHub dataset with concatenated context

1. train and evaluate the model 
python train_context_model.py --model_type SentenceAttention --epochs 10 --data_folder ../data/context_matters/ --result_folder ./result --num_classes 2 --sufix _multi_newguideline --glove_file ..\data\embeddings\ --word_vec glove.6B.100d.txt --train_or_test train

2. evaluate the model

python train_context_model.py --model_type SentenceAttention --epochs 10 --data_folder ../data/context_matters/ --result_folder ./result --num_classes 2 --sufix _multi_newguideline --glove_file ..\data\embeddings\ --word_vec glove.6B.100d.txt --train_or_test test