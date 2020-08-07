DATAFOLDER=$1
SAVEDFOLDER=$2
python main.py --train_file ${DATAFOLDER}/train.json --valid_file ${DATAFOLDER}/valid.json --saved_folder ${SAVEDFOLDER} --config_file ./rnn_config.json --test_file ${DATAFOLDER}/test.json
