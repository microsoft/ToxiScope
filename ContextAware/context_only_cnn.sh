DATAFOLDER=$1
SAVEDFOLDER=$2
GPUID=$3
python main.py --train_file ${DATAFOLDER}/train.json --valid_file ${DATAFOLDER}/valid.json --saved_folder ${SAVEDFOLDER} --config_file ./cnn_context_only_config.json --test_file ${DATAFOLDER}/test.json --gpuid $GPUID --use_context
