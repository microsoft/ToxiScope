DATAFOLDER=$1
SAVEDFOLDER=$2
GPUID=$3
python main.py --train_file ${DATAFOLDER}/train.json --valid_file ${DATAFOLDER}/valid.json --saved_folder /data/wei/WSDM/${SAVEDFOLDER} --config_file ./cnn_context_config.json --test_file ${DATAFOLDER}/test.json --gpuid $GPUID --use_context
