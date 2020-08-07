DATAFOLDER=$1
SAVEDFOLDER=$2
python main.py --train_file ${DATAFOLDER}/train.json --valid_file ${DATAFOLDER}/valid.json --saved_folder /data/wei/WSDM/${SAVEDFOLDER} --config_file ./cnn_config.json --test_file ${DATAFOLDER}/test.json
