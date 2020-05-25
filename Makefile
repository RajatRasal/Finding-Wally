train_dist: train.py x_test.pkl x_train.pkl y_test.pkl y_train.pkl dist_config.json
	python train.py \
	  -f x_train.pkl y_train.pkl x_test.pkl y_test.pkl \
	  -d dist_config.json \
	  -l logs/train/ \
	  -o ./saved_model

train: train.py x_test.pkl x_train.pkl y_test.pkl y_train.pkl dist_config.json
	python train.py \
	  -f x_train.pkl y_train.pkl x_test.pkl y_test.pkl \
	  -l logs/train/ \
	  -o ./saved_model

test:
	python metrics.py \
	  -f x_train.npy y_train.pkl x_test.npy y_test.pkl \
	  -m ./saved_model

infer:
	python inference.py

tensorboard_train:
	tensorboard --logdir ./logs/train --port=8080

tensorboard_train_data:
	python log.py
	tensorboard --logdir ./logs/train_data

tensorboard_eval:
	tensorboard --logdir ./logs/test_data --port=8080

generate_dataset:
	python3 generate_dataset.py ./data/data.csv -t 19 31 49 20 56 21
