train_dist: train.py x_test.pkl x_train.pkl y_test.pkl y_train.pkl dist_config.json
	python train.py \
	  -f x_train.pkl y_train.pkl x_test.pkl y_test.pkl \
	  -d dist_config.json \
	  -l logs/train/ \
	  -o ./saved_model

train: ./saved_model

./saved_model: train.py ./data/data.csv
	python train.py \
	  -f ./data/data.csv \
	  -i 19 31 49 20 56 21 \
	  -l logs/train/ \
	  -o ./saved_model \
	  -t local

retrain:
	python train.py \
	  -f ./data/data.csv \
	  -i 19 31 49 20 56 21 \
	  -l logs/train/ \
	  -o ./saved_model \
	  -t retrain

test_in:
	python metrics.py \
	  -f ./data/data.csv \
	  -i 19 31 49 20 56 21 \
	  -m ./saved_model \
	  -t in

test_out:
	python metrics.py \
	  -f ./data/data.csv \
	  -i 19 31 49 20 56 21 \
	  -m ./saved_model \
	  -t out

infer:
	python inference.py

tensorboard_train:
	tensorboard --logdir ./logs/train --port=8080

tensorboard_train_data:
	python log.py
	tensorboard --logdir ./logs/train_data

tensorboard_train_data_vis:
	python draw_positives.py
	tensorboard --logdir ./logs/train_data/true_positives --port=8080

tensorboard_eval:
	tensorboard --logdir ./logs/test_data --port=8080

generate_dataset:
	python3 generate_dataset.py ./data/data.csv -t 19 31 49 20 56 21

region_proposals:
	python3 region_proposals.py \
	  -m ss \
	  -f ./data/train \
	  -o ./data/train.csv \
	  -t 0.5
