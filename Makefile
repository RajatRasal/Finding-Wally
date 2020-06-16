test_images = 31 49 20 56 21
gpu_memory  = 7000

train_dist: ./saved_model_dist

./saved_model_dist: train.py model.py ./data/data.csv dist_config.json
	python train.py \
	  -f ./data/data.csv \
	  -i ${test_images} \
	  -g ${gpu_memory} \
	  -l logs/train/ \
	  -o ./saved_model \
	  -t dist \
	  -d dist_config.json

train_cycle: train retrain retrain retrain retrain retrain

train: ./saved_model

./saved_model: train.py model.py ./data/data.csv
	python train.py \
	  -f ./data/data.csv \
	  -i ${test_images} \
	  -g ${gpu_memory} \
	  -l logs/train/ \
	  -e 500 \
	  -o ./saved_model \
	  -t local

retrain:
	python train.py \
	  -f ./data/data.csv \
	  -i ${test_images} \
	  -g ${gpu_memory} \
	  -l logs/train/ \
	  -o ./saved_model \
	  -t retrain

test_in:
	python metrics.py \
	  -f ./data/data.csv \
	  -i ${test_images} \
	  -m ./saved_model \
	  -t in

test_out:
	python metrics.py \
	  -f ./data/data.csv \
	  -i ${test_images} \
	  -m ./saved_model \
	  -t out

test_with_datasets:
	./inference_test.sh "${test_images}"

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
	python3 generate_dataset.py ./data/data.csv -t ${test_images}

region_proposals:
	python3 region_proposals.py \
	  -m ss \
	  -f ./data/train \
	  -o ./data/train.csv \
	  -t 0.5
