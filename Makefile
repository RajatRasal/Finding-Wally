train_dist: train.py x_test.pkl x_train.pkl y_test.pkl y_train.pkl dist_config.json
	python train.py \
	  -f x_train.pkl y_train.pkl x_test.pkl y_test.pkl \
	  -d dist_config.json \
	  -l logs/train/

tensorboard_single_image:
	python log.py
