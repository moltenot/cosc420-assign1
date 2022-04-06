.PHONY: activate

age:
	. ./env/bin/activate ; python3 age.py

age-tb:
	. ./env/bin/activate ; tensorboard --logdir=age-logs

dataset:
	. ./env/bin/activate ; python3 make_numpy_dataset.py

clean_ds:
	. ./env/bin/activate ; python3 clean_dataset.py
