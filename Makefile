.PHONY: activate test

task1: race age gender

test:
	. ./env/bin/activate ; python3 run_predictor.py

race:
	. ./env/bin/activate ; python3 race.py

race-tb:
	. ./env/bin/activate ; tensorboard --logdir=race-logs


age:
	. ./env/bin/activate ; python3 age.py

age-tb:
	. ./env/bin/activate ; tensorboard --logdir=age-logs


gender:
	. ./env/bin/activate ; python3 gender.py

gender-tb:
	. ./env/bin/activate ; tensorboard --logdir=gender-logs

dataset:
	. ./env/bin/activate ; python3 make_numpy_dataset.py

clean_ds:
	. ./env/bin/activate ; python3 clean_dataset.py
