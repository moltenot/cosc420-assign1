.PHONY: activate test gender race age

task1: race age gender

test:
	. ./env/bin/activate ; python3 run_predictor.py

race:
	. ./env/bin/activate ; python3 race.py

age:
	. ./env/bin/activate ; python3 age.py

gender:
	. ./env/bin/activate ; python3 gender.py

dataset:
	. ./env/bin/activate ; python3 make_numpy_dataset.py

clean_ds:
	. ./env/bin/activate ; python3 clean_dataset.py
