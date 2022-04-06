COSC420 Assignment 1 - UTKFace models

to set this up, setup a python venv with the name env in this directory.
Then you will need to put the UTKFace dataset in a subdir called "train"
since there are some files that don't use valid labelling criteria 
you have to run clean_dataset.py to remove these samples.

Then it is smart to create .npy files of the data to be loaded. The file
make_numpy_dataset.py will produce 4 files.
 - images.npy
 - ages.npy
 - genders.npy
 - races.npy
where each one contains the processed data from the train subdir


Training:

Age predictor: age.py
to make the age model you can use the command
	make age
which will construct a model to predict ages, putting the tensorboard 
details in 'age-logs/*'. The tensorboard details can be viewed using 
the command
	make age-tb
