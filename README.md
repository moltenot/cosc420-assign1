# COSC420 Assignment 1 - UTKFace models

## Setup
you will need to setup the python virtual environment. I am using one called 
"env" in this directory. Use the command
```
python3 -m venv env
```
Then you will need to activate it and install the packages from the
`requirements.txt`. You can use the commands below to do this
```
source env/bin/activate
pip install -r requirements.txt
```
Then once those have installed you simply need to unzip the UTKFace dataset
into a subfolder of this repository called "train"

To clean these files run the python file `clean_dataset.py` using the command
```
make clean
```

Now that every file in the dataset fits the specification of having a valid
label you can create `.npy` files that will be used as the datasets for training.
The code for this is in `make_numpy_dataset.py` and can be run using
```
make dataset
```

## Training:

### Task 1
The code for each of the 3 models is in `age.py`, `gender.py`, and `race.py`.
To run each of these you can use the commands `make age`, `make gender`, or
`make race`. If you want to train all 3 in sequence you can use the commnand
```
make task1
```


### Task 2