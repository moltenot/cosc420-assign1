"""
The dataset has some files that are not in the correct format
Running this file cleans the dataset
"""

import re
import os

DATA_DIR='train'

files = os.listdir(DATA_DIR)
pat = '.*_.*_.*_.*'

for f in files:
    if re.match(pat, f) == None:
        print(f"file: {f} does not match the pattern, deleting")
        os.remove(os.path.join(DATA_DIR, f))