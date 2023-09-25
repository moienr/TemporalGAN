import json
import os
import glob
from collections import defaultdict
import pprint

"""Add this file to the same directory as the json files you want to process.
Then run the following code in the python console:
```
python crossval.py
```
This will print the average of the HardEval data in the json files.

"""


def read_json_get_HardEval(filename):
    # Open and load json file
    with open(filename, 'r') as f:
        data = json.load(f)
    # Get HardEval data
    HardEval = data.get('HardEval', None)
    return HardEval

def process_json_files_in_directory(directory):
    # Find json files
    files = glob.glob(os.path.join(directory, '*.json'))

    # Preparing dictionary for cumulative sum of each subkey in HardEval
    cummulative_HardEval = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    total_files = len(files)
    for file in files:
        HardEval = read_json_get_HardEval(file)
        if HardEval:
            for main_key, main_value in HardEval.items():
                for sub_key, sub_value in main_value.items():
                    for inner_key, inner_value in sub_value.items():
                        cummulative_HardEval[main_key][sub_key][inner_key] += inner_value
              
    # Now compute the average for each subkey
    average_HardEval = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for main_key, main_value in cummulative_HardEval.items():
        for sub_key, sub_value in main_value.items():
            for inner_key, inner_value in sub_value.items():
                average_HardEval[main_key][sub_key][inner_key] = inner_value / total_files

    # Conversion to dictionary
    average_HardEval = json.loads(json.dumps(average_HardEval))
    return average_HardEval

# usage
average_HardEval = process_json_files_in_directory('./')
pprint.pprint(average_HardEval)
