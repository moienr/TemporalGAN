import os
import sys

script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
script_dir = os.path.dirname(script_path) #i.e. /path/to/dir/
sys.path.append(script_dir) # adding the module folder to the path so, that, when calling the modules outside of this foler
# the modules that import each other, won't get error.