import os

def remove_files_by_name(names, directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_name = os.path.basename(file_path)
            if not any(name in file_name for name in names):
                os.remove(file_path)
                
import pandas as pd

def get_column_values(csv_file, column_name):
    df = pd.read_csv(csv_file)
    column_values = df[column_name].tolist()
    return column_values

# if __name__ == "__main__":
#     csv_file = "D:\\python\\TemporalGAN\\changedetection\\chaned_pairs.csv"
#     names_list = get_column_values(csv_file, "name")
#     print(names_list[0], len(names_list))
    