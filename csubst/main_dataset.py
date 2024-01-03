import os
import shutil

def main_dataset(g):
    dir_csubst = os.path.dirname(os.path.abspath(__file__))
    dir_dataset = os.path.join(dir_csubst, 'dataset')
    files = [ f for f in os.listdir(dir_dataset) if f.startswith(g['name']) ]
    for file in files:
        new_file_name = file.replace(g['name']+'.', '')
        path_from = os.path.join(dir_dataset, file)
        path_to = os.path.join('.', new_file_name)
        print(f"Copying {g['name']} file: {new_file_name}")
        shutil.copy(path_from, path_to)