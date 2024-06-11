import os
import shutil
import random

full_data_path = 'data/'
extension_allowed = '.jpg'
split_percentage = 90

files = [file[:-len(extension_allowed)] for file in os.listdir(full_data_path) if file.endswith(extension_allowed)]
random.shuffle(files)

split_index = int(split_percentage * len(files) / 100)
training_files = files[:split_index]
validation_files = files[split_index:]

def copy_files(file_list, img_dest, lbl_dest):
    for file in file_list:
        shutil.copy2(os.path.join(full_data_path, file + extension_allowed), img_dest)
        shutil.copy2(os.path.join(full_data_path, file + '.txt'), lbl_dest)

copy_files(training_files, 'data/images/train/', 'data/labels/train/')
copy_files(validation_files, 'data/images/test/', 'data/labels/test/')

print("Finished")
