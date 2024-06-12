import json
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split

with open('val.json') as f:
    data = json.load(f)

image_dir = 'val/'
image_output = 'data/images/'
text_output = 'data/labels/'

train_dir = 'train/'
test_dir = 'test/'

image_ids = [image['id'] for image in data['images']]
train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

for image in data['images']:
    img_id = image['id']
    img_file_name = image['file_name']
    img_width = image['width']
    img_height = image['height']
    
    annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
    if not annotations:
        continue
    
    bboxes = np.array([ann['bbox'] for ann in annotations])
    category_ids = np.array([ann['category_id'] for ann in annotations])

    x_center = (bboxes[:, 0] + bboxes[:, 2] / 2) / img_width
    y_center = (bboxes[:, 1] + bboxes[:, 3] / 2) / img_height
    width = bboxes[:, 2] / img_width
    height = bboxes[:, 3] / img_height

    yolo_format = np.column_stack((category_ids, x_center, y_center, width, height))

    if img_id in train_ids:
        image_output_dir = image_output + train_dir
        label_output_dir = text_output + train_dir
    else:
        image_output_dir = image_output + test_dir
        label_output_dir = text_output + test_dir

    txt_file_path = os.path.join(label_output_dir, f'{img_id}.txt')
    np.savetxt(txt_file_path, yolo_format, fmt='%d %.6f %.6f %.6f %.6f')

    src_image_path = os.path.join(image_dir, img_file_name)
    dst_image_path = os.path.join(image_output_dir, f'{img_id}.jpg')
    shutil.copyfile(src_image_path, dst_image_path)