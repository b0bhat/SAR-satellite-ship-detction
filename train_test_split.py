import json
import numpy as np
import pandas as pd
import os
import sys
import shutil
from sklearn.model_selection import train_test_split

type = sys.argv[1]

with open(type + '.json') as f:
    data = json.load(f)

images_df = pd.DataFrame(data['images'])
annotations_df = pd.DataFrame(data['annotations'])

image_ids = images_df['id'].tolist()

image_dir = 'HRSID/'
image_output = 'data/images/'
text_output = 'data/labels/'

spec_dir = type + '/'

for image in data['images']:
    img_id = image['id']
    img_file_name = image['file_name']
    img_width = image['width']
    img_height = image['height']
    
    annotations = annotations_df[(annotations_df['image_id'] == img_id)]
    if annotations.empty:
        continue
    
    bboxes = np.array(annotations['bbox'].tolist())
    category_ids = annotations['category_id'].values

    x_center = (bboxes[:, 0] + bboxes[:, 2] / 2) / img_width
    y_center = (bboxes[:, 1] + bboxes[:, 3] / 2) / img_height
    width = bboxes[:, 2] / img_width
    height = bboxes[:, 3] / img_height

    yolo_format = np.column_stack((category_ids-1, x_center, y_center, width, height))

    image_output_dir = image_output + spec_dir
    label_output_dir = text_output + spec_dir

    txt_file_path = os.path.join(label_output_dir, f'{img_id}.txt')
    np.savetxt(txt_file_path, yolo_format, fmt='%d %.6f %.6f %.6f %.6f')

    src_image_path = os.path.join(image_dir, img_file_name)
    dst_image_path = os.path.join(image_output_dir, f'{img_id}.jpg')
    shutil.copyfile(src_image_path, dst_image_path)