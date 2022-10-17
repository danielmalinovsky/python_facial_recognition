# %%
import cv2
from matplotlib import pyplot as plt
import pandas as pd

# User defined functions
import src.proprietary_functions as src


"""Image dataset source: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"""


# %%
# Setting filepaths

image_file = './sample_data/Img/'
annotation_file = './sample_data/Anno/'

identity_file = annotation_file + 'identity_CelebA.txt'
bbox_file = annotation_file + 'list_bbox_celeba.txt'
image_list = ['000001.jpg', '000002.jpg']

image_id_col = 'image_id'
bbox_col_names = {
    'x_start' : 'x_1',
    'y_start' : 'y_1',
    'width' : 'width',
    'height' : 'height',
    'x_end' : '',
    'y_end' : ''}

# Loading dataset metadata
identity = pd.read_csv(identity_file, sep=" ", header = None,names=['image', 'image_id'])
bbox = pd.read_csv(bbox_file, delim_whitespace=True)

# %%
# Cropping images

crop_img = src.face_crop(image_list[0], image_file, image_id_col, bbox_col_names, bbox, False)
plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

# %%
