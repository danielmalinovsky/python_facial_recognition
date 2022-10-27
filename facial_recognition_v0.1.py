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
#Resizing cropped image
resize_crop_img = cv2.resize(crop_img, (300, 300))
plt.imshow(cv2.cvtColor(resize_crop_img, cv2.COLOR_BGR2RGB))

# %%
#reading the annotations
from sklearn.model_selection import train_test_split
annot_path = str(input()) #enter your path. C:/Users/ngnpe/OneDrive/Desktop/Agile_ML_zip/Anno/
#%%
annots = pd.read_csv(annot_path +'identity_CelebA.txt',
                        delim_whitespace = True,
                        header = None,
                        names = ['jpg', 'label'])
#filtering lables having at least 3 observations.
labels_annot = pd.DataFrame(annots.label.value_counts(ascending=True)).query('label > 20').index.tolist()
annots_filtered = annots[annots['label'].isin(labels_annot)]
#Splitting the annotations
imgs = annots_filtered['jpg']
labels = annots_filtered['label']
temp_imgs, test_imgs, _, __ = train_test_split(imgs, labels,
                                               test_size = 0.2,
                                               random_state = 123,        
                                               stratify = labels)
train_imgs, valid_imgs, _, __ = train_test_split(temp_imgs, _,
                                               test_size = 0.25,
                                               random_state = 123,        
                                               stratify = _)

# %%
print('Export the files?')
if input() == 'Yes':
    train_imgs.to_csv('train_imgs.csv', index = False)
    valid_imgs.to_csv('valid_imgs.csv', index = False)
    test_imgs.to_csv('test_imgs.csv', index = False)
    
# %%
#10 random pictures
import random
random.seed(123)
random_pics = random.choices(annots_filtered['jpg'].values, k=10)
#bounding boxes and the path
bbox_filtered = bbox[bbox['image_id'].isin(random_pics)]
#%%
#path of images
imgs_path = str(input()) #enter your path - C:/Users/ngnpe/OneDrive/Desktop/Agile_ML_zip/img_celeba.7z/img_celeba_001/
# %%
#Cropping 10 random pictures
for pic in random_pics:
    crop_img = src.face_crop(pic, imgs_path, 'image_id',
                                bbox_col_names, bbox_filtered,
                                show_bbox = False )
    plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    plt.show()
#%%
#Resizing of 10 random cropped pictures.
for pic in random_pics:
    crop_img = src.face_crop(pic, imgs_path, 'image_id', bbox_col_names, bbox_filtered, show_bbox = False )
    resize_crop_img = cv2.resize(crop_img, (300, 300))
    plt.imshow(cv2.cvtColor(resize_crop_img, cv2.COLOR_BGR2RGB))
    plt.show()