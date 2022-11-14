# %%
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import os
import tensorflow as tf
import keras as keras
from PIL import Image
#import cvlib as cv


#resnet 50
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


# User defined functions
import src.proprietary_functions as src

"""Image dataset source: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"""


# %%
# Setting filepaths
image_file = './sample_data/Img/'
annotation_file = './sample_data/Anno/'
export_file = './export'

identity_file = annotation_file + 'identity_CelebA.txt'
bbox_file = annotation_file + 'list_bbox_celeba.txt'

# Train/test split variables
random_seed = 123
test_size = 0.2
validation_size = 0.2

# Bbox parameters
method1_min_neighbours = 13

# Setting column names
image_id_col = 'image_id'
bbox_col_names = {
    'x_start' : 'x_1',
    'y_start' : 'y_1',
    'width' : 'width',
    'height' : 'height',
    'x_end' : '',
    'y_end' : ''}

identity_file = annotation_file + 'identity_CelebA.txt'
bbox_file = annotation_file + 'list_bbox_celeba.txt'



# Loading dataset metadata
identity = pd.read_csv(identity_file, sep=" ", header = None,names=['image', 'image_id'])
bbox = pd.read_csv(bbox_file, delim_whitespace=True)


#%% Filtering faces that appear at least 20 times
labels_annot = pd.DataFrame(identity.image_id.value_counts(ascending=True)).query('image_id > 20').index.tolist()
identity_filtered = identity[identity['image_id'].isin(labels_annot)]

#%% [SPRINT 2] Train/test split of the annotations
imgs = identity_filtered['image']
labels = identity_filtered['image_id']

temp_imgs, test_imgs, _, __ = train_test_split(imgs, labels,
                                               test_size = test_size,
                                               random_state = random_seed,        
                                               stratify = labels)
train_imgs, valid_imgs, _, __ = train_test_split(temp_imgs, _,
                                               test_size = validation_size/(1-test_size),
                                               random_state = random_seed,        
                                               stratify = _)

#%% 
# Safe train/test split

if not os.path.exists(export_file):
    os.makedirs(export_file)

if not os.path.exists(export_file + '/setting'):
    os.makedirs(export_file + '/setting')

if export_file != '':
    train_imgs.to_csv(export_file + '/setting/train_imgs.csv', index = False)
    valid_imgs.to_csv(export_file + '/setting/valid_imgs.csv', index = False)
    test_imgs.to_csv(export_file + '/setting/test_imgs.csv', index = False)

#%%
# [SPRINT 2] Random selection
random.seed(random_seed)
random_pics = random.choices(identity_filtered['image'].values, k=10)
bbox_filtered = bbox[bbox['image_id'].isin(random_pics)] #generate bboxes

# %% [SPRINT 2] BBox engine demonstration

img = cv2.imread(image_file+random_pics[0])
bbox_generated = src.bbox_engine(img, method = 0)

cv2.rectangle(img, (bbox_generated['x_1'],bbox_generated['y_1']), (bbox_generated['x_end'],bbox_generated['y_end']), (0,255,0), 2)       
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# %%
# [SPRINT 2 demonstration]
# %% Generating Bboxes

bbox_generated = pd.DataFrame(columns= ['image_id', 'x_1', 'y_1', 'width', 'height', 'x_end', 'y_end'])

for pic in random_pics:

    img = cv2.imread(image_file+pic)
    bbox_coordinates = src.bbox_engine(img, method = 1, m1_min_neighbors= method1_min_neighbours)
    bbox_coordinates['image_id'] = pic
    bbox_generated = bbox_generated.append(bbox_coordinates, ignore_index = True)


# %% Bbox verification

image_lens = pd.DataFrame(columns= ['image_id', 'len'])

for pic, in zip(random_pics):

    image_path = image_file + pic
    img = cv2.imread(image_path)
    len_dict = {'image_id' : pic, 'len' : len(img)}
    image_lens = image_lens.append(len_dict, ignore_index = True)

bbox_verification_df = src.bbox_verification(bbox_filtered, bbox_generated, image_lens)

# %% Large bbox mismatch plot
large_diff_images = bbox_verification_df[bbox_verification_df['relative_diff'] > 0.5]['image_id']

for i in range(len(large_diff_images)):
    src.bbox_show_compare(large_diff_images.iloc[i], image_file, 'image_id', bbox_col_names, bbox_filtered, bbox_generated)

#%%
# [SPRINT 2] Plotting 10 random pictures
fig_1, axs_1 = plt.subplots(nrows = 5, ncols = 2, figsize = (15, 35))

for pic, ax_1 in zip(random_pics, axs_1.ravel()):

    image_path = image_file + pic
    img = cv2.imread(image_path)

    ax_1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_1.title.set_text(pic)

fig_1.tight_layout()
plt.show()

#%%
# [SPRINT 2] Generating bboxes for 10 random pictures
fig_2, axs_2 = plt.subplots(nrows = 5, ncols = 2, figsize = (15, 35))

for pic, ax_2 in zip(random_pics, axs_2.ravel()):

    crop_img = src.face_crop(pic, image_file, 'image_id',
                             bbox_col_names, bbox_generated,
                             show_bbox = True)

    ax_2.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    ax_2.title.set_text(pic)

fig_2.tight_layout()
plt.show()

#%%
# [SPRINT 2] Cropping 10 random pictures
fig_3, axs_3 = plt.subplots(nrows = 5, ncols = 2, figsize = (15, 35))

for pic, ax_3 in zip(random_pics, axs_3.ravel()):

    crop_img = src.face_crop(pic, image_file, 'image_id',
                             bbox_col_names, bbox_generated,
                             show_bbox = False)

    ax_3.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    ax_3.title.set_text(pic)

fig_3.tight_layout()
plt.show()

#%%
# [SPRINT 2] Resizing of 10 random cropped pictures.
fig_4, axs_4 = plt.subplots(nrows = 5, ncols = 2, figsize = (15, 35))

for pic, ax_4 in zip(random_pics, axs_4.ravel()):

    resize_crop_img = src.face_crop(pic, image_file, 'image_id',
                             bbox_col_names, bbox_generated,
                             show_bbox = False,
                             resize = True)
    
    ax_4.imshow(cv2.cvtColor(resize_crop_img, cv2.COLOR_BGR2RGB))
    ax_4.title.set_text(pic)

fig_4.tight_layout()
plt.show()


# %%
# [Sprint 3] Task 1 - Dist per person

plt.figure(figsize=(20,6))
pd.DataFrame(identity.image_id.value_counts(ascending=True)).groupby('image_id').size().plot.bar()
plt.show()

# %%
# [Sprint 3] Task 2 - Graph distribution of atributes

attbs = pd.read_csv("./data/Anno/list_attr_celeba.txt", delim_whitespace=True)
dist_var_df = pd.DataFrame(columns = ['var', 'dist_1_rel', 'dist_0_rel'])
for col in attbs.columns:

    dist_1_rel = pd.DataFrame(attbs[col].value_counts(normalize = True)).loc[1,col]
    dist_0_rel = pd.DataFrame(attbs[col].value_counts(normalize = True)).loc[-1,col]

    dist_var_df = pd.concat((dist_var_df, pd.DataFrame([col, dist_1_rel, dist_0_rel]).transpose().rename(columns = {0:'var', 1:'dist_1_rel', 2:'dist_0_rel'})))

dist_var_df
# %% Plotting distributions

sorted_vars = dist_var_df.sort_values(by = "dist_0_rel")
sorted_vars = sorted_vars.set_index(sorted_vars['var'])

sorted_vars.plot(kind='bar', stacked=True)
plt.legend(['Yes', 'No'], loc = 'upper left')

# %% [Sprint 3] Task 2 - Correlation Matrix (PETER)

corr_df = src.corr_atrbs(attbs)

# %%
#positive associations
corr_df.query('coef > 0.3')

# %%
#negative associations
corr_df[corr_df['coef'] < - 0.3]

# %%
# Correlation matrix

src.corr_matrix(attbs)

# %%
# Plotting correlation matrix

src.corr_matrix(attbs, True)

# %%
#relative distributions of attributes
dist_var_df = pd.DataFrame(columns = ['var', 'dist_1_rel', 'dist_0_rel'])

for col in attbs.columns:

    dist_1_rel = pd.DataFrame(attbs[col].value_counts(normalize = True)).loc[1,col]
    dist_0_rel = pd.DataFrame(attbs[col].value_counts(normalize = True)).loc[-1,col]

    dist_var_df = pd.concat((dist_var_df, pd.DataFrame([col, dist_1_rel, dist_0_rel]).transpose().rename(columns = {0:'var', 1:'dist_1_rel', 2:'dist_0_rel'})))

dist_var_df

# %%
#joining annotations with attributes
df_joined = identity_filtered.merge(attbs.reset_index().rename(columns = {'index':'image'}), on = 'image')
df_joined

# %% [Sprint 3] Task 3 - Generate balance pairs (PETER)

src.balanced_pairs(df_joined, 10, ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive'])

# %%
# generating balanced pairs

src.balanced_pairs(df_joined, 15, ['Wearing_Necklace', 'Straight_Hair', 'Blond_Hair'])

# %% [Sprint 3] Task 4 - Resnet 50 with keras application

img_from_array = Image.fromarray(cv2.cvtColor(resize_crop_img, cv2.COLOR_BGR2RGB))

img_resnet = img_from_array.resize((224,224))
img_resnet

#%%
#ResNet50 model
model = ResNet50(weights='imagenet')

x = image.img_to_array(img_resnet)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

# %% [Sprint 3] Task 5 - Resnet Binary classification



# %% 
# [Sprint 3] 