import cv2
#import cvlib as cv
import src.face_detection as cv
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import matthews_corrcoef
import random
import numpy as np
import seaborn as sns


def face_crop(image_name, image_file, image_id_col_name, bbox_col_names, bbox_df, show_bbox = False, resize = False):
# Loading Image
    image_path = image_file + image_name
    img = cv2.imread(image_path)

    # Setting bounding box coordinates
    startX = bbox_df[bbox_df[image_id_col_name] == image_name][bbox_col_names['x_start']].values[0]
    startY = bbox_df[bbox_df[image_id_col_name] == image_name][bbox_col_names['y_start']].values[0]
    endX = startX + bbox_df[bbox_df[image_id_col_name] == image_name][bbox_col_names['width']].values[0]
    endY = startY + bbox_df[bbox_df[image_id_col_name] == image_name][bbox_col_names['height']].values[0]

    if show_bbox == True:
        cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)
        output_img = img
        return output_img
    else:
        # Cropping and saving boounding box
        crop_img = img[startY:endY, startX:endX]
        output_img = crop_img

        if resize == True:
            output_img = cv2.resize(crop_img, (300, 300))
            return output_img
        else:
            return output_img

def bbox_engine(image, image_id = '', method = 0, m1_scale_factor = 1.1, m1_min_neighbors = 6):
    if method == 0:
        faces, confidences = cv.detect_face(image)
        for face in faces:
            #(startX,startY) = face[0],face[1]
            #(endX,endY) = face[2],face[3]

            bbox = {
        #       "image_id" : image_id,
                "x_1" : face[0],
                "y_1" : face[1],
                "width" : face[2] - face[0],
                "height" : face[3] - face[1],
                'x_end' : face[2],
                'y_end' : face[3]}
        return bbox

    if method == 1:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image = image, scaleFactor = m1_scale_factor, minNeighbors = m1_min_neighbors)
        #for face in faces:
            #(startX,startY) = face[0],face[1]
            #(endX,endY) = face[2],face[3]

        bbox = {
    #       "image_id" : image_id,
            "x_1" : faces[0][0],
            "y_1" : faces[0][1],
            "width" : faces[0][2],
            "height" : faces[0][3],
            'x_end' : faces[0][0] + faces[0][2],
            'y_end' : faces[0][1] + faces[0][3]}
    return bbox

def bbox_engine_img_input(pic,image_path, m1_scale_factor = 1.1, m1_min_neighbors = 13):
    img = cv2.imread(image_path+pic)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image = img, scaleFactor = m1_scale_factor, minNeighbors = m1_min_neighbors)
    
    #New feature of the function - iteration over the neighbors parameter in case of empty bounding boxes.
    #Perhaps, we could also perform a hyperparameter tuning in order to choose an optimal number of neighbors which has the lowest validation error.
    while len(faces) < 1:
        m1_min_neighbors -= 1
        faces = face_cascade.detectMultiScale(image = img, scaleFactor = m1_scale_factor, minNeighbors = m1_min_neighbors)
        if m1_min_neighbors < 0:
            break
    
        #for face in faces:
            #(startX,startY) = face[0],face[1]
            #(endX,endY) = face[2],face[3]
    if len(faces) > 0:

        bbox = {
    #       "image_id" : image_id,
            "x_1" : faces[0][0],
            "y_1" : faces[0][1],
            "width" : faces[0][2],
            "height" : faces[0][3],
            'x_end' : faces[0][0] + faces[0][2],
            'y_end' : faces[0][1] + faces[0][3]}

        return bbox

bbox_col_names = {
    'x_start' : 'x_1',
    'y_start' : 'y_1',
    'width' : 'width',
    'height' : 'height',
    'x_end' : '',
    'y_end' : ''}

def bbox_verification( bbox_annotation, bbox_generated, image_len):

    calc_df = bbox_annotation.merge(bbox_generated, on= "image_id", how='left', suffixes= ['_anno', '_generated'])
    calc_df = calc_df.merge(image_len, on = 'image_id', how = 'left')

    result_df = pd.DataFrame(columns= ['image_id'])

    result_df['image_id'] = calc_df['image_id']
    result_df['image_len'] = calc_df['len']
    result_df['x_1_diff'] = ((calc_df['x_1_anno'] - calc_df['x_1_generated'])**2)**(1/2)
    result_df['y_1_diff'] =  ((calc_df['y_1_anno'] - calc_df['y_1_generated'])**2)**(1/2)
    result_df['width_diff'] = ((calc_df['width_anno'] - calc_df['width_generated'])**2)**(1/2)
    result_df['height_diff'] = ((calc_df['height_anno'] - calc_df['height_generated'])**2)**(1/2)
    result_df['total_diff'] = (result_df['x_1_diff'] + result_df['y_1_diff'] + result_df['width_diff'] + result_df['height_diff'])
    result_df['relative_diff'] = (result_df['total_diff']/result_df['image_len'])

    return result_df

def bbox_show_compare(image_name, image_file, image_id_col_name, bbox_col_names, bbox_df_anno, bbox_df_generated):
# Loading Image

    fig_1, axs_1 = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 7))

    #axs_1.ravel()

    crop_img = face_crop(image_name, image_file, image_id_col_name,
                                bbox_col_names, bbox_df_anno,
                                show_bbox = True)
    axs_1[0].imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

    crop_img2 = face_crop(image_name, image_file, image_id_col_name,
                                bbox_col_names, bbox_df_generated,
                                show_bbox = True)
    axs_1[1].imshow(cv2.cvtColor(crop_img2, cv2.COLOR_BGR2RGB))

    fig_1.tight_layout()
    plt.show()

def corr_atrbs(atrbs):
    dictt = {}
    atrbs_copy = atrbs.copy()

    for col in atrbs_copy.columns:
        temp = atrbs_copy.drop(col, axis = 1)

        for col2 in temp.columns:
            coef = matthews_corrcoef(atrbs_copy[col], temp[col2])
            dictt[col+'_&_'+col2] = coef
    
        atrbs_copy.drop(col, axis = 1, inplace = True)

    corr_df = pd.DataFrame([list(dictt.keys()), list(dictt.values())]).transpose().rename(columns = {0:'combos', 1:'coef'}).sort_values(by = 'coef')

    return corr_df

def corr_matrix(atr_df, plot = False):

    corr_mat = pd.DataFrame(columns = atr_df.columns, index = atr_df.columns)

    for row in corr_mat.index:
        for col in corr_mat.index:
            coeff = matthews_corrcoef(atr_df[row], atr_df[col])
            corr_mat.loc[row, col] = coeff

    if plot:
        plt.figure(figsize=(20,15))
        corr_mat = corr_mat.fillna(0.0)
        mask = np.triu(np.ones_like(corr_mat))
        cax = sns.heatmap(corr_mat, vmin = -1.0, vmax = 1.0, mask = mask, cmap ='plasma')
        cax.tick_params(labelsize=10)
        plt.show()
        
    else:
        return corr_mat

def balanced_pairs(joined_df, n, atrbs):

    pairs_df = pd.DataFrame(columns = ['pic_1', 'pic_2', 'label'])

    joined_df_grouped = joined_df.groupby(atrbs)
    balanced_df = joined_df_grouped.apply(lambda x: x.sample(joined_df_grouped.size().min()).reset_index(drop=True))

    if n % 2 != 0:
        adj_list = random.sample([0, 1], k = 2)
        range_same = int(n/2) + adj_list[0]
        range_diff = int(n/2) + adj_list[1]
    else:
        range_same = n/2
        range_diff = n/2
 
    for i in range(int(range_same)):
        random_id_same = random.sample(list(balanced_df['image_id']), k = 1)
        while balanced_df.loc[balanced_df['image_id'] == random_id_same[0],'image'].shape[0] < 2:
            random_id_same = random.sample(list(balanced_df['image_id']), k = 1)
        random_same_pics = random.sample(list(balanced_df.loc[balanced_df['image_id'] == random_id_same[0],'image']), k = 2)

        hehehe = pd.DataFrame({'pic_1': [random_same_pics[0]], 'pic_2': [random_same_pics[1]], 'label':1})
        pairs_df = pd.concat((pairs_df, hehehe))

    for j in range(int(range_diff)):
        random_id_same1 = random.sample(list(balanced_df['image_id']), k = 1)
        random_id_same2 = random.sample(list(balanced_df.loc[balanced_df['image_id'] != random_id_same1[0],'image_id']), k = 1)

        random_same_pic1 = random.sample(list(balanced_df.loc[balanced_df['image_id'] == random_id_same1[0],'image']), k = 1)
        random_same_pic2 = random.sample(list(balanced_df.loc[balanced_df['image_id'] == random_id_same2[0],'image']), k = 1)

        hahaha = pd.DataFrame({'pic_1': random_same_pic1, 'pic_2': random_same_pic2, 'label': 0})
        pairs_df = pd.concat((pairs_df, hahaha))

    final_df = pairs_df.merge(joined_df[['image_id', 'image']+atrbs], left_on = 'pic_1', right_on = 'image').\
                                drop('image', axis=1).\
                            merge(joined_df[['image_id', 'image']+atrbs], left_on = 'pic_2', right_on = 'image', suffixes= ('_1', '_2')).\
                                drop('image', axis=1)

    return final_df.iloc[random.sample(list(final_df.index), final_df.shape[0]),:].reset_index(drop = True)
