import cv2
import os
import numpy as np
import time
from scipy import ndimage
from six.moves import cPickle as pickle
from skimage.feature import hog
from skimage import data, exposure
from sklearn.datasets import make_blobs
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score



def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value
def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x - 1, y - 1))
    val_ar.append(get_pixel(img, center, x - 1, y))
    val_ar.append(get_pixel(img, center, x - 1, y + 1))
    val_ar.append(get_pixel(img, center, x, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y))
    val_ar.append(get_pixel(img, center, x + 1, y - 1))
    val_ar.append(get_pixel(img, center, x, y - 1))
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

def get_lbp(image):
    img_bgr = image.copy()
    height, width, _ = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr,
                            cv2.COLOR_BGR2GRAY)
    img_lbp = np.zeros((height, width),
                       np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

    return img_lbp





def get_hog(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_image_rescaled , fd


filename = 'final_model.sav'
target=[]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
# with open('final_model.sav', 'rb') as f:
#     clf2 = pickle.load(f)
load_model=pickle.load(open(filename,'rb'))
for i in range (0,100000):
    os.system('cls')
    time.sleep(0.4)

    flat_data_arr = []

    # reads frames from a camera
    ret, img = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detects eyes of different sizes in the input image

        #To draw a rectangle in eyes
        # eyes = eye_cascade.detectMultiScale(roi_gray)
    img_copy = img.copy()
    dsize=(64,64)
    img_copy=cv2.resize(img_copy,dsize)
    hog_pic, _ = get_hog(img_copy)
    lbp = get_lbp(img_copy)
    # print(img_copy.shape)
    # print(hog_pic.shape)
    # img_copy[:, :, 0] = hog_pic
    # img_copy[:, :, 1] = lbp
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_copy[:, :, 2] = img_gray

    img_test=np.zeros([img_copy.shape[0],img_copy.shape[1],2])
    img_test[:,:,0]=hog_pic
    img_test[:,:,1]=lbp
    # df = pd.DataFrame(img_test)
    flat_data_arr.append(img_test.flatten())

    flat_data = np.array(flat_data_arr)

    # print(flat_dat)
    # df = pd.DataFrame(flat_data)
    # target.append(0)
    # df['Target'] = target
    # df.shape
    # x = df.iloc[:, :-1]
    # print(len(x))
    ynew = load_model.predict(flat_data)
    # print(ynew)

    # Wait for Esc key to stop
    # k = cv2.waitKey(30) & 0xFF
    # if k == 32:
    #     break
    if ynew[0]==1:
        # print(ynew)
        # print(img_copy)
        print("you smiled :)))")
        cv2.imwrite(str(i)+"im_smile.jpg",img)
    # target.pop()
    # removelist=[]
    # removelist.append(df)
    # del removelist
    del ynew
    del flat_data



cap.release()
cv2.destroyAllWindows()
