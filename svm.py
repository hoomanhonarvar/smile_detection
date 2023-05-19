import cv2
import os
import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle
from skimage.feature import hog
from skimage import data, exposure
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
    img_bgr = cv2.imread(image, 1)
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


catagoris=['0','1']
flat_data_arr=[]
num_train=50
target_arr=[]
data_dir='./datasets/train_folder/'


for i in catagoris:
    x=0
    print(f'loading ... catagoris:{i}')
    path=os.path.join(data_dir,i)
    # print(path)
    for img in os.listdir(path):
        # print(img)
        # print(os.path.join(path,img))
        img_array=cv2.imread(os.path.join(path,img))
        # print(img_array)
        hog_pic,_=get_hog(img_array)
        lbp=get_lbp(os.path.join(path,img))
        img1=np.zeros([img_array.shape[0],img_array.shape[1],2])
        img1[:,:,0]=hog_pic
        img1[:,:,1]=lbp
        # print(img_array)
        flat_data_arr.append(img1.flatten())
        target_arr.append(catagoris.index(i))

        x+=1
        if x==num_train:
            break
    print(f'loaded category:{i} successfully')


flat_data=np.array(flat_data_arr)
target=np.array(target_arr)

df=pd.DataFrame(flat_data)
df['Target']=target
df.shape


#input data
x=df.iloc[:,:-1]

#output data
y=df.iloc[:,-1]
y=df.drop['Target']



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# Defining the parameters grid for GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.0001, 0.001, 0.1, 1],
              'kernel': ['rbf', 'poly']}

# Creating a support vector classifier
svc = svm.SVC(probability=True)

# Creating a model using GridSearchCV with the parameters grid
model = GridSearchCV(svc, param_grid)

# Training the model using the training data
model.fit(x_train,y_train)

# Testing the model using the testing data
y_pred = model.predict(x_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_pred, y_test)

# Print the accuracy of the model
print(f"The model is {accuracy*100}% accurate")

filename = 'final_model.sav'
pickle.dump(model, open(filename, 'wb'))


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)