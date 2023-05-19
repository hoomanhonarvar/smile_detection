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
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_image_rescaled , fd


train_folders = ['datasets/train_folder/0', 'datasets/train_folder/1']
test_folders = ['datasets/test_folder/0', 'datasets/test_folder/1']
image_size = 64
pixel_depth = 255.0
image_depth = 3


def load_image(folder, min_num_images):
  """Load the image for a single smile/non-smile lable."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size, image_depth),
                         dtype=np.float32)
  image_index = 0
  for image in os.listdir(folder):
    image_file = os.path.join(folder, image)
    try:
      A=cv2.imread(image_file)
      hog,_=get_hog(A)
      lbp=get_lbp(image_file)
      # cv2.imshow("kir",A)
      # cv2.waitKey(0)
      # cv2.imshow("hog",hog)
      # cv2.waitKey(0)
      # cv2.imshow("lbp", lbp)
      # cv2.waitKey(0)
      image_data = (cv2.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
      # print(image_data)
      # print(hog.shape)
      # print(image_data[0].shape)
      image_data[:,:,0]=hog
      image_data[:,:,1]=lbp
      image_data[:,:,2]=cv2.imread(image_file,0)
      if image_data.shape != (image_size, image_size, image_depth):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[image_index, :, :, :] = image_data
      image_index += 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))

  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
# Pickling datasets/train_folder/0.pickle.
# Full dataset tensor: (1238, 64, 64, 3)
# Mean: -0.0335986
# Standard deviation: 0.247544
# Pickling datasets/train_folder/1.pickle.
# Full dataset tensor: (1562, 64, 64, 3)
# Mean: -0.0137995
# Standard deviation: 0.249232
# Pickling datasets/test_folder/0.pickle.
# Full dataset tensor: (600, 64, 64, 3)
# Mean: -0.0210533
# Standard deviation: 0.249451
# Pickling datasets/test_folder/1.pickle.
# Full dataset tensor: (600, 64, 64, 3)
# Mean: -0.00345457
# Standard deviation: 0.249467


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_image(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names

train_datasets = maybe_pickle(train_folders, 800)
test_datasets = maybe_pickle(test_folders, 200)
print("DONE")
print(train_datasets)
print(test_datasets)

def make_arrays(nb_rows, img_size, img_depth=3):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size, img_depth), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)

    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes  # 400
    tsize_per_class = train_size // num_classes  # 200

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        # print(pickle_file)

        try:
            with open(pickle_file, 'rb') as f:
                smile_nonsmile_set = pickle.load(f)
                # print(smile_nonsmile_set.shape)

                # let's shuffle the smile / nonsmile class
                # to have random validation and training set
                np.random.shuffle(smile_nonsmile_set)
                if valid_dataset is not None:
                    valid_smile_nonsmile = smile_nonsmile_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_smile_nonsmile
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_smile_nonsmile = smile_nonsmile_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_smile_nonsmile
                train_labels[start_t:end_t] = label

                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    return valid_dataset, valid_labels, train_dataset, train_labels

train_size = 2400
valid_size = 600
test_size = 600
print("ttt",train_datasets)
valid_size1, valid_labels1, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size)
valid_dataset, valid_labels, test_dataset, test_labels = merge_datasets(
  test_datasets, test_size, valid_size)

print("valid_size1",valid_size1)
print("valid_labels1",valid_labels1)
print("train_dataset",train_dataset.shape)
print(train_datasets[1][0][0])
print("train_labels",train_labels)


def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


print("train_dataset",train_dataset)
print("train_labels",train_labels)
pickle_file="hooman.pickle"
try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


num_labels = 2
num_channels = image_depth

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
# print('Training set', train_dataset, train_labels)
# print('Validation set', valid_dataset, valid_labels)
# print('Test set', test_dataset, test_labels)

# pretty_labels = {0: 'non-smile', 1: 'smile'}
# def disp_sample_dataset(dataset, labels):
#   print("labels",labels)
#   print(labels.shape)
#   print("dataset",dataset)
#   print(dataset.shape)
#   items = random.sample(range(len(labels)), 8)
#   i=0
#   for  item in (items):
#     plt.subplot(2, 4, i+1)
#     plt.axis('off')
#     i += 1
#     if labels[item][0]==1:
#         x=0
#     else:
#         x=1
#     plt.title(pretty_labels[x])
#     plt.imshow(dataset[item],interpolation='nearest')
#     plt.show()
# disp_sample_dataset(train_dataset, train_labels)



pickle_file = 'hooman.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
X_train, X_test, y_train, y_test = train_test_split(train_dataset, labels, test_size=0.3, random_state=0)