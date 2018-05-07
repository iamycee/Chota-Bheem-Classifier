import os
import scipy.ndimage
import numpy as np
from PIL import Image, ImageOps

MAIN_DIR = 'C:/Users/YC/Desktop/Chota Bheem Classifier'

def preprocess_images():
    if not os.path.exists(MAIN_DIR + '/processed_images'):
        os.makedirs(MAIN_DIR + '/processed_images')
        
        files = os.listdir(MAIN_DIR + '/raw_images')

        for file in files:
            os.chdir(MAIN_DIR + '/raw_images')       #go into the raw_images folder to handle particular character's folders
            print("exploring {}".format(file))
            images = os.listdir(file)       #store all image names of a character in images list
            for image_name in images:
                os.chdir(MAIN_DIR + '/raw_images/' + file)        #go into folder of bheem/chutki/jaggu/raju
                

                img = Image.open(image_name)
                img = img.resize((100, 100))
                img = ImageOps.grayscale(img)
                img_name, img_ext = os.path.splitext(image_name)
                print(img_name)
                
                os.chdir(MAIN_DIR + '/processed_images')
                img.convert('RGB').save('{}_resized{}'.format(img_name, img_ext))



#vectorize the images, store in an np array
def create_dataset():
    X = []
    y = []
    images = os.listdir(MAIN_DIR + '/processed_images')
    os.chdir(MAIN_DIR + '/processed_images')
    for image_name in images:
        arr = scipy.ndimage.imread(image_name).flatten() # Read the flattened image
        X.append(arr)
        #extract target names from the image_name
        image_target = image_name.split(' (')[0]        #split the name to just the name of character without the jargon, extension
        y.append(image_target)

    X = np.array(X)
    y = np.array(y)

    #We need a shuffled dataset as now they are organized into blocks of certain charaters
    y = y.reshape(len(y),1)
    dataset = np.hstack((X, y))
    np.random.shuffle(dataset)

    return dataset

#to see the images : regeneration:
#img = Image.fromarray(images_arr[2].reshape(100,100, -1))
#img.show()
