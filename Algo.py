#!/usr/bin/env python
# coding: utf-8

# In[2]:


from mylib.gradcam import GradCAM
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import matplotlib.image as mpimg
import PIL.Image as pilimg
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from os import listdir
from os.path import isfile, join
from keras import backend as K
from PIL import Image
import sys

K.clear_session()


# In[3]:


img_dir = "images/"
img_paths = [img_dir + f for f in listdir(img_dir) if isfile(join(img_dir, f))]


# In[4]:


total=len(img_paths)


# In[4]:


Model = ResNet50
# load the pre-trained CNN model imagenet
print("[INFO] loading model...")
model = Model(weights="imagenet")


# In[5]:


for n in range(total):
# load the image for predicting 
    img_path=img_paths[n]
    orig = cv2.imread(img_path)
    
# resize the image 
    resized = cv2.resize(orig, (224, 224))
    
# preprocess the image 
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    
# predict image to specific class i
    result = model.predict(image)
    i = np.argmax(result[0])

# image is decoded by imagenet_utils
    decode = imagenet_utils.decode_predictions(result)
    (imagenetID, label, prob) = decode[0][0]
    
# print label of decoded image 
    label = "{}: {:.2f}%".format(label, prob * 100)
    print("[INFO] {}".format(label))

# use ResNet50 model, class i to gradcam
    cam = GradCAM(model, i)
# compute heatmap for using mask
    heatmap = cam.compute_heatmap(image)

# resize heatmap 
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
# put heatmap on the image
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

# print image predicted
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    output = np.vstack([orig, heatmap, output])
    output = imutils.resize(output, height=700)
    plt.imshow(output)
# save the heatmap in images_modify/img_00000***_1.jpg
    dirout = 'images_modify/img_00000'+img_path[16]+img_path[17]+img_path[18]+'_1.jpg'
    print("[SAVE] in "+dirout)
    cv2.imwrite(dirout, heatmap)
    


# In[6]:


for n in range(total):
# load the image for predicting 
    img_path=img_paths[n]
    img = cv2.imread(img_path)
    
# load the heatmap saved before
    img2 = cv2.imread('images_modify/img_00000'+img_path[16]+img_path[17]+img_path[18]+'_1.jpg')
    mask = np.zeros(img2.shape[:2],np.uint8)
    
# create temporary array 
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
# do grabcut with the mask
    rect = (50,50,450,290)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
# mask2 is based on Grabcut output  
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    imgout = img*mask2[:,:,np.newaxis]
    
# print the image
    plt.imshow(imgout),plt.colorbar(),plt.show()
    
# save the image in images_modify/img_00000***_2.jpg
    dirout = 'images_modify/img_00000'+img_path[16]+img_path[17]+img_path[18]+'_2.jpg'
    print("[SAVE] in "+dirout)
    cv2.imwrite(dirout,imgout)


# In[16]:


for n in range(total):
# load the image saved 
    img_path=img_paths[n]
    dirin = 'images_modify/img_00000'+img_path[16]+img_path[17]+img_path[18]+'_2.jpg'
    src = cv2.imread(dirin)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# make blackground transparent by adding alpha value
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    
# save the image in images_modify/img_00000***_3.png 
    dirout = 'images_modify/img_00000'+img_path[16]+img_path[17]+img_path[18]+'_3.png'
    print("[SAVE] in "+dirout)
    cv2.imwrite(dirout,dst)
    
# crop the image as much as it can contain detected object
    a=Image.open(dirout)
    a.size
    a.getbbox()
    a2=a.crop(a.getbbox())
    plt.imshow(a2),plt.show()
    
# save the image in images_modify/img_00000***_4.png
    dirout = 'images_modify/img_00000'+img_path[16]+img_path[17]+img_path[18]+'_4.png'
    print("[SAVE] in "+dirout)
    a2.save(dirout)
    
    


# In[6]:


def image_color_cluster(image_path, k = 10):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    
    cluster = KMeans(n_clusters = k)
    cluster.fit(image)

    hist = make_histogram(cluster)
    colorBar = make_colorBar(hist, cluster.cluster_centers_)
    
    dirout='images_modify/'+img_path[23]+img_path[24]+img_path[25]+'.jpg'
    print("[SAVE] Color Bar in "+dirout)
    cv2.imwrite(dirout,colorBar)
    
    plt.figure()
    plt.axis("off")
    plt.imshow(colorBar)
    plt.show()


# In[5]:


def make_histogram(cluster):   
# grab k number of clusters based on pixel
    numLabels = np.arange(1, len(np.unique(cluster.labels_)) + 1)
    (hist, bins) = np.histogram(cluster.labels_, bins=numLabels)

# sums it to one to create histogram
    hist = hist.astype("float")
    hist /= hist.sum()
    
    return hist

def make_colorBar(hist, centroids):
# prepare bar to visualize
    colorBar= np.zeros((50, 300, 3), dtype="uint8")
    start = 0

# calculate percentages of each color
    for (per, color) in zip(hist, centroids):
        end = start + (per * 300)
        cv2.rectangle(colorBar, (int(start), 0), (int(end), 50),
                      color.astype("uint8").tolist(), -1)
        start = end

    return colorBar


# In[10]:


#print array max error checking
#import sys
#np.set_printoptions(threshold=sys.maxsize)


# In[ ]:


for n in range(total):
# load the image cropped
    img_path=img_paths[n]
    img_path='images_modify/img_00000'+img_path[16]+img_path[17]+img_path[18]+'_4.png'
    image = Image.open(img_path)
    plt.imshow(image)
    
# color clustering with KMEANS
    print("[PRINT] color quantitizing"+img_path)
    image_color_cluster(img_path)



# In[12]:


# delete some files from dir
'''
import os
img_dir=img_dir = "images_modify/"
img_paths = [img_dir + f for f in listdir(img_dir) if isfile(join(img_dir, f))]

for n in range(len(img_paths)):
    img_path=img_paths[n]
    try:
        os.remove(r'images_modify/img_00000'+img_path[16]+img_path[17]+img_path[18]+'_4.png')
        os.remove(r'images_modify/img_00000'+img_path[16]+img_path[17]+img_path[18]+'_1.jpg')
        os.remove(r'images_modify/img_00000'+img_path[16]+img_path[17]+img_path[18]+'_2.jpg')
        os.remove(r'images_modify/img_00000'+img_path[16]+img_path[17]+img_path[18]+'_3.png')
    except FileNotFoundError:
        pass
'''


# In[13]:

'''
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
model = VGG16(weights='imagenet', include_top=False)
model.summary()

img_path = '../../pyflask/fileUpload/7.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

vgg16_feature = model.predict(img_data)

print(vgg16_feature.shape)
'''

# In[ ]:




