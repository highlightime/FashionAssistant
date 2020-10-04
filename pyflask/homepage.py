#!/usr/bin/env python
# coding: utf-8

# In[3]:

import os
from flask import Flask, render_template, request, redirect, url_for, flash, session

from werkzeug.utils import secure_filename
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

import pandas as pd
import scipy.stats as stats
import sklearn.linear_model as linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import operator

app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
app.config["CACHE_TYPE"] = "null"


def makeMatrix(input_path):
    color_path=input_path
    screen=Image.open(color_path)
    a=[]
    plt.imshow(screen),plt.show()
    for i in range(300):
        k=screen.getpixel((i,1))
        line = []              
        for j in range(3):
            line.append(k[j])     
        a.append(line)
    dirout="matrix/new"
    np.save(dirout,a)
    
def colorMatrix():
    colors_dir = "matrix/"
    colors_paths = [colors_dir + f for f in listdir(colors_dir) if isfile(join(colors_dir, f))]
    aaa = []
    for num in range(len(colors_paths)):
        color_path=colors_paths[num]
        img = np.load(color_path) 
        img = Image.fromarray(np.uint8(img))
        img = img.resize((300, 3), Image.ANTIALIAS)
        aaa.append(np.array(img))
    color_path="matrix/new.npy"
    img = np.load(color_path) 
    img = Image.fromarray(np.uint8(img))
    img = img.resize((300, 3), Image.ANTIALIAS)
    aaa.append(np.array(img))
    
    return aaa
    
def colorMean(aaa):
    # Mean Image
    sums = np.zeros((3, 300))
    for x in range(len(aaa)):
        for i in range(3):
            for j in range(300):
                sums[i][j] += aaa[x][i][j]
    for idx in range(3):
        for idy in range(300):
            sums[idx][idy] /= len(aaa)
    return sums,aaa

def SquaredDistance(x, y):
    return (x-y)**2
def ImageDistance(img1, img2):
    distance = 0
    for i in range(3):
        for j in range(300):
            distance += SquaredDistance(img1[i][j], img2[i][j])      
    return distance

def calDistance(sums,aaa):
    mean_image = sums
    Distance_Dict = {}
    for i in range(len(aaa)):
        Distance_Dict[i] = ImageDistance(mean_image, aaa[i])
    return Distance_Dict

def sortDistance(Distance_Dict,start):
    sorted_d = sorted(Distance_Dict.items(), key=operator.itemgetter(1))
    for i in range(200):
        if(sorted_d[i][0]==start):
            print(sorted_d[i][0])
            if sorted_d[i-1][0]<100:
                rec_path='images/img_000000'+str(sorted_d[i-1][0])+'.jpg'
            else:
                rec_path='images/img_00000'+str(sorted_d[i-1][0])+'.jpg'
    recImg=cv2.imread(rec_path)
    plt.imshow(recImg),plt.show()
    cv2.imwrite('static/rec.jpg',recImg)

def image_color_cluster(flag,image_path, k = 5):
    image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    
    
    cluster = KMeans(n_clusters = k)
    cluster.fit(image)

    hist = make_histogram(cluster)
    colorBar = make_colorBar(hist, cluster.cluster_centers_)
    
    if(flag==0):
        dirout='images_modify/'+img_path[23]+img_path[24]+img_path[25]+'.jpg'
        print("[SAVE] Color Bar in "+dirout)
        cv2.imwrite(dirout,colorBar)
    else:
        dirout1='images_modify/barImg.jpg'
        dirout2='static/barImg.jpg'
        print("[SAVE] Color Bar in "+dirout1)
        cv2.imwrite(dirout1,colorBar)
        cv2.imwrite(dirout2,colorBar)
    
    return dirout2


# In[5]:


def make_histogram(cluster):   
# grab k number of clusters based on pixel
    numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
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
        cv2.rectangle(colorBar, (int(start), 0), (int(end), 50), color.astype("uint8").tolist(), -1)
        start = end

    return colorBar

def mygradCAM(input_path):
    Model = ResNet50
# load the pre-trained CNN model imagenet
    print("[INFO] loading model...")
    model = Model(weights="imagenet")
    
# load the image for predicting 
    origImg = cv2.imread(input_path)
    
# resize the image 
    resized = cv2.resize(origImg, (224, 224))
    
# preprocess the image 
    image = load_img(input_path, target_size=(224, 224))
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
    heatmap = cv2.resize(heatmap, (origImg.shape[1], origImg.shape[0]))
# put heatmap on the image
    (heatmap, output) = cam.overlay_heatmap(heatmap,origImg, alpha=0.5)

# print image predicted
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    output = np.vstack([origImg, heatmap, output])
    output = imutils.resize(output, height=700)
    plt.imshow(output)
    
# save the heatmap in images_modify/gradImg.jpg'
    dirout = 'images_modify/gradImg.jpg'
    print("[SAVE] in "+dirout)
    cv2.imwrite(dirout, heatmap)
    
    return dirout



def mygrabCUT(input_path,gradImg):
# load the image for predicting 
    img = cv2.imread(input_path)
    
# load the heatmap saved before
    img2 = cv2.imread(gradImg)
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
    
# save the image in 'static/cutImg.png'
    dirout = 'static/cutImg.png'
    print("[SAVE] in "+dirout)
    cv2.imwrite(dirout,imgout)
    
# load the image saved 
    dirin = dirout
    src = cv2.imread(dirin)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# make blackground transparent by adding alpha value
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    
# save the image in static/resultImg.png 
    dirout3 = 'static/resultImg.png'
    cv2.imwrite(dirout3,dst)
    
# crop the image as much as it can contain detected object
    a=Image.open(dirout3)
    a.size
    a.getbbox()
    a2=a.crop(a.getbbox())
    a2.save(dirout3)
    
    return dirout3


def dodo(input_path):
    gradImg=mygradCAM(input_path)
    cutImg=mygrabCUT(input_path,gradImg)
    barImg=image_color_cluster(1,cutImg)
    makeMatrix(barImg)
    aaa=colorMatrix()
    sums,aaa=colorMean(aaa)
    Distance_Dict=calDistance(sums,aaa)
    sortDistance(Distance_Dict,200)
    
    return barImg




# In[4]:




# In[5]:


#upload html rendering
@app.route('/upload')
def render_file():
    return render_template('upload.html')


# In[6]:


#upload file
@app.route('/fileUpload',methods=['GET','POST'])
def upload_file():
    if request.method=='POST':
        f=request.files['file']
        if f.filename == '':
            flash('No selected file. Try upload again!','error')
            return render_template('upload.html')
        else:
            f.save('/home/ec2-user/pyflask/fileUpload/'+secure_filename(f.filename))
            input_path='/home/ec2-user/pyflask/fileUpload/'+secure_filename(f.filename)
            result=dodo(input_path)
            return render_template('send.html')
        return render_template('upload.html')


# In[7]:


#show the output image
@app.route('/output',methods=['GET','POST'])
def output_file():
    return render_template('output.html')


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'public,no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response




if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(host='0.0.0.0', port=5000,debug=True)





