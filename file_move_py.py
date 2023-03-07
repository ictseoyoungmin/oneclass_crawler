from fileinput import filename
import numpy as np
import shutil
from keras import models
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2

def load_images(path=None,size=(224,224)):
    image_path = path

    # 디렉토리 내 모든 파일 불러오기
    img_list_jpg = [img for img in os.listdir(image_path) if img.endswith(".jpg")]  # 지정된 확장자만 필터링
    img_list = []
    img_names = []
    for j,i in enumerate(img_list_jpg):
        try:
            img = Image.open(image_path + i)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = np.array(img)
            if np.array_equal(img,np.zeros_like(img)):
                print(f"zeros array : [{j}]{i}")
                continue
            img = cv2.resize(img, size)
            img_list.append(img)
            img_names.append(i)
        except:
            pass
    img_np = np.array(img_list)  # 리스트를 numpy로 변환
    img_names_np = np.array(img_names) 
    print(img_np.shape, img_names_np.shape)
    
    return img_np,img_names_np

def draw(arr, where,filenames,ratio=2):
    n = len(arr)   
    if n == 0:
        return 0
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, 
                            figsize=(cols*ratio, rows*ratio), squeeze=False,constrained_layout=True)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
                axs[i,j].set_title('['+str(where[i*10+j])+']'+filenames[i*10+j])
            axs[i, j].axis('off')
    os.makedirs('./result',exist_ok=True)
    plt.savefig(f'./result/result{where[0]}.jpg')# plt.show()
    
def load_filenames(path=None):
    img_list_jpg = [img.split('\\')[-1] for img in os.listdir(path) if img.endswith(".jpg")]  
    return np.array(img_list_jpg)

class FileMove():
    def __init__(self):
        self.base_path      = ''
        self.crawled_path   = ''
        self.key            = ''
        self.class_path     = ''
        self.store_path     = ''
        self.data_path      = ''
        self.model_path     = '' 
        self.target_path    = ''

    def set_para(self,key,base_path,crawled_path,model_path):
        self.base_path      = base_path
        self.crawled_path   = crawled_path
        self.key            = key
        self.data_path      = crawled_path[:-len(key)]
        self.class_path     = self.key+'/' # cat/
        self.store_path     = self.base_path+'/data/'+f'{self.key}_result/'
        self.model_path     = model_path
        self.target_path    = self.data_path+self.class_path
                                
    def get_para(self):
        return self.store_path
    
    def set_second_target(self):
        self.target_path = self.store_path+'abnormal/'
        # self.issecond = True

    def move(self,filenames,label,issecond=False):
        for filename in filenames:
            if label == 1:
                shutil.move(self.target_path+filename,self.store_path+self.class_path+'/'+filename)
            else :
                if issecond:pass
                shutil.move(self.target_path+filename,self.store_path+'abnormal/'+filename)

    def run(self):
        os.makedirs(self.store_path+self.key,exist_ok=True)
        os.makedirs(self.store_path+'abnormal',exist_ok=True)

        # main model
        print(self.model_path)
        model = models.load_model(self.model_path)
        
        
        # load images
        images,filenames = load_images(self.target_path)

        # predict
        print(f'model predict : {self.model_path}')
        pre = np.argmax(model.predict(images),axis=1)
        # (300,)
        # one class classification result - plt image file save, image move
        print('classification.')
        for label in [0,1]:
            where =np.where(pre == label)
            print(self.key+'..' if label != 0 else "abnormal..") 
            draw(images[where],where[0],filenames)
            self.move(filenames[where],label)
            print("\n")
    
    
    
    











