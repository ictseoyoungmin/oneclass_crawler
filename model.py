import os
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from keras.models import Model
from keras.layers import *
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.metrics import  accuracy_score


SHAPE = (224,224,3)

class u():
    # SHAPE = (224,224,3)

    # image list 10 * a + b 출력
    def draw(arr, ratio=1):
        n = len(arr)   
        if n == 0:
            return 0
        rows = int(np.ceil(n/10))
        cols = n if rows < 2 else 10
        fig, axs = plt.subplots(rows, cols, 
                                figsize=(cols*ratio, rows*ratio), squeeze=False)
        for i in range(rows):
            for j in range(cols):
                if i*10 + j < n:   
                    axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
                axs[i, j].axis('off')
        plt.show()

    ### GENERATOR WRAPPER TO CREATE FAKE LABEL ###
    def wrap_generator(generator):
        
        while True:
            x,y = next(generator)
            y = tf.keras.utils.to_categorical(y)
            zeros = tf.zeros_like(y) + tf.constant([1.,0.])
            y = tf.concat([y,zeros], axis=0)
            yield x,y

    def get_model_pre(train=True):
        """return Pseudo-labeling Model"""
        pre_process = Lambda(preprocess_input)
        vgg = VGG16(weights = 'imagenet', include_top = True, input_shape = SHAPE)
        vgg = Model(vgg.input, vgg.layers[-3].output)
        vgg.trainable = False
        
        inp = Input(SHAPE)
        vgg_16_process = pre_process(GaussianNoise(1e-8)(inp))
        vgg_out = vgg(vgg_16_process)
        
        noise = Lambda(tf.zeros_like)(vgg_out)
        noise = GaussianNoise(0.1)(noise) # 가우시안 노이즈로 과대적합

        if train:
            x = Lambda(lambda z: tf.concat(z, axis=0))([vgg_out,noise])
            x = Activation('relu')(x)
        else:
            x = vgg_out

        x = Dense(256, activation='relu')(x) # 256,128 
        x = Dense(256, activation='relu')(x)
        out = Dense(2, activation='softmax')(x)

        model = Model(inp, out)
        model.compile(tf.keras.optimizers.Adam(lr=1e-4), 
        loss='binary_crossentropy',
        metrics=['accuracy'])
        
        return model

    def get_model_paper(train=True):
        pre_process = Lambda(preprocess_input)
        vgg = VGG16(weights = 'imagenet', include_top = True, input_shape = (224,224,3))
        vgg = Model(vgg.input, vgg.layers[-3].output)
        vgg.trainable = False
        
        inp = Input((224,224,3))
        vgg_16_process = pre_process(GaussianNoise(1e-8)(inp))
        vgg_out = vgg(vgg_16_process)
        
        noise = Lambda(tf.zeros_like)(vgg_out)
        noise = GaussianNoise(0.01)(noise)

        if train:
            x = Lambda(lambda z: tf.concat(z, axis=0))([vgg_out,noise])
            x = Activation('relu')(x)
        else:
            x = vgg_out
            
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        out = Dense(2, activation='softmax')(x)

        model = Model(inp, out)
        model.compile(tf.keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy')
        
        return model

    def get_model_feature_extractor():
        global SHAPE 
        pre_process = Lambda(preprocess_input)
        vgg = VGG16(weights = 'imagenet', include_top = True, input_shape = SHAPE)
        vgg = Model(vgg.input, vgg.layers[-3].output)
        vgg.trainable = False
        
        inp = Input(SHAPE)
        vgg_16_process = pre_process(GaussianNoise(1e-8)(inp))
        vgg_out = vgg(vgg_16_process) # shape (,4096)

        feature_extractor = Model(inp, vgg_out)
                             
        return feature_extractor

    def get_label_test(test_gen):
        test_num = test_gen.samples
        label_test = []
        for i in range((test_num // test_gen.batch_size)+1):
            X,y = test_gen.next()
            label_test.append(y)
                
        label_test = np.argmax(np.vstack(label_test), axis=1)
        print(label_test.shape)
        
        return label_test
    
    
# 1. 사용자의 사전 훈련용 이미지를 받아야 함
# 2. 사전 훈련이미지를 train/key/에 저장 
# ex) train/cat/cat0.jpg, ... ,cat300.jpg
# 3. 훈련이미지를 불러와 학습 후 모델 models/에 저장

class Train():
    def __init__(self):
        self.key            = ''    
        self.base_path      = ''    # 현재 dir
        self.train_path     = ''    # 사용자의 보유중인 훈련이미지 저장 dir
        self.models_path    = ''    # 모델 저장 dir
        self.model_name     = ''    # 모델 이름
        self.store_path     = ''    # pseudo-labeled path 겸 normal path
        self.abnormal_path  = ''    # abnormal 에서 normal 찾기

    def set_para(self,main_key='',main_base_path='',main_store_path=''):
        self.key            = main_key
        self.base_path      = main_base_path # main_base_path + ''
        self.train_path     = self.base_path + '/train/'
        self.models_path    = self.base_path + '/models/'
        self.store_path     = main_store_path

    def get_para(self):
        """ retrun model_path """
        return os.path.join(self.models_path,self.model_name) 
         
    def run(self):
        batch_size = 32
        MAIN_EPOCHS = 40

        os.makedirs(self.models_path,exist_ok=True)

        train_datagen = ImageDataGenerator()
        test_datagen = ImageDataGenerator()
        train_generator = train_datagen.flow_from_directory(
                    self.train_path,
                    target_size = (SHAPE[0], SHAPE[1]),
                    batch_size = batch_size,
                    class_mode = 'categorical',
                    shuffle = False,
                    classes = [self.key] # 389 / 76
            )
        test_generator = test_datagen.flow_from_directory(
                    # 실제 서버에서는 필요 x
                    '../data/cat_dog/' + 'test/',
                    target_size = (SHAPE[0], SHAPE[1]),
                    batch_size = batch_size,
                    class_mode = 'categorical',
                    shuffle = True,
                    seed=42,
                    classes = ['dogs','cats']
        )

        ############   1st classification code ##################
        train_model = u.get_model_pre()
        train_model.fit(u.wrap_generator(train_generator),
                        steps_per_epoch=train_generator.samples/train_generator.batch_size, 
                        epochs=MAIN_EPOCHS)
        
        ####### save
        predict_model = u.get_model_pre(train=False)
        predict_model.set_weights(train_model.get_weights())
        self.model_name = f'model_'+str(self.key)+'.h5'
        predict_model.save(self.models_path+self.model_name)
        
        
        ####### test - 실제 서버에서는 필요 x
        ground_truth = u.get_label_test(test_generator)
        pred_test = np.argmax(predict_model.predict(test_generator), axis=1)
        main_acc = accuracy_score(ground_truth, pred_test)
        print('ACCURACY:', main_acc)
        
        ###################################################
        

    def cat_data_generator(self):
        """generator 합성  ( user data + pseudo-labeled data)"""

        user_datagen    = ImageDataGenerator()
        pseudo_datagen  = ImageDataGenerator()

        user_generator  = user_datagen.flow_from_directory(
                        self.train_path,
                        target_size = (SHAPE[0], SHAPE[1]),
                        batch_size = len(os.listdir(self.train_path+self.key+'/')),
                        class_mode = 'categorical',
                        shuffle = False,
                        classes = [self.key] 
                    )
        pseudo_generator  = pseudo_datagen.flow_from_directory(
                        self.store_path,
                        target_size = (SHAPE[0], SHAPE[1]),
                        batch_size = len(os.listdir(self.store_path+self.key+'/')),
                        class_mode = 'categorical',
                        shuffle = False,
                        classes = [self.key] 
                    )
        flow1_data = user_generator.next()
        flow2_data = pseudo_generator.next()
        print(flow1_data[0].shape,flow2_data[0].shape)

        train_x_cat = np.concatenate([flow1_data[0],flow2_data[0]],axis=0)
        train_y_cat = np.concatenate([flow1_data[1],flow2_data[1]],axis=0)
        
        return train_x_cat,train_y_cat

    def run2(self):
        batch_size = 32
        MAIN_EPOCHS = 40

        os.makedirs(self.models_path,exist_ok=True)
        train_gen = ImageDataGenerator()
        x,y = self.cat_data_generator()
        print(x.shape,y.shape)

        ############   2st classification code ##################
        train_model = u.get_model_paper()
        train_model.fit(u.wrap_generator(train_gen.flow(x,y,batch_size=batch_size)),
                        steps_per_epoch=len(y)/batch_size,    
                        epochs=MAIN_EPOCHS)
        
        ####### save
        predict_model = u.get_model_paper(train=False)
        predict_model.set_weights(train_model.get_weights())
        self.model_name = f'model_'+str(self.key)+'.h5'
        predict_model.save(self.models_path+self.model_name)
        
        
        ####### test cat, dog 위치 변환하며 테스트
        test_datagen = ImageDataGenerator()
        test_generator = test_datagen.flow_from_directory(
                            # 실제 서버에서는 필요 x
                            '../data/cat_dog/' + 'test/',
                            target_size = (SHAPE[0], SHAPE[1]),
                            batch_size = batch_size,
                            class_mode = 'categorical',
                            shuffle = True,
                            seed=42,
                            classes = ['cats','dogs']
                )

        ground_truth = u.get_label_test(test_generator)
        pred_test = np.argmax(predict_model.predict(test_generator), axis=1)
        main_acc = accuracy_score(ground_truth, pred_test)
        print('ACCURACY:', main_acc)
        
        ###################################################


        