
# coding: utf-8

# In[1]:

from PIL import Image
import os
import glob
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split


# In[2]:

from resizeimage import resizeimage


# In[3]:

from keras.utils import to_categorical


# In[4]:

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


# In[5]:

images=[]
y=[]


# In[6]:

c=0
for i in range(0,2):
    if(i==0):
        path = u'C:\\Users\\Vaibhav\\Desktop\\MCA testing\\good'
        for infile in glob.glob( os.path.join(path, u'*') ):
            im= Image.open(infile)
            im1 = resizeimage.resize_cover(im, [300, 300]) 
            img1 = np.asarray(im1,np.uint8)
            #img1 = img1.resize(img1,(400,200))
            #print img1.format, img1.size, img1.mode
            #print img1.shape
            images.append(img1)
            y.append(int(i))
    elif(i==1):
        path = u'C:\\Users\\Vaibhav\\Desktop\\MCA testing\\bad'
        for infile in glob.glob( os.path.join(path, u'*') ):
            im= Image.open(infile)
            im1 = resizeimage.resize_cover(im, [300, 300]) 
            img1 = np.asarray(im1,np.uint8)
            #print img1.format, img1.size, img1.mode
            #print img1.shape
            images.append(img1)
            y.append(int(i))
        


# In[7]:

X = np.array(images)


# In[8]:

print(X.shape)


# In[ ]:




# In[ ]:




# In[9]:

#X = np.array(images)
y= np.array(y)


# In[10]:

y


# In[ ]:




# In[11]:

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,test_size=0.2)


# In[12]:

print(y_test.shape)


# In[13]:

X_train = X_train.reshape(-1, 300,300, 3)
X_test = X_test.reshape(-1, 300,300, 3)
X_train.shape, X_test.shape


# In[14]:

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.
X_test = X_test / 255.


# In[15]:

train_Y_one_hot = to_categorical(y_train)
test_Y_one_hot = to_categorical(y_test)

# Display the change for category label using one-hot encoding
print('Original label:', y_train[0])
print('After conversion to one-hot:', train_Y_one_hot[0])


# In[25]:

batch_size = 3
epochs = 1000
num_classes = 2


# In[26]:

fashion_model = Sequential() #declaration of the model


# In[27]:

#Convolution layer 2D, RELU activation function, layer output dimentionality should not change
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(300,300,3),padding='same'))


# In[28]:

#LeakyReLU activation function. some feature will be selected
fashion_model.add(LeakyReLU(alpha=0.1))


# In[29]:

#selects feature with maximum relevance, image size decreases.
fashion_model.add(MaxPooling2D((2, 2),padding='same'))


# In[30]:

fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#flatten makes the image an array for the whole colm
fashion_model.add(Flatten())
#to remove errors, is fully connected.
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_classes, activation='softmax'))


# In[31]:

#fashion_model.compile(keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
#fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#defining type of looses(after the forward pass we can check how much Y is different from X)
#optimizers( lr = learning rate)
fashion_model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])


# In[32]:

#print the number of layers
fashion_model.summary()


# In[ ]:

#epochs = 5000
with tf.device('/device:GPU:0'):
    fashion_train = fashion_model.fit(X_train, train_Y_one_hot, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test, test_Y_one_hot))


# In[ ]:

#save as a h5 file. save the keras model using code. when using the image and pass it through layers and print softmax.


# In[ ]:




# In[ ]:



