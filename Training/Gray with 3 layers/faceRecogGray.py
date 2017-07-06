import theano
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from keras.layers.core import Dense
import Image
import os,sys
import numpy as np
import scipy.misc
import glob
from keras.models import model_from_json
import h5py
import cv2


st = "Untitled Folder"


def getImage(path):
	face_cascade = cv2.CascadeClassifier('/home/mohamed/OpenCV/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
	img = cv2.imread(path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if(len(faces)==0):
		return [] 
	(x,y,w,h) = faces[0]
    	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    	gray = gray[y:y+h, x:x+w]
    	#img = img[y:y+h, x:x+w]
	resize = cv2.resize(gray, (28, 28))
	data = np.array(resize, dtype=np.uint8)
	data = data.reshape(1, 28, 28, 1)
	data = data.astype('float32')
	data /= 255
	return data

def loadTrain():
	trainData=[]
	trainOutput = []
	imagesList =[]
	for i in range(1,25)+range(31,36)+range(46,48)+range(39,43)+range(49,51):
		imgs =[]
		path = os.path.join('/GP/Aberdeen',st+str(i), '*.jpg')
		files = glob.glob(path) # de btgeb kol el files el .jpg fe el folder
		for j in range (0,len(files)):
			img1 = getImage(files[j])
			if(len(img1) == 0):
				continue
			imgs.append(img1)
			for k in range (j+1 , len(files)):
				print(i ," ", j , " " , k)
				img2 = getImage(files[k])
				if(len(img2) == 0):
					continue
				if(i == 1 and j==0 and k==1):
					entry1 = np.append( img1 , img2 , axis = 1)
				elif (i == 1 and j == 0 and k==2):
					entry2 = np.append( img1 , img2 , axis = 1)
					trainData = np.append( entry1 , entry2 , axis = 0)
				else :
					entry = np.append( img1 , img2 , axis = 1)
					trainData = np.append(trainData , entry , axis=0)
				trainOutput.append([1.0,0.0])
		for m in imagesList:
			entry = np.append( m[0] , imgs[0] , axis = 1)
			trainOutput.append([0.0,1.0])
			entry1 = np.append( m[1] , imgs[1] , axis = 1)
			trainOutput.append([0.0,1.0])
			trainData = np.append(trainData , entry,axis=0)
			trainData = np.append(trainData , entry1,axis=0)
		imagesList.append(imgs)
	trainOutput = np.array(trainOutput, dtype=np.uint8)
	return trainData , trainOutput


def loadTest():
	testData=[]
	testOutput = []
	imagesList =[]
	for i in range(25,31)+range(36,39)+range(43,46)+[48]:
		imgs =[]
		path = os.path.join('/GP/Aberdeen',st+str(i), '*.jpg')
		files = glob.glob(path)
		for j in range (0,len(files)):
			img1 = getImage(files[j])
			if(len(img1) == 0):
				continue
			imgs.append(img1)
			for k in range (j+1 , len(files)):
				print(i ," ", j , " " , k)
				img2 = getImage(files[k])
				if(len(img2) == 0):
					continue
				if(i == 25 and j==0 and k==1):
					entry1 = np.append( img1 , img2 , axis = 1)
				elif (i == 25 and j == 0 and k==3):############################5let k=3 34an k=2 m7sal4 feha face detection
					entry2 = np.append( img1 , img2 , axis = 1)
					testData = np.append( entry1 , entry2 , axis = 0)
				else :
					entry = np.append( img1 , img2 , axis = 1)
					testData = np.append(testData , entry , axis=0)
				testOutput.append([1.0,0.0])
		for m in imagesList:
			entry = np.append( m[0] , imgs[0] , axis = 1)
			testOutput.append([0.0,1.0])
			entry1 = np.append( m[1] , imgs[1] , axis = 1)
			testOutput.append([0.0,1.0])
			testData = np.append(testData , entry,axis=0)
			testData = np.append(testData , entry1,axis=0)
		imagesList.append(imgs)
	testinOutput = np.array(testOutput, dtype=np.uint8)
	return testData , testOutput



trainData,trainOutput=loadTrain()
testData,testOutput = loadTest()




model = Sequential()
model.add(Convolution2D(20, 5, 5, border_mode="same",input_shape=(56, 28 , 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(20, 3, 3, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(20, 3, 3, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
'''model.add(Convolution2D(20, 3, 3, border_mode="same"))
model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(20, 3, 3, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))'''
model.add(Flatten())
#model.add(Dense(2000))
#model.add(Activation("relu"))
model.add(Dense(1000))
model.add(Activation("relu"))
model.add(Dense(500))
model.add(Activation("relu"))
model.add(Dense(2))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainData, trainOutput, nb_epoch=70, batch_size=100)
scores = model.evaluate(testData, testOutput)

##Only code needed to save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
###########################
'''
#Only code needed to  Load Code
json_file = open("model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
'''

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

'''
img1 = getImage("test.jpg")
img2 = getImage("test1.jpg")
entry = np.append( img1 , img2 , axis = 1)


predictions = model.predict(entry)
print predictions
'''

'''
data2 = testData.reshape(36*100,50,3)
scipy.misc.toimage(data2).save("output.jpg")
'''
