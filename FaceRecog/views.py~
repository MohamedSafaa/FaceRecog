# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
from django.http import HttpResponse
#import json
from serializers import ResultSerializer
import Image
import numpy as np
from keras.models import model_from_json
import scipy.misc

'''from django.utils import simplejson
from django.http import JsonResponse
'''



@api_view(['GET', 'POST'])
def index(request):
        

	'''#img1 = request.data["img1"]
	#img2 = request.data["img2"]
	return Response(json.dumps(request.data))
	'''

	im1 = request.GET['img1']
	im2 = request.GET['img2']
	height = int(request.GET["height"])
	width = int(request.GET["width"])
	print("-----------------------------------------",im1)

	d = {'res':run(im1,im2,height,width)}
	serializer = ResultSerializer(data=d)
        if serializer.is_valid():
            #serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
	



def run(im1,im2,height,width):
	
	img1 = editImage(im1,height,width)
	img2 = editImage(im2,height,width)
	json_file = open("model.json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")

	entry = np.append( img1 , img2 , axis = 1)

	predictions = loaded_model.predict(entry)
	
	if(predictions[0][0]>predictions[0][1]):
		return ("Yes")
	else :
		return ("No")

def editImage(im ,height , width):
	st = im.split(",")
	lint = [int(s)for s in st]
	print(lint)
	t = []
	l = []
	i = 0
	for r in lint:
		if(i != width):
			i = i+1 
		else :
			t.append(l)
			l = []
			i = 0
		l.append(r)
	
	t.append(l)
	'''with file('pic.jpg', 'wb') as f:
        	for i in range(0,width):
			for j in range(0,height):
				print(t[i][j])
				f.write(str(t[i][j]))
			f.write("\n")
	f.close()
'''

	data = np.array(t)
	data = data.reshape(height,width,1)
	
	scipy.misc.toimage(data).save("pic.jpg")
	#scipy.misc.imsave(data, "pic.jpg")
	image = Image.open("pic.jpg")
	
	resize = image.resize((28,28), Image.NEAREST)
	resize.load()
	data = data.reshape(1, 28, 28, 1)
	data = data.astype('float32')
	data /= 255
	return data



	
# Create your views here.

