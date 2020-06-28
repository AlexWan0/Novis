import base64
import io
import cv2
from imageio import imread
from keras.models import load_model, Model
import numpy as np
import tensorflow as tf
import base64
import io
from matplotlib import pyplot as plt


interpreter = tf.lite.Interpreter(model_path='model_mseACTUALe91_det_multout.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

import os
import glob

latent_data = []

for class_folder in os.listdir('image_database/blood_smear'):
	print(class_folder)
	
	for img_fn in os.listdir('image_database/blood_smear/' + class_folder):
		img_fp = 'image_database/blood_smear/' + class_folder + '/' + img_fn
		
		print(img_fp)
		img = cv2.imread(img_fp)
		img = cv2.resize(img, (96, 96))
		
		if(img.shape[-1] == 4):
			img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

		img = img/255.0
		
		img = np.expand_dims(img, axis=0)

		interpreter.set_tensor(input_details[0]['index'], np.float32(img, axis=0))

		interpreter.invoke()

		mean = interpreter.get_tensor(output_details[0]['index'])
		logvar = interpreter.get_tensor(output_details[1]['index'])
		
		latent_data.append({'fp':img_fp, 'mean': mean, 'logvar': logvar, 'class': class_folder})

import pickle
with open('image_database/db_files/blood_smear_db.pkl', 'wb') as file_out:
    pickle.dump(latent_data, file_out)