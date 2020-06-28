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
from similarity import find_similar
import time

def init_blood_model():
	interpreter = tf.lite.Interpreter(model_path='model_mseACTUALe91_det_multout.tflite')
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	global blood_model

	blood_model = (interpreter, input_details, output_details)

def plt_to_base64(plt):
	pic_IObytes = io.BytesIO()
	plt.savefig(pic_IObytes,  format='png')
	pic_IObytes.seek(0)
	pic_hash = base64.b64encode(pic_IObytes.read())
	return pic_hash

def vlb_num(img, img_ae, mu, logvar, kl_weight=1):
	kl = 0.5 * np.sum((np.exp(logvar) + np.square(mu) - 1.0 - logvar), axis=-1)
	mse = np.square(img - img_ae)
	mse = np.sum(mse, axis=(0, 1, 2))
	return kl, mse, kl_weight * kl + mse

def get_resid(im1, im2, thresh=(0.0, 1.0)):
	img = np.clip(np.abs(im1 - im2), 0.0, 1.0)
	img[img > thresh[1]] = 0
	img[img < thresh[0]] = 0
	img = img*255.0
	img = img.astype(np.uint16)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img

def blood_detect(img_data):
	interpreter, input_details, output_details = blood_model
	
	b64_string = img_data.decode()

	img = imread(io.BytesIO(base64.b64decode(b64_string)))

	img = cv2.resize(img, (96, 96))

	if(img.shape[-1] == 4):
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

	img = img/255.0

	img = np.expand_dims(img, axis=0)

	print(img.shape)

	interpreter.set_tensor(input_details[0]['index'], np.float32(img))

	interpreter.invoke()

	mean = interpreter.get_tensor(output_details[0]['index'])
	logvar = interpreter.get_tensor(output_details[1]['index'])
	ae = interpreter.get_tensor(output_details[2]['index'])

	fig, axs = plt.subplots(2, 2, tight_layout=True)
	axs[0,0].imshow(img[0])
	axs[0,1].imshow(ae[0])
	axs[1,0].imshow(get_resid(img[0], ae[0]))
	axs[1,1].imshow(get_resid(img[0], ae[0], thresh=(0.0, 0.5)))

	axs[0,0].title.set_text('Original')
	axs[0,1].title.set_text('Reconstructed')
	axs[1,0].title.set_text('Anomalies (thresh: 0.0-1.0)')
	axs[1,1].title.set_text('Anomalies (thresh: 0.0-0.5)')

	detection_diag = plt_to_base64(fig)
	vlb_res = vlb_num(img[0], ae[0], mean[0], logvar[0])

	most_similar = find_similar('blood_smear_db.pkl', mean[0])

	fig, axs = plt.subplots(1, 5, tight_layout=True, figsize=(12, 2))

	for i, lt_obj in enumerate(most_similar):
		#print(lt_obj['fp'])

		im_sim = cv2.imread(lt_obj['fp'])

		im_sim = cv2.cvtColor(im_sim, cv2.COLOR_BGR2RGB)

		axs[i].imshow(im_sim)
		axs[i].title.set_text(lt_obj['class'])

	sim_base64 = plt_to_base64(fig)

	return {'success': True, 'main_img': detection_diag.decode("utf-8"), 'sim_img': sim_base64.decode('utf-8'), 'kl':float(vlb_res[0]), 'recon':float(vlb_res[1]), 'vlb': float(vlb_res[2])}

if __name__ == '__main__':
	init_blood_model()
	with open('test.txt', 'r') as file_in:
		img_txt = str.encode(file_in.read())
	print(img_txt)
	print(blood_detect(img_txt))