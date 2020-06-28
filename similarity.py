import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.metrics import log_loss, mutual_info_score

db_files = 'image_database/db_files/'

def distance(t1, t2):
	#print(t1.shape)
	#print(t2.shape)
	return cosine_similarity(t1, np.expand_dims(t2, axis=0))

def find_similar(db, t_mean, n_most=5):
	fp = db_files + db

	with open(fp, 'rb') as file_in:
		latent_data = pickle.load(file_in)

	for lt_obj in latent_data:
		prop_mean = lt_obj['mean']
		#print(prop_mean[0])
		dist = distance(prop_mean, t_mean)

		#print(dist)

		lt_obj['dist'] = dist

	latent_data_sorted = sorted(latent_data, key=lambda x: x['dist'])
	
	return latent_data_sorted[:n_most]

if __name__ == '__main__':
	print(find_similar('blood_smear_db.pkl', np.zeros((64,))))