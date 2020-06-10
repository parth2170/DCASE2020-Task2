import os
import pickle
import librosa
import numpy as np 
import pandas as pd
import noisereduce as nr
from srmrpy import srmr

def reshape_(x):
	return np.array([i.flatten() for i in x])

def get_noise(m, mid, reducenoise, mode):
	if mode == 'd':
		path = '../../dev_data/{}/'
	elif mode == 'e':
		path = '../../eval_data/{}/'
	normal_train_files = os.listdir(path + m +'/train')
	normal_train_files = [file for file in normal_train_files if int(file.split('_')[2]) == mid and file[-4:] == '.wav']
	noise = []
	for sample in normal_train_files[:reducenoise]:
		y, fs = librosa.load('../../dev_data/'+ m +'/train/'+ sample, sr = 16000)
		noise.append(y)
	noise = np.mean(noise, axis = 0)
	return noise
 
def read_spectrum(path, noise, reducenoise):
	y, fs = librosa.load(path, sr = 16000)
	if reducenoise:
		y = nr.reduce_noise(audio_clip=y, noise_clip=noise)
	modspec = srmr(y, fs, n_cochlear_filters = 60, norm = False, low_freq=125, min_cf=4, max_cf=128)[1]
	modspec = np.mean(modspec, axis = 2)
	return modspec

def get_train(m, mid, noise, reducenoise, mode):
	X = []
	if mode == 'd':
		path = '../../dev_data/{}/'
	elif mode == 'e':
		path = '../../eval_data/{}/'
	normal_train_files = os.listdir(path + m + '/train')
	normal_train_files = [file for file in normal_train_files if (int(file.split('_')[2]) == mid and file[-4:] == '.wav')]
	normal_train_files.sort()
	for sample in normal_train_files:
		X.append(np.array(read_spectrum(path + m + '/train/' + sample, noise, reducenoise)))
	return X

def get_test(m, mid, noise, reducenoise, mode):
	X, y = [], []
	if mode == 'd':
		path = '../../dev_data/{}/'
		test_files = os.listdir(path + m + '/test')
		normal_test_files = ([file for file in test_files if (file[0] == 'n' and int(file.split('_')[2]) == mid and file[-4:] == '.wav')])
		anom_test_files = ([file for file in test_files if (file[0] == 'a' and int(file.split('_')[2]) == mid and file[-4:] == '.wav')])
		normal_test_files.sort()
		anom_test_files.sort()
		for sample in normal_test_files:	
			X.append(np.array(read_spectrum(path + m + '/test/' + sample, noise, reducenoise)))
			y.append(0)
		for sample in anom_test_files:
			X.append(np.array(read_spectrum(path + m + '/test/' + sample, noise, reducenoise)))
			y.append(1)
		return X, y
	elif mode == 'e':
		path = '../../eval_data/{}/'
		test_files = os.listdir(path + m + '/test')
		test_files = ([file for file in test_files if (int(file.split('_')[1]) == mid and file[-4:] == '.wav')])
		test_files.sort()
		for sample in normal_test_files:	
			X_test.append(np.array(read_spectrum(path + m + '/test/' + sample, noise, reducenoise)))
		return X, test_files

def get_spectrums(machine, mid, reducenoise, mode):
	file_name = '_NR-{}_{}_.npy'.format(reducenoise, mode)
	if mode == 'd':
		try:
			X_train = np.load('saved/'+machine+str(mid)+'_X_train'+file_name)
			X_test = np.load('saved/'+machine+str(mid)+'_X_test'+file_name)
			y_test = np.load('saved/'+machine+str(mid)+'_y_test'+file_name)
		except:
			print('Computing Modulation Spectrums', machine, mid)
			noise = get_noise(machine, mid, reducenoise, mode)
			X_train = get_train(machine, mid, noise, reducenoise, mode)
			X_test, y_test = get_test(machine, mid, noise, reducenoise, mode)
			np.save('saved/'+machine+str(mid)+'_X_train'+file_name, X_train)
			np.save('saved/'+machine+str(mid)+'_X_test'+file_name, X_test)
			np.save('saved/'+machine+str(mid)+'_y_test'+file_name, y_test)
		return X_train, X_test, y_test
	elif mode == 'e':
		try:
			X_train = np.load('saved/'+machine+str(mid)+'_X_train'+file_name)
			X_test = np.load('saved/'+machine+str(mid)+'_X_test'+file_name)
			eval_files = np.load('saved/'+machine+str(mid)+'_eval_files'+file_name)
		except:
			print('Computing Modulation Spectrums', machine, mid)
			noise = get_noise(machine, mid, reducenoise, mode)
			X_train = get_train(machine, mid, noise, reducenoise, mode)
			X_test, eval_files = get_test(machine, mid, noise, reducenoise, mode)
			np.save('saved/'+machine+str(mid)+'_X_train'+file_name, X_train)
			np.save('saved/'+machine+str(mid)+'_X_test'+file_name, X_test)
			np.save('saved/'+machine+str(mid)+'_eval_files'+file_name, y_test)
		return X_train, X_test, eval_files

def get_machine_ids(machines, mode):
	mid_dict = {}
	if mode == 'd':
		path = '../../saved_iVectors/ivector_mfcc_100'
		folder = 'test'
		for m in machines:
			file_list = os.listdir(os.path.join(path, m, folder))
			id_list = list(set([int(file.split('_')[2]) for file in file_list]))
			mid_dict[m] = id_list
	elif mode == 'e':
		path = '../../saved_iVectors/ivector_mfcc_100'
		folder = 'test_eval'
		for m in machines:
			file_list = os.listdir(os.path.join(path, m, folder))
			id_list = list(set([int(file.split('_')[1]) for file in file_list]))
			mid_dict[m] = id_list
	return mid_dict
