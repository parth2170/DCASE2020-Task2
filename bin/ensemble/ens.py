import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def get_machine_ids(machines, mode):
	mid_dict = {}
	if mode == 'd':
		path = '../../dev_data'
		folder = 'test'
		for m in machines:
			file_list = os.listdir(os.path.join(path, m, folder))
			id_list = list(set([int(file.split('_')[2]) for file in file_list]))
			mid_dict[m] = id_list
	elif mode == 'e':
		path = '../../eval_data'
		folder = 'test'
		for m in machines:
			file_list = os.listdir(os.path.join(path, m, folder))
			id_list = list(set([int(file.split('_')[1]) for file in file_list]))
			mid_dict[m] = id_list
	return mid_dict


def get_test_files(m, mid, mode):
	y = []
	if mode == 'd':
		path = '../../dev_data/'
		test_files = os.listdir(path + m + '/test')
		normal_test_files = ([file for file in test_files if (file[0] == 'n' and int(file.split('_')[2]) == mid and file[-4:] == '.wav')])
		anom_test_files = ([file for file in test_files if (file[0] == 'a' and int(file.split('_')[2]) == mid and file[-4:] == '.wav')])
		normal_test_files.sort()
		anom_test_files.sort()
		for sample in normal_test_files:	
			y.append(0)
		for sample in anom_test_files:
			y.append(1)
		return y
	if mode == 'e':
		path = path = '../../eval_data/'
		files = os.listdir(path + m + '/test/')
		files = [f for f in files if int(f.split('_')[1]) == mid]
		files.sort()
		return files

def main(mode):

	machines = ['ToyCar', 'ToyConveyor', 'fan', 'pump', 'slider', 'valve']

	if mode == 'd':

		mid_dict = get_machine_ids(machines, mode)
		results = {'Machine':[], 'Mid':[], 'AUC':[], 'pAUC':[]}

		anom_scores_ensemble_iv = pickle.load(open('individual_scores/dev/iVectors_gmm_dev_data.pickle', 'rb'))
		anom_scores_ensemble_gr = pickle.load(open('individual_scores/dev/modspec_graph_dev_data.pickle', 'rb'))

		for m in machines:

			avg  = {'AUC':[], 'pAUC':[]}

			for mid in mid_dict[m]:

				y_pred_iv = anom_scores_ensemble_iv[m][mid]['iv']
				y_pred_gr = anom_scores_ensemble_gr[m][mid]['gr']

				y_pred_iv = (y_pred_iv - np.min(y_pred_iv))/(np.max(y_pred_iv) - np.min(y_pred_iv))
				y_pred_gr = (y_pred_gr - np.min(y_pred_gr))/(np.max(y_pred_gr) - np.min(y_pred_gr))

				y_pred_ens = y_pred_iv * y_pred_gr

				y_test = get_test_files(m, mid, mode)

				AUC = roc_auc_score(y_test, y_pred_ens)
				pAUC = roc_auc_score(y_test, y_pred_ens, max_fpr = 0.1)
				
				results['Machine'].append(m)
				results['Mid'].append(mid)
				results['AUC'].append(AUC)
				results['pAUC'].append(pAUC)
				
				avg['AUC'].append(AUC)
				avg['pAUC'].append(pAUC)
			
			results['Machine'].append(m)
			results['Mid'].append('Average')
			results['AUC'].append(np.mean(avg['AUC']))
			results['pAUC'].append(np.mean(avg['pAUC']))

		results = pd.DataFrame(results)
		results.to_csv('ensemble_dev_data_results.csv')
		print(results)

	elif mode == 'e':

		mid_dict = get_machine_ids(machines, mode)
		anom_scores_ensemble_iv = pickle.load(open('individual_scores/eval/iVectors_gmm_eval_data.pickle', 'rb'))
		anom_scores_ensemble_gr = pickle.load(open('individual_scores/eval/modspec_graph_eval_data.pickle', 'rb'))

		for m in machines:

			for mid in mid_dict[m]:

				anom_scores = {'file':[], 'anomaly_score':[]}

				y_pred_iv = anom_scores_ensemble_iv[m][mid]['iv']
				y_pred_gr = anom_scores_ensemble_gr[m][mid]['gr']

				y_pred_iv = (y_pred_iv - np.min(y_pred_iv))/(np.max(y_pred_iv) - np.min(y_pred_iv))
				y_pred_gr = (y_pred_gr - np.min(y_pred_gr))/(np.max(y_pred_gr) - np.min(y_pred_gr))

				y_pred_ens = y_pred_iv * y_pred_gr

				eval_files = get_test_files(m, mid, mode)

				anom_scores['file'] = eval_files
				anom_scores['anomaly_score'] = y_pred_ens

				submission_file = pd.DataFrame(anom_scores)
				submission_file.to_csv('../../task2/Tiwari_IITKGP_task2_3/anomaly_score_{}_id_0{}.csv'.format(m, mid), header = False, index = False)

if __name__ == '__main__':
	n = len(sys.argv)
	if n < 2:
		print('Please enter dev/eval mode')
	mode = sys.argv[1]

	if mode == 'd' or mode == 'e':
		main(mode)
	else:
		print('Invalid mode')
