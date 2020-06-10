import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from utils import *
from sklearn.metrics import roc_auc_score



def build_net(X_train):
    L = len(X_train)
    dist_mat = np.ones((L, L))
    for i in range(L):
        for j in range(i + 1):
            dist = np.sum(np.abs(X_train[i] - X_train[j]))
            dist_mat[i][j] = dist
            dist_mat[j][i] = dist
        dist_mat[i][i] = 999
    edge_dict = {e:[] for e in range(L)}
    for e in range(L):
        distances = list(dist_mat[e])
        min1 = np.argmin(distances)
        edge_dict[e].append(min1)
    G = nx.Graph(edge_dict)
    sub_graphs = list(nx.connected_component_subgraphs(G))
    # print('# of sub graphs = {}'.format(len(sub_graphs)))
    sub_graph_dict = {k:list(sub_graphs[k].nodes) for k in range(len(sub_graphs))}
    return sub_graph_dict, G

def get_means(X_train, subgraphs):
	means = {k:0 for k in subgraphs}
	deviations = {k:0 for k in subgraphs}
	for G in subgraphs:
		means[G] = np.mean([X_train[i] for i in subgraphs[G]], axis = 0)
		deviations[G] = np.std([X_train[i] for i in subgraphs[G]], axis = 0) + 0.000001
	return means, deviations

def get_anom_score(X_test, means, deviations):
	y_pred = []
	for sample in X_test:
		pred = np.min([np.sum((np.abs(sample - means[i])/deviations[i])) for i in means])
		y_pred.append(pred)
	return np.array(y_pred)


def main(mode):

	machines = ['ToyCar', 'ToyConveyor', 'fan', 'pump', 'slider', 'valve']

	reducenoise = 100

	if mode == 'd':

		mid_dict = get_machine_ids(machines, mode)
		anom_scores_ensemble = {}
		results = {'Machine':[], 'Mid':[], 'AUC':[], 'pAUC':[]}

		for m in machines:

			anom_scores_ensemble[m] = {}
			avg  = {'AUC':[], 'pAUC':[]}

			for mid in mid_dict[m]:

				anom_scores_ensemble[m][mid] = {}

				X_train, X_test, y_test = get_spectrums(m, mid, reducenoise, mode)

				subgraphs, Graph = build_net(X_train)
				means, deviations = get_means(X_train, subgraphs)
				y_pred_gr = get_anom_score(X_test, means, deviations)

				AUC = roc_auc_score(y_test, y_pred_gr)
				pAUC = roc_auc_score(y_test, y_pred_gr, max_fpr = 0.1)
				anom_scores_ensemble[m][mid]['gr'] = y_pred_gr
				
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
		results.to_csv('modspec_graph_dev_data_results.csv')
		print(results)
		with open('../ensemble/individual_scores/dev/modspec_graph_dev_data.pickle', 'wb') as file:
			pickle.dump(anom_scores_ensemble, file)

	elif mode == 'e':

		mid_dict = get_machine_ids(machines, mode)
		anom_scores_ensemble = {}

		for m in machines:

			anom_scores_ensemble[m] = {}

			for mid in mid_dict[m]:

				anom_scores_ensemble[m][mid] = {}
				anom_scores = {'file':[], 'anomaly_score':[]}

				X_train, X_test, eval_files = get_spectrums(m, mid, reducenoise, mode)
				
				subgraphs, Graph = build_net(X_train)
				means, deviations = get_means(X_train, subgraphs)
				y_pred_gr = get_anom_score(X_test, means, deviations)

				anom_scores['file'] = eval_files
				anom_scores['anomaly_score'] = y_pred_gr
				anom_scores_ensemble[m][mid]['iv'] = y_pred_gr

				submission_file = pd.DataFrame(anom_scores)
				submission_file.to_csv('../../task2/Tiwari_IITKGP_task2_2/anomaly_score_{}_id_{}.csv'.format(m, mid), header = False, index = False)

		with open('../ensemble/individual_scores/eval/modspec_graph_eval_data.pickle', 'wb') as file:
			pickle.dump(anom_scores_ensemble, file)

if __name__ == '__main__':
	n = len(sys.argv)
	if n < 2:
		print('Please enter dev/eval mode')
	mode = sys.argv[1]

	if mode == 'd' or mode == 'e':
		main(mode)
	else:
		print('Invalid mode')