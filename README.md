# MODULATION SPECTRAL SIGNAL REPRESENTATIONAND I-VECTORS FOR ANOMALOUS SOUND DETECTION

This repository can be used to reproduce our submissions for **DCASE Challenge 2020 Task 2** - <em>Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring</em>


#### Abstract
This report summarizes our submission for Task-2 of the DCASE 2020 Challenge. We propose two different anomalous sound detection systems, one based on features extracted from a modula- tion spectral signal representation and the other based on i-vectors extracted from mel-band features. The first system uses a nearest neighbour graph to construct clusters which capture local variations in the training data. Anomalies are then identified based on their distance from the cluster centroids. The second system uses i-vectors extracted from mel-band spectra for training a Gaussian Mixture Model. Anomalies are then identified using their negative log likelihood. Both these methods show significant improvement over the DCASE Challenge baseline AUC scores, with an average improvement of 6% across all machines. An ensemble of the two systems is shown to further improve the average performance by 11% over the baseline.


**Requirements**
- `librosa`
- [`noisereduce`](https://pypi.org/project/noisereduce/)
- [`SRMRpy`](https://github.com/jfsantos/SRMRpy)
- `networkx == 2.2`
- `scikit-learn`
- `numpy`
- `pandas`
- `tqdm`

## Usage

#### 1. Clone this repository
#### 2. Download datasets
- Datasets are available [here](https://zenodo.org/record/3678171)
- Datasets for all machines can be downloaded and unzipped by running
    - `sh download_dev_data.sh` for development data
    - `sh download_eval_data.sh` for evaluation data

#### 3. Running System 1
- `cd bin/modspec_graph/`
- `python graph_anom_detection.py d` - for running on development data
    - Modulation Spectrums for each machine-id will be stored in `npy` files in `saved/` in the same directory
    - The results for development data are stored in `modspec_graph_dev_data_results.csv` in the same directory
- `python graph_anom_detection.py e` - for running on evaluation data
    - The results for evaluation data are stored in the submission format in the directory `task2`

#### 3. Running System 2
- i-Vectors for both development and evaluation have been provided in the zip file -  `saved_iVectors/ivector_mfcc_100.zip`
- Unzip `ivector_mfcc_100.zip` in the same directory
    - Code for extracting i-Vectors will be added soon
- `cd bin/iVectors_gmm/`
- `python gmm.py d` - for running on development data
    - The results for development data are stored in `iVectors_gmm_dev_data_results.csv` in the same directory
- `python gmm.py e` - for running on evaluation data
    - The results for evaluation data are stored in the submission format in the directory `task2`

#### 3. Running ensemble of System 1 and System 2
- Run both System 1 and System 2
- `cd bin/ensemble/`
- `python ens.py d` - for running on development data
    - The results for development data are stored in `ensemble_dev_data_results.csv` in the same directory
- `python ens.py e` - for running on evaluation data
    - The results for evaluation data are stored in the submission format in the directory `task2`
    
    
## Results

| Machine     | Mid | Baseline  AUC | Modspec  Graph AUC | iVGmm  AUC | Ensemble  AUC | Baseline  pAUC | Modspec  Graph pAUC | iVGmm  pAUC | Ensemble  pAUC |
|-------------|-----|---------------|--------------------|------------|---------------|----------------|---------------------|-------------|----------------|
| ToyCar      | 1   | 81.36%        | 78.24%             | 75.04%     | 81.64%        | 68.40%         | 64.69%              | 57.54%      | 66.75%         |
|             | 2   | 85.97%        | 89.06%             | 83.30%     | 91.72%        | 77.72%         | 76.14%              | 67.00%      | 79.78%         |
|             | 3   | 63.30%        | 67.16%             | 79.47%     | 78.21%        | 55.21%         | 52.58%              | 59.52%      | 56.37%         |
|             | 4   | 84.45%        | 89.40%             | 94.84%     | 96.44%        | 68.97%         | 63.54%              | 82.94%      | 84.80%         |
|             | Avg | 78.77%        | 80.96%             | 83.16%     | 87.00%        | 67.58%         | 64.24%              | 66.75%      | 71.92%         |
| ToyConveyor | 1   | 78.07%        | 62.56%             | 55.51%     | 64.62%        | 64.25%         | 51.59%              | 52.82%      | 52.24%         |
|             | 2   | 64.16%        | 54.03%             | 53.80%     | 56.65%        | 56.01%         | 49.99%              | 50.95%      | 50.23%         |
|             | 3   | 75.35%        | 59.10%             | 59.09%     | 64.06%        | 61.03%         | 50.31%              | 52.82%      | 52.25%         |
|             | Avg | 72.53%        | 58.57%             | 56.13%     | 61.78%        | 60.43%         | 50.63%              | 52.20%      | 51.58%         |
| fan         | 0   | 54.41%        | 63.37%             | 67.85%     | 67.12%        | 49.37%         | 49.73%              | 57.38%      | 52.92%         |
|             | 2   | 73.40%        | 79.32%             | 70.39%     | 80.48%        | 54.81%         | 57.16%              | 61.93%      | 59.21%         |
|             | 4   | 61.61%        | 71.76%             | 73.52%     | 78.07%        | 53.26%         | 50.68%              | 57.53%      | 53.99%         |
|             | 6   | 73.92%        | 74.00%             | 81.15%     | 81.90%        | 52.35%         | 49.38%              | 56.31%      | 49.23%         |
|             | Avg | 65.83%        | 72.11%             | 73.23%     | 76.89%        | 52.45%         | 51.74%              | 58.29%      | 53.84%         |
| pump        | 0   | 67.15%        | 86.66%             | 74.99%     | 86.95%        | 56.74%         | 82.52%              | 67.10%      | 78.32%         |
|             | 2   | 61.53%        | 62.44%             | 74.91%     | 70.06%        | 58.10%         | 64.77%              | 60.08%      | 65.72%         |
|             | 4   | 88.33%        | 84.16%             | 92.02%     | 90.73%        | 67.10%         | 59.95%              | 73.74%      | 68.00%         |
|             | 6   | 74.55%        | 81.64%             | 71.10%     | 82.65%        | 58.02%         | 66.20%              | 51.70%      | 66.56%         |
|             | Avg | 72.89%        | 78.72%             | 78.26%     | 82.60%        | 59.99%         | 68.36%              | 63.15%      | 69.65%         |
| slider      | 0   | 96.19%        | 99.91%             | 83.92%     | 98.72%        | 81.44%         | 99.53%              | 50.04%      | 93.44%         |
|             | 2   | 78.97%        | 84.36%             | 56.93%     | 77.93%        | 63.68%         | 73.86%              | 47.84%      | 52.89%         |
|             | 4   | 94.30%        | 97.83%             | 87.84%     | 95.93%        | 71.98%         | 88.59%              | 62.71%      | 79.69%         |
|             | 6   | 69.59%        | 79.03%             | 59.04%     | 71.40%        | 49.02%         | 55.47%              | 49.91%      | 52.28%         |
|             | Avg | 84.76%        | 90.28%             | 71.93%     | 86.00%        | 66.53%         | 79.36%              | 52.63%      | 69.57%         |
| valve       | 0   | 68.76%        | 100.00%            | 79.33%     | 98.80%        | 51.70%         | 100.00%             | 52.94%      | 95.62%         |
|             | 2   | 68.18%        | 99.88%             | 85.35%     | 98.69%        | 51.83%         | 99.34%              | 56.27%      | 93.29%         |
|             | 4   | 74.30%        | 98.26%             | 84.10%     | 95.88%        | 51.97%         | 91.32%              | 56.32%      | 80.26%         |
|             | 6   | 53.90%        | 89.22%             | 69.84%     | 85.01%        | 48.43%         | 72.59%              | 49.91%      | 59.65%         |
|             | Avg | 66.28%        | 96.84%             | 79.65%     | 94.59%        | 50.98%         | 90.81%              | 53.86%      | 82.21%         |
