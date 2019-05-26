
############################# IMPORT LIBRARY  #################################
seed=75
import numpy as np
from tensorflow import set_random_seed 
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interp
from itertools import cycle
from xgboost import XGBClassifier
from collections import Counter
from sklearn.metrics import average_precision_score, precision_recall_curve, matthews_corrcoef, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score, roc_auc_score, auc, cohen_kappa_score, precision_recall_curve, log_loss, roc_curve, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics.classification import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn import model_selection
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, MiniBatchSparsePCA, NMF, TruncatedSVD, FastICA, FactorAnalysis, LatentDirichletAllocation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import  Normalizer, MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, label_binarize, QuantileTransformer
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, RFE, RFECV
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE 
from imblearn.combine import SMOTEENN, SMOTETomek
from keras.initializers import RandomNormal
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from variational_autoencoder import *
#from aae_architechture import *
from aae_single_layer import *
from deep_autoencoder import *
from denoising_autoencoder import *
from shallow_autoencoder import *
from weight_analysis_pca import *
matplotlib.use('Agg')
np.random.seed(seed)


def zero_mix(x, n):
    temp = np.copy(x)
    noise=n
    if 'spilt' in noise:
        frac = float(noise.split('-')[1])
    for i in temp:
        n = np.random.choice(len(i), int(round(frac * len(i))), replace=False)
        i[n] = 0
    return (temp)

def gaussian_mix(x):
    n = np.random.normal(0, 0.1, (len(x), len(x[0])))
    return (x + n)

# The above two functions are used to add noise in the data
# And used to train denoising autoencoder

#########################   LOAD BREAST CANCER DATA ####################

file_1 = pd.read_csv('./data/subtype_molecular_rna_seq.csv')
data = file_1.iloc[0:20439,2:607].values  
X=data.T
       
file_2 = pd.read_csv('./data/subtype_molecular_rna_seq_label.csv', low_memory=False)
label= file_2.iloc[0,2:607].values   
y=label.T

print('Actual dataset shape {}'.format(Counter(y)))


##########################  LOAD UCEC  DATA   ###########################
'''
file_1 = pd.read_csv('./data/ucec_rna_seq.csv')
data = file_1.iloc[0:20482,2:232].values 
X=data.T

file_2 = pd.read_csv('./data/ucec_rna_seq_label.csv', low_memory=False)
label = file_2.iloc[0,2:232].values   #First row then column from dataset
y=label.T   

print('Actual dataset shape {}'.format(Counter(y)))
'''
              

sm = SMOTE(sampling_strategy='auto', kind='borderline1', random_state=seed)
X, y = sm.fit_sample(X, y)
#x_train, y_train = sm.fit_sample(x_train, y_train)
#x_test, y_test = sm.fit_sample(x_test, y_test)

print('Resampled dataset shape {}'.format(Counter(y)))
    

########################  FEATURE SCALING/NORMALIZATION ##################
qt = QuantileTransformer(n_quantiles=10, random_state=seed)
qt.fit(X)
X=qt.transform(X)
#x_train=qt.transform(x_train)
#x_test=qt.transform(x_test)



############## HEAT MAP INPUT DATA   #######################
'''
sns_plot=sns.heatmap(X, cmap="PiYG", cbar=True, xticklabels=False, yticklabels=False)
fig = sns_plot.get_figure()
fig.savefig("./figures/input_data.png", format='png', dpi=500)
'''
 

###############################################################################       
###############################DIMENSION REDUCTION ############################
###################   PCA, NMF, ICA, FA, LDA, SVD     #########################
#index = dataset.iloc[0:20482,0] # this is for valiadtion data
index = file_1.iloc[0:20439,0]


model = PCA(n_components=50, random_state=seed)
#model = NMF(n_components=50, init='random', random_state=seed)       
#model = FastICA(n_components=50, random_state=seed)
#model = BernoulliRBM(n_components=50, learning_rate=0.1, batch_size=10, n_iter=100, random_state=seed)
#model=FactorAnalysis(n_components=50, random_state=seed)
#model=LatentDirichletAllocation(n_components=50, random_state=seed)
#model=TruncatedSVD(n_components=50, random_state=seed)

model.fit(X)


X_transformed = model.transform(X)
encoded_pca = pd.DataFrame(X_transformed)

################ load model  #########################
weight_matrix=(model.components_).T
#print (model.components_.T)
#print (model.components_.T.shape)

################# analyze weight  ####################
top_gene(weight_matrix=weight_matrix, weight_file='./results/PCA/pca_weight_matrix.tsv',
         encoded_matrix=encoded_pca, encoded_file='./results/PCA/pca_encoded.tsv', 
         gene_file='./results/PCA/pca_sorted_gene.tsv', index=index,
         feature_distribution='./results/PCA/pca_weight_distribution.png')
