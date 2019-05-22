import os
import numpy as np 
import pandas as pd
import seaborn as sns

# This following function is for weight analysis
def top_gene(weight_matrix, weight_file, encoded_matrix, encoded_file, gene_file, index, feature_distribution):
    
    en_file =os.path.join(encoded_file)
    encoded = pd.DataFrame(encoded_matrix)
    encoded.to_csv(en_file, sep='\t')

    wt_file = os.path.join(weight_file)
    wt_matrix = weight_matrix.set_index(index)  
    #use the following line when using pca, lda, svd etc.
    #wt_matrix = pd.DataFrame(weight_matrix, index=index)
    wt_matrix.to_csv(wt_file, sep='\t')
    
    #sum_node_activity = encoded.sum(axis=0).sort_values(ascending=False)
    #gene_sorted = wt_matrix.loc[:,int(sum_node_activity.index[0])].sort_values(ascending=False)
    
    gene_sorted = wt_matrix.sum(axis=1).sort_values(ascending=False)
    
    
    gn_file = os.path.join(gene_file)
    gene_matrix = pd.DataFrame(gene_sorted)
    gene_matrix.to_csv(gn_file, sep='\t')

    g = sns.FacetGrid(gene_matrix, sharex=False, sharey=False)
    #g.map(sns.distplot,sum_node_activity.index[0])
    g.map(sns.distplot,0)
    g.savefig(feature_distribution, dpi=300)
    
    print ('After Applying Dimension Reduction', encoded.shape)
    print('Weight Matrix Shape', wt_matrix.shape)
    #print('Sum of Encoded Node and Sorted',sum_node_activity)
