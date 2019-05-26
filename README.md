# Extracting Biologically Relevant Genes using Unsupervised Adversarial Autoencoder (AAE) from Cancer Transcriptomes
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![contribution](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/raktimmondol/latent-space-discovery/pulls)
![python version](https://img.shields.io/badge/python-2.7%20%7C%203.5%20-green.svg)
![keras version](https://img.shields.io/badge/keras-2.0.6-brightgreen.svg)
![tensorflow version](https://img.shields.io/badge/tensorflow-1.13.1-orange.svg)
![imblearn version](https://img.shields.io/badge/imbalanced--learn-0.4.3-blue.svg)

In this project, we introduce neural network based adversarial autoencoder (AAE) model to extract biologically-relevant features from RNA-Seq data. We also developed a method named TopGene to ﬁnd highly interactive genes from the latent space. AAE in combination with TopGene method ﬁnds important genes which could be useful for ﬁnding cancer biomarkers.


![project_logo_transparent](https://user-images.githubusercontent.com/28592095/56498063-8039da00-6543-11e9-8b4a-a551bad3ed0f.png)


## Getting Started

The following instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See the instruction below:

### Prerequisites

The following libraries are required to reproduce this project:

1) Keras (2.0.6)

2) Keras-adverserial (0.0.3)

3) Tensorflow (1.13.1)

4) Scikit-Learn (0.20.3)

5) Numpy (1.16.3)

6) Imbalanced-Learn (0.4.3)

Supports both Python 2.5.0 and Python 3.5.6

### Directory Layout
```bash
├── results
│   ├── saved_results
│   │   ├── Gene_Analysis_Breast_Cancer.xlsx
│   │   ├── Gene_Analysis_UCEC.xlsx
│   ├── AAE
│   │   ├── aae_encoded.tsv
│   │   ├── aae_sorted_gene.tsv
│   │   ├── aae_weight_distribution.png
│   │   ├── aae_weight_matrix
│   ├── PCA
│   ├── ... # add LDA, SVD etc
├── data
│   ├── data will be stored here
├── feature_extraction
│   ├── AAE
│   │   ├── aae_encoder.h5
│   │   ├── aae_decoder.h5
│   │   ├── aae_discriminator.h5
│   │   ├── aae_history.csv
│   ├── PCA
│   ├──VAE
│   ├── ...
├── README.md
├── figures
│   ├── saved_figures
│   │   ├── Olfactory__Transduction_pathway.png
└── .gitignore
```

### Usage


Run the following to extract features using different autoencoders

```
main.py
```

And run the following to extract features when PCA, NMF, FastICA, ICA, RBM etc. are used

```
main_pca.py
```


Gene ontology of molecular function was performed using DAVID 6.7 https://david-d.ncifcrf.gov/

More regarding gene ontology http://geneontology.org/docs/ontology-documentation/

## Proposed Architecture

![aae_latent_space_updated](https://user-images.githubusercontent.com/28592095/57984823-e8c49a00-7aa2-11e9-9379-b7a3fd47309e.png)

## Datasets

* [cBioPortal](https://www.cbioportal.org/) - Cancer Genomics Datasets
* [Breast Invasive Carcinoma (TCGA, Cell 2015)](https://www.cbioportal.org/study?id=brca_tcga_pub2015) - Clinical information is used to label various molecular subtypes

``` Breast Invasive Carcinoma (BRCA) ```

| Molecular Subtypes | Number of Patients | Label |
| ------------------ | ------------------ | ------------ |
| Luminal A | 304 | 0 |
| Luminal B | 121 | 1 |
| Basal & Triple Negetive | 137 | 2 |
| Her 2 Enriched | 43 | 3 |

| Total Number of Samples (Patients) | Total Number of Features (Genes) |
| :------------------: | :------------------: |
| 605 | 20439 |

* [Details about Molecular Subtypes of Breast Cancer](https://www.breastcancer.org/symptoms/types/molecular-subtypes)

### Validation Data

* [Uterine Corpus Endometrial Carcinoma (TCGA, Nature 2013)](http://www.cbioportal.org/study?id=ucec_tcga_pub) - Clinical information is used to label various molecular subtypes. 

``` Uterine Corpus Endometrial Carcinoma (UCEC) ```

| Molecular Subtypes | Number of Patients | Label |
| ------------- | ------------- | ----------- |
| Copy Number High | 60 | 0 |
| Copy Number Low | 90 | 1 | 
| Hyper Mutated (MSI) | 64 | 2 |
| Ultra Mutated (POLE) | 16 | 3 |

| Total Number of Samples (Patients) | Total Number of Features (Genes) |
| :------------------: | :------------------: |
| 230 | 20482 |

* [Details about Molecular Subtypes of Endometrial Cancer](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5154099/)

## Contributing

If you want to contribute to this project and make it better, your help is very welcome. When contributing to this repository please make a clean pull request.


## Acknowledgments

* The proposed architecture is inspired by https://github.com/bstriner/keras-adversarial


