<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">One for All, All for One</h1>

<!-- ABOUT THE PROJECT -->
## About The Project
Multi-target Cross-domain recommendation with implicit feedback data in each domain. 
Paper under submission to CIKM-22, "One for All, All for One: Learning and Transferring User Embeddings for Cross-Domain Recommendation".  

## Code Structure
```bash
One-For-All   
├── README.md                                 Read me file
├── bert                                      Bert modules and Transformer layers
├── data                                      Demo data  
├── dataset                                   Data set and processing methods
│  ├── check_sparsity.ipynb                   Calculate data set distribution for the all domains.   
│  ├── multi-domain_amazon.ipynb              Demo on how to process the amazon multiple data sets for MTCDR   
├── utils                                     Utilities, e.g. early stop.
│  ├── __init__.py                            Module init 
│  ├── data.py                                All data loaders 
│  ├── eval_metrics.py                        Evaluation metrics 
│  ├── loss_func.py                           Loss functions 
│  ├── pytorchtools.py                        Early stop implementation
│  ├── result_plot.py                         Line plot function 
│  ├── Save_embedding.py                      Functions for saving item/user embedding, preference scores, etc. 
│  ├── scheduler.py                           Learning rate scheduler
│  ├── tools.py                               Others (not in using)
│── Data_loader.py                            Entry for data split and data loader creators      
├── loss_functions.py                         Loss functions
├── main_cross.py                             CAT-ART model, without pre-training of the CAT module 
├── main_cross_pre.py                         CAT-ART model, with pre-training of the CAT module
├── main_cross_scores.py                      Get the averaged attention-scores of the ART module on the test set.
├── main_single_mf.py                         Single domain MF model for recommendation
├── models.py                                 Models 
├── run_functions.py                          Run functions, e.g. train step, test step
├── show_result.py                            Print out result from pickle file
├── run.sh                                    General run entry on venus 
├── run_cross.sh                              My run demos with parameters 
├── run_cross_auto.sh                         My run demos with parameters
├── run_cross_pre.sh
└── .gitignore                                gitignore file
```

### Packages
```shell
pip install torch
pip install sklearn
```

<!-- Usage -->
## Usage
1. Dataset and process
   
   1.1 Original Full Data Processing Stream 

   ```shell
    # Train data at '/data/ceph/seqrec/UMMD/data/hdfs/q36_age_train_08' (17G);
    cd bert/dataset
    # for each file in the train data folder run the following:
    python data_txt2pickle.py --filename "demo_data.gz" --domain 0
    # Results were saved at: /data/ceph/seqrec/UMMD/data/pickle/q36_age_train_org
    
    cd ../../   # Return to the main folder 
    ipython notebook
    run the train_test_split.ipynb  # for train and test splitting (only for missing 0 data samples)
    # Results were saved to:/data/ceph/seqrec/UMMD/data/pickle/q36_age_train_rec2  
   ```
   Here we provide a single processed file for both train and test as demo data at ./data folder. 

   We are still discussing whether and how to open-source the original dataset.

   1.2 Data Structure of used pickle file
   
   Each pickle file stores a dict data with keys: 'uid', 'age', 'gender', and 'feature', in which the values of keys 'uid', 'age', 'gender' are list data. 
   The value for the 'feature' key is a list of length 5, where each element represent the 'features' (interacted items) in each of the 5 domains. 
   For example:
   ```
   'uid': [1, 2, 3],
   'age': [11, 21, 31],
   'gender': [1, 2, 1], # 1-male, 2-female
   'feature': [
               [[a, b], [1, 2, 1], [i,j, k,l]],  # user 'feawture' in domain 0 (for the three users)
               [[b, c], [1, 2, 4], [i,k,l]],  # user 'feawture' in domain 1 (for the three users)
               [[a, d], [1, 2, 10], [k,l]],  # user 'feawture' in domain 2 (for the three users)
               [[d, b], [1,2, 6], [1, l]],  # user 'feawture' in domain 3 (for the three users)
               [[g, b], [1, 2, 5, 6], [i,h, p]],  # user 'feawture' in domain 4 (for the three users)
               ]           
   ```
  

2. Single-domain recommendation
   
    *configure your dataset path in function train_test_split() at Data_loader.py* 
    ```shell
    # Single domain recommendation with BPR based MF
    python main_single_mf.py --batch_size 2048 --num_run 0 --domain 0 --epoch 20 --bar_dis True --result_dir 'your path' --train True
    ```

3. Multi-target cross-domain recommendation
    ```shell
    # CAT-ART
    python main_cross.py --result_dir 'your path'
    python main_cross_pre.py --result_dir 'your path'
    ```
    For baseline HeroGRAPH, please refer to their open source at: https://github.com/cuiqiang1990/HeroGRAPH

<!-- Contact -->
## Contact
Chenglin Li: ch11 @ ualberta dot ca

