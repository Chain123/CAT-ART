# -*- coding: utf-8 -*-

from utils import SingleEmbedRec, MulCrossRec, SingleEmbedRec_mpf, SingleEmbedRec_cmf
from utils import SingleEmbedRec_mpf2, AmazonCrossRec, AmazonColdRec
import numpy as np 
import os
import time
import _pickle as pickle
import pandas as pd


####################################
#       business datasets
# if want to add missing domain for cold start replace file suffix 0.pickle with x.pickle x=1,2,3,4
####################################
def train_test_split():
    # demo path
    data_dir = "./data"
    file_ids = [65]
    # IDs to filenames
    train_files = []
    test_files = []
    for f_id in file_ids: 
        val_tmp = f_id + 100000
        train_files.append(os.path.join(data_dir, f"part-{str(val_tmp)[1:]}.gz_{0}_train.pickle"))
        test_files.append(os.path.join(data_dir, f"part-{str(val_tmp)[1:]}.gz_{0}_test.pickle"))
    return train_files, test_files


def read_single_file(single_args):
    seed = 1
    filename, target_domain = single_args[0], int(single_args[1])
    if target_domain < 0:
        train_data = {"feature": [[], [], [], [], []], 'age': [], 'gender': [], 'uid': []}
        valid_data = {"feature": [[], [], [], [], []], 'age': [], 'gender': [], 'uid': []}
    else:
        train_data = {"feature": [], 'age': [], 'gender': [], 'uid': []}
        valid_data = {"feature": [], 'age': [], 'gender': [], 'uid': []}
    try:
        data = pickle.load(open(filename, 'rb'))
        for u in range(len(data['uid'])):
            if target_domain < 0:
                for i in range(5):
                    user_set = data['feature'][i][u]
                    # if i == 3:
                    #     if len(user_set) > 100:
                    #         user_set = user_set[0:100]
                    np.random.seed(seed)
                    np.random.shuffle(user_set)
                    # split
                    train_length = int(len(user_set) * 0.8)
                    train_data['feature'][i].append(user_set[0:train_length])
                    valid_data['feature'][i].append(user_set[train_length:])
                    # if len(user_set[0:train_length]) < 1:
                    #     print(f"Empty train: {single_args} in domain {i} user {data['uid'][u]}")
                    #     sys.exit()
                    # if len(user_set[train_length:]) < 1:
                    #     print(f"Empty valid: {single_args} in domain {i} user {data['uid'][u]}")
                    #     sys.exit()
            else:
                user_set = data['feature'][target_domain][u]
                # if target_domain == 3:
                #     if len(user_set) > 100:
                #         user_set = user_set[0:100]
                np.random.seed(seed)
                np.random.shuffle(user_set)
                # split
                train_length = int(len(user_set) * 0.8)
                train_data['feature'].append(user_set[0:train_length])
                valid_data['feature'].append(user_set[train_length:])

            valid_data['age'].append(data['age'][u])
            valid_data['gender'].append(data['gender'][u])
            valid_data['uid'].append(data['uid'][u])
            train_data['age'].append(data['age'][u])
            train_data['gender'].append(data['gender'][u])
            train_data['uid'].append(data['uid'][u])

    except OSError as e:
        print(filename)
        print(e)
    return train_data, valid_data


def ReadDataParallel(files, target_domain, num_process=8):
    from multiprocessing import Pool

    target_domains = list(np.ones(len(files)) * target_domain)
    # print(target_domains)
    args = list(zip(files, target_domains))
    # print(args[0])
    # sys.exit()
    # start processing
    start = time.time()
    pool = Pool(num_process)
    df_collection = pool.map(read_single_file, args)
    pool.close()
    pool.join()
    print("Finishing loading:", time.time() - start)
    # collect all data
    if target_domain < 0:
        train_data = {"feature": [[], [], [], [], []], 'age': [], 'gender': [], 'uid': []}
        valid_data = {"feature": [[], [], [], [], []], 'age': [], 'gender': [], 'uid': []}
    else:
        train_data = {"feature": [], 'age': [], 'gender': [], 'uid': []}
        valid_data = {"feature": [], 'age': [], 'gender': [], 'uid': []}

    for val in df_collection:
        for key in train_data.keys():
            if target_domain < 0:
                if key == "feature":
                    for i in range(5):
                        train_data[key][i].extend(val[0][key][i])
                        valid_data[key][i].extend(val[1][key][i])
                else:
                    train_data[key].extend(val[0][key])
                    valid_data[key].extend(val[1][key])
            else:
                train_data[key].extend(val[0][key])
                valid_data[key].extend(val[1][key])
    return train_data, valid_data


def read_single_file_cmf(single_args):
    seed = 1
    filename = single_args
    train_data = {"feature": [], 'uid': []}
    valid_data = {"feature": [], 'uid': []}
    try:
        data = pickle.load(open(filename, 'rb'))
        for u in range(len(data['uid'])):
            user_domain_train = []
            user_domain_valid = []
            for domain in range(5):
                user_set = data['feature'][domain][u]
                # off-set
                if domain > 1:
                    off_set = 200000 + (domain - 2) * 50000
                else:
                    off_set = domain * 100000
                user_set = [val + off_set for val in user_set]
                np.random.seed(seed)
                np.random.shuffle(user_set)
                # split
                train_length = int(len(user_set) * 0.8)
                user_domain_train.extend(user_set[0:train_length])
                user_domain_valid.extend(user_set[train_length:])
            train_data['feature'].append(user_domain_train)
            valid_data['feature'].append(user_domain_valid)
            valid_data['uid'].append(data['uid'][u])
            train_data['uid'].append(data['uid'][u])
    except OSError as e:
        print(filename)
        print(e)
    return train_data, valid_data


def read_single_file_cmf_test(single_args):
    filename, target_domain = single_args[0], int(single_args[1])
    train_data = {"feature": [], 'uid': []}
    try:
        data = pickle.load(open(filename, 'rb'))
        for u in range(len(data['uid'])):
            user_set = data['feature'][target_domain][u]
            # off-set
            if target_domain > 1:
                off_set = 200000 + (target_domain - 2) * 50000
            else:
                off_set = target_domain * 100000
            user_set = [val + off_set for val in user_set]
            train_data['feature'].append(user_set)
            train_data['uid'].append(data['uid'][u])
    except OSError as e:
        print(filename)
        print(e)
    return train_data


def ReadDataParallel_cmf(files, num_process=8):
    from multiprocessing import Pool
    start = time.time()
    pool = Pool(num_process)
    df_collection = pool.map(read_single_file_cmf, files)
    pool.close()
    pool.join()
    print("Finishing loading:", time.time() - start)
    # collect all data
    train_data = {"feature": [], 'uid': []}
    valid_data = {"feature": [], 'uid': []}

    for val in df_collection:
        for key in train_data.keys():
            train_data[key].extend(val[0][key])
            valid_data[key].extend(val[1][key])
    return train_data, valid_data


def ReadDataParallelTest(files, target_domain):
    # collect all data
    test_data = {"feature": [], 'age': [], 'gender': [], 'uid': []}
    for filename in files:
        data = pickle.load(open(filename, 'rb'))
        for key in test_data.keys():
            if key == "feature":
                test_data[key].extend(data[key][target_domain])
            else:
                test_data[key].extend(data[key])
    return test_data


def ReadDataParallelTest_cmf(files, target_domain, num_process):
    from multiprocessing import Pool
    target_domains = list(np.ones(len(files)) * target_domain)
    args = list(zip(files, target_domains))

    start = time.time()
    pool = Pool(num_process)
    df_collection = pool.map(read_single_file_cmf_test, args)
    pool.close()
    pool.join()
    print("Finishing loading:", time.time() - start)
    # collect all data
    test_data = {"feature": [], 'uid': []}
    for val in df_collection:
        for key in test_data.keys():
            test_data[key].extend(val[key])
    return test_data


# single domain data loader
def single_domain_loader(domain, train=True, mask="False"):
    train_files, test_files = train_test_split()
    train_loader, valid_loader = None, None
    # train data loader
    print(f" *** Loading training data from {len(train_files)} files ... ")
    train_data, valid_data = ReadDataParallel(train_files, domain, num_process=16)
    if train:
        train_loader = SingleEmbedRec(train_data, domain, mask=mask)
        valid_loader = SingleEmbedRec(valid_data, domain, mask=mask)
    # test data loader
    print(f" *** Loading testing data from {len(test_files)} files ... ")
    # TODO change n_neg for recommender
    test_data = ReadDataParallelTest(test_files, domain)
    # test_loader = SingleEmbedRecTest(train_data, valid_data, test_data, domain)
    test_loader = single_domain_test_data(train_data, valid_data, test_data)
    return train_loader, valid_loader, test_loader


def single_domain_test_data(train_data, valid_data, test_data):
    ground_truth = test_data['feature']
    user = test_data['uid']
    train_pos_dict = dict(zip(train_data['uid'], train_data['feature']))
    for ind, u in enumerate(valid_data['uid']):
        train_pos_dict[u].extend(valid_data['feature'][ind])
    train_pos = []
    for u in user:
        train_pos.append(train_pos_dict[u])
    return {"user": user, "truth": ground_truth, 'train_pos': train_pos}


def single_domain_loader_mpf(domain, train=True):
    train_files, test_files = train_test_split()
    # train data loader
    print(f" *** Loading training data from {len(train_files)} files ... ")
    if train:
        train_data, valid_data = ReadDataParallel(train_files, -1, num_process=16) # data in all domains
        train_loader = SingleEmbedRec_mpf(train_data, domain)
        valid_loader = SingleEmbedRec_mpf(valid_data, domain)
        return train_loader, valid_loader
    else:
        # test data loader
        print(f" *** Loading testing data from {len(test_files)} files ... ")
        train_data, valid_data = ReadDataParallel(train_files, domain, num_process=16)  # data in all domains
        test_data = ReadDataParallelTest(test_files, domain)
        test_loader = single_domain_test_data(train_data, valid_data, test_data)
        return test_loader


def single_domain_loader_mpf2(domain, train=True):
    train_files, test_files = train_test_split()
    # train data loader
    print(f" *** Loading training data from {len(train_files)} files ... ")
    if train:
        train_data, valid_data = ReadDataParallel(train_files, -1, num_process=16) # data in all domains
        train_loader = SingleEmbedRec_mpf2(train_data)
        valid_loader = SingleEmbedRec_mpf2(valid_data)
        return train_loader, valid_loader
    else:
        # test data loader
        print(f" *** Loading testing data from {len(test_files)} files ... ")
        train_data, valid_data = ReadDataParallel(train_files, domain, num_process=16)  # data in all domains
        test_data = ReadDataParallelTest(test_files, domain)
        test_loader = single_domain_test_data(train_data, valid_data, test_data)
        return test_loader


def single_domain_loader_cmf(domain, train=True):
    train_files, test_files = train_test_split()
    # train data loader
    print(f" *** Loading training data from {len(train_files)} files ... ")
    if train:
        train_data, valid_data = ReadDataParallel_cmf(train_files, num_process=16)      # data in all domains
        train_loader = SingleEmbedRec_cmf(train_data)
        valid_loader = SingleEmbedRec_cmf(valid_data)
        return train_loader, valid_loader
    else:
        print(f" *** Loading testing data from {len(test_files)} files ... ")
        train_data, valid_data = ReadDataParallelTest_cmf(train_files, domain, num_process=16)  # data in all domains
        test_data = ReadDataParallelTest_cmf(test_files, domain, num_process=16)
        test_loader = single_domain_test_data_cmf(train_data, valid_data, test_data)
        return test_loader


def single_domain_test_data_cmf(train_data, valid_data, test_data):
    ground_truth = test_data['feature']
    user = test_data['uid']
    train_pos_dict = dict(zip(train_data['uid'], train_data['feature']))
    for ind, u in enumerate(valid_data['uid']):
        train_pos_dict[u].extend(valid_data['feature'][ind])
    train_pos = []
    for u in user:
        train_pos.append(train_pos_dict[u])
    return {"user": user, "truth": ground_truth, 'train_pos': train_pos}


def single_domain_test_data_mpf(train_data, valid_data, test_data, domain):
    ground_truth = test_data['feature']
    user = test_data['uid']
    train_pos_dict = dict(zip(train_data['uid'], train_data['feature'][domain]))
    for ind, u in enumerate(valid_data['uid']):
        train_pos_dict[u].extend(valid_data['feature'][domain][ind])
    train_pos = []
    for u in user:
        train_pos.append(train_pos_dict[u])
    return {"user": user, "truth": ground_truth, 'train_pos': train_pos}


def multi_domain_loader(length, pad, train=True):
    train_files, test_files = train_test_split()
    np.random.shuffle(train_files)
    train_num = int(len(train_files) * 0.85)
    train_loader, valid_loader = None, None
    # train data loader
    if train:
        print(f" *** Loading training data from {train_num} files ... ")
        start = time.time()
        train_loader = MulCrossRec(train_files[0:train_num], length, pad)
        print(f" ** Using {time.time() - start} seconds")
        # valid data loader
        print(f" *** Loading validation data from {len(train_files) - train_num} files ... ")
        start = time.time()
        valid_loader = MulCrossRec(train_files[train_num:], length, pad)
        # train: indicating whether to insert pos item to candidate set. (for recommendation task only)
        print(f" ** Using {time.time() - start} seconds")
    # test data loader
    print(f" *** Loading testing data from {len(test_files)} files ... ")
    start = time.time()
    # TODO change n_neg for recommender
    test_loader = MulCrossRec(test_files, length, pad)
    print(f" ** Using {time.time() - start} seconds")
    return train_loader, valid_loader, test_loader


####################################
#       Amazon datasets
####################################
def read_csv_data(filename):
    train_data = {'uid': [], 'feature': []}
    uid = 0
    item_id = 0
    n_inter = 0
    df = pd.read_csv(filename)
    for index, row in df.iterrows():
        item_list = eval(row['item'])
        if len(item_list) < 1:
            continue
        uid = max(uid, int(row['uid']))
        item_id = max(item_id, max(item_list))
        train_data['uid'].append(int(row['uid']))
        train_data['feature'].append(item_list)
        n_inter += len(item_list)
    print("max user id", uid)
    print("max item id", item_id)
    print(f"number of interactions: {n_inter}")
    return train_data


def single_domain_loader_ama(domain, train=True):
    data_path = "/data/ceph/seqrec/UMMD/data/amazon_p/CDR"
    domains = ['Books', 'Electronics', 'Movies', 'Sports', 'Video']
    n_items = [274552,  94657, 41896, 76172, 24649]
    domain_name = domains[domain]
    # load train
    print("loading train file ")
    train_data = read_csv_data(os.path.join(data_path, domain_name + "_train.csv"))
    print("loading valid file")
    valid_data = read_csv_data(os.path.join(data_path, domain_name + "_valid.csv"))
    print("loading test file")
    test_data = read_csv_data(os.path.join(data_path, domain_name + "_test.csv"))
    # generate data loader
    train_loader, valid_loader = None, None
    if train:
        train_loader = SingleEmbedRec(train_data, domain, n_item=n_items[domain])
        valid_loader = SingleEmbedRec(valid_data, domain, n_item=n_items[domain])
    test_loader = single_domain_test_data(train_data, valid_data, test_data)
    return train_loader, valid_loader, test_loader
