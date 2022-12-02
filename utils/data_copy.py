import _pickle as pickle
import copy
import os
import random
import sys
import time 
import torch.utils.data as data
from torch.utils.data import DataLoader
import pandas as pd

import numpy as np
from tfrecord.torch import TFRecordDataset, MultiTFRecordDataset


def load_pickle(filename):
    return pickle.load(open(filename, "rb"))


def save_pickle(filename, data, mode='wb'):
    with open(filename, mode) as fid:
        pickle.dump(data, fid, -1)


def deal_with_age_qq(line_components):
    """History predict age encode.
    Encoding the history version prediction age, the same encode methods as target age encodes.

    :param line_components: History predict age

    :return: History predict age encoding result.
    """
    dapan_age = int(line_components)
    if dapan_age <= 12:
        dapan_age = 0
    elif dapan_age <= 46:
        dapan_age -= 12
    else:
        dapan_age = 34
    return dapan_age


def TagSetProcessDense(tag, pad, max_length):
    np.random.shuffle(tag)
    length = len(tag)
    if length >= max_length:
        return tag[0:max_length]
    else:
        return np.concatenate((np.array([pad] * (max_length - length)), tag), axis=0)


def norm_prob(in_data):
    total = np.sum(in_data)
    return [val / total for val in in_data]


def CutPad(length, tag_set, pad_ind):
    tag_len = len(tag_set)
    if length >= tag_len:
        return np.array([pad_ind] * (length - tag_len) + tag_set)
    else:
        return np.array(tag_set[0: length])


def mult_hot(label):
    '''
    Args:
        label (int): 0-34
    Return: 
        multi-hot embedding of label: dim:  38 (0-37)      
    '''
    result = np.zeros(38)
    result[label] = 1.0
    # +/- 3 
    for i in range(1, 4):
        val_neg = label - i 
        val_pos = label + i 
        if val_neg >= 0: 
            result[val_neg] = 1.0
        result[val_pos] = 1.0

    return result


def read_single_file(single_args):
    filename, target_domain = single_args[0], int(single_args[1])
    
    if target_domain < 0:
        result_data = {"feature":[[], [], [], [], []]}
    else:
        result_data = {"feature":[]}
    result_data['age'] = []
    result_data['gender'] = []

    try:     
        data = pickle.load(open(filename, 'rb'))
        if target_domain >= 0 and "0.pickle" not in filename:
            # if we are loading single-domain data and the file does not contain all the dense users.
            for i in range(len(data['uid'])):
                if len(data['feature'][target_domain][i]) > 0:
                    result_data['feature'].append(data['feature'][target_domain][i])
                    result_data['age'].append(data['age'][i])
                    result_data['gender'].append(data['gender'][i])
        elif target_domain >=0:  # single domain but dense file
            result_data['feature'] = data['feature'][target_domain]
            result_data['age'] = data['age']
            result_data['gender'] = data['gender']
        else:   # target_domain < 0, loading multi-domain data
            for key in result_data.keys():
                if key == "feature":
                    for ind in range(5):
                        result_data['feature'][ind] = data['feature'][ind]
                result_data["age"] = data['age']
                result_data["gender"] = data['gender']
    except OSError as e:
        print(filename)
        print(e)
    return filename, result_data


def ReadDataParallel(files, target_domain, num_process=8):
    from multiprocessing import Pool
    
    target_domains = list(np.ones(len(files)) * target_domain )
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
    print("Finishing loading:", time.time() - start )
    # collect all data
    if target_domain < 0:
        final_result = {"feature":[[], [], [], [], []]}
    else:
        final_result = {"feature":[]}
    final_result['age'] = []
    final_result['gender'] = []

    for val in df_collection:
        # print(len(val[1]['age'])) 
        for key in final_result.keys():
            if target_domain < 0:
                if key == "feature":
                    for i in range(5):
                        final_result[key][i].extend(val[1][key][i])
                else:
                    final_result[key].extend(val[1][key])    
            else:
                final_result[key].extend(val[1][key])
    # print(len(final_result['age']))
    return final_result


#############################
# Recommendation data loaders
#############################
class DataSingleRec(data.Dataset):
    def __init__(self, filename, domain, length, pad_ind, n_neg, train=True):
        self.domain = domain
        self.train = train
        self.pad_ind = pad_ind
        self.length = length
        self.n_neg = n_neg
        # load data
        self.tag = []
        # self.neg = None

        for file in filename:
            data_tmp = load_pickle(file)
            self.tag.extend(data_tmp['feature'][domain])

        if domain > 1:
            self.n_item = 50000
        else:
            self.n_item = 100000
        self.p = np.ones(self.n_item)

    def __getitem__(self, index):
        tag_set = self.tag[index]
        if len(tag_set) >= 5:  # including 0
            if 0 in tag_set:
                tag_set.remove(0)
            # pos
            pos = np.random.choice(tag_set, 1, replace=False)[0]
            tag_set.remove(pos)
            # tag set
            tag_len = len(tag_set)
            if self.length >= tag_len:
                tag_set_f = np.array([self.pad_ind] * (self.length - tag_len) + tag_set)
            else:
                tag_set_f = np.array(tag_set[0:self.length])
            # neg
            # p = copy.deepcopy(self.p)
            # p[np.array(tag_set) - 1] = 0
            # neg = np.random.choice(self.n_item, self.n_neg, replace=False) + 1
            neg = random.sample(range(1, self.n_item + 1), self.n_neg)
            # if self.train:
            #     # while neg in tag_set:
            #     #     neg = np.random.randint(0, self.n_item, self.n_neg)[0] + 1
            # else:   # test there can be overlapping.
            #
            #     neg = np.random.choice(self.n_item, self.n_neg, p=p) + 1
            # # in case there is overlap.
            # while len(set(tag_set) & set(neg)):
            #     neg = np.random.randint(0, self.n_item, self.n_neg) + 1
            if not self.train:  # test
                neg = np.insert(neg, 0, pos)  # insert pos in front of all neg samples
            neg = np.array(neg)
            pos = np.array([pos])
        else:
            print("Wrong user, ")
            if self.train:
                neg = np.zeros(self.n_neg)
            else:
                neg = np.zeros(self.n_neg + 1)
            pos = np.array([0])
            tag_set_f = np.zeros(self.length)

        if self.train:
            return {"tag_set": tag_set_f, "pos": pos, "neg": neg}
        else:
            return {"tag_set": tag_set_f, "pos": neg, "neg": np.array([0])}

    def __len__(self):
        return len(self.tag)


class DataRecCross(data.Dataset):
    def __init__(self, filename, domain, length, pad_ind, n_neg, train=True, sparse=False):
        self.domain = domain
        self.train = train
        self.sparse = sparse
        self.pad_ind = pad_ind
        self.length = length  # list of int
        self.n_neg = n_neg
        # load data
        self.tag = [[], [], [], [], []]
        for file in filename:
            # print(file)
            data_tmp = load_pickle(file)
            for i in range(5):
                self.tag[i].extend(data_tmp['feature'][i])
        # self.print_data()
        self.sparsity = 0
        self.source_domains = list(range(5))
        self.source_domains.remove(domain)

        if self.domain > 1:
            self.n_item = 50000
        else:
            self.n_item = 100000
        # self.p = np.ones(self.n_item)

    def set_sparsity(self, sparsity=0):
        self.sparsity = sparsity

    def print_data(self):
        print(len(self.tag[0]))
        print(len(self.tag[1]))
        print(len(self.tag[2]))
        print(len(self.tag[3]))
        print(len(self.tag[4]))
        print(self.tag[0][0])  # list

    def __getitem__(self, index):
        tag_set = self.tag[self.domain][index]  # target set
        if len(tag_set) >= 5:  # including 0
            if 0 in tag_set:
                tag_set.remove(0)
            # pos
            pos = np.random.choice(tag_set, 1, replace=False)[0]
            tag_set.remove(pos)
            # tag set
            tag_set_f = CutPad(self.length[self.domain], tag_set, self.pad_ind)
            # neg
            # p = copy.deepcopy(self.p)
            # p[np.array(tag_set) - 1] = 0
            # neg = np.random.choice(self.n_item, self.n_neg, replace=False) + 1
            neg = random.sample(range(1, self.n_item + 1), self.n_neg)
            if not self.train:  # test
                neg = np.insert(neg, 0, pos)
            # origin
            # if self.train:
            #     neg = np.random.randint(0, self.n_item, self.n_neg)[0] + 1
            #     while neg in tag_set:
            #         neg = np.random.randint(0, self.n_item, self.n_neg)[0] + 1
            # else:
            #     neg = np.random.randint(0, self.n_item, self.n_neg) + 1
            #     while len(set(tag_set) & set(neg)):
            #         neg = np.random.randint(0, self.n_item, self.n_neg) + 1
            #     neg = np.insert(neg, 0, pos)
            neg = np.array(neg, dtype=int)
            pos = np.array([pos])
        else:
            print("wrong user")
            if self.train:
                neg = np.zeros(self.n_neg, dtype=int)
            else:
                neg = np.zeros(self.n_neg + 1, dtype=int)
            pos = np.array([0])
            tag_set_f = np.zeros(self.length[self.domain], dtype=int)

        # tag set in other domain according to sparsity.
        if self.sparse:
            # random sparsity
            self.sparsity = np.random.choice(5, 1, p=[0, 0.25, 0.25, 0.25, 0.25])[0]
        masked_domains = np.random.choice(self.source_domains, self.sparsity, replace=False)

        tag_all = []
        mask = []
        for i in range(5):  # 5 domains.
            if i == self.domain:
                tag_all.append(tag_set_f.astype(int))
                mask.append(0)
            elif i not in masked_domains:
                tag_tmp = self.tag[i][index]
                if len(tag_tmp) >= 5:  # if sparse itself
                    tag_all.append(CutPad(self.length[i], tag_tmp, self.pad_ind).astype(int))
                    mask.append(0)
                else:
                    tag_all.append(np.zeros(self.length[i], dtype=int))
                    mask.append(1)
            else:
                tag_all.append(np.zeros(self.length[i], dtype=int))
                mask.append(1)
        if self.train:
            return {"tag_set": tag_all, "pos": pos, "neg": neg, "mask": np.array(mask)}
        else:
            return {"tag_set": tag_all, "pos": neg, "neg": np.array([0]), "mask": np.array(mask)}

    def __len__(self):
        return len(self.tag[0])


##########################
# Prediction data loader #
##########################
class DataAgeCross(data.Dataset):
    # dense users, multi-domains
    def __init__(self, filename, domain, length, pad_ind, train=True, sparse=False):
        """
        sparse: where to randomly masked a random number of domains
        Sparse = False, randomly mask self.sparsity number of domains

        """
        self.domain = domain
        self.train = train
        self.manual_sparse = sparse
        self.pad_ind = pad_ind
        self.length = length  # list of int
        # load data
        self.tag = [[], [], [], [], []]
        self.target = []
        self.age = []
        self.gender = []
        for file in filename:
            data_tmp = load_pickle(file)
            for i in range(5):
                self.tag[i].extend(data_tmp['feature'][i])
            self.age.extend(data_tmp['age'])
            self.gender.extend(data_tmp['gender'])
        self.sparsity = 0
        self.source_domains = np.arange(len(self.tag))

    def __len__(self):
        return len(self.age)

    def set_sparsity(self, sparsity):
        self.sparsity = sparsity

    def set_sparse(self, manual=False):
        self.manual_sparse = manual

    def __getitem__(self, index):
        tag_set_all = [self.tag[i][index] for i in range(len(self.tag))]
        age = deal_with_age_qq(self.age[index])
        age = mult_hot(age)
        gender = self.gender[index]

        if self.manual_sparse:
            # random manual sparse
            self.sparsity = np.random.choice(5, 1)[0]   # p=[0.2, 0.2, 0.2, 0.1, 0.1]
            # self.sparsity = 3
        masked_domains = np.random.choice(self.source_domains, self.sparsity, replace=False) # fixed manual sparse
        mask = []

        for i in range(5):  # 5 domains.
            tag_set_all[i] = CutPad(self.length[i], tag_set_all[i], self.pad_ind).astype(int)
            if i in masked_domains:        
                mask.append(1)
            else:
                mask.append(0)
        return {"tag_set": tag_set_all, "mask": np.array(mask), "age": age, "gender": gender}


class DataAgeCrossAll(data.Dataset):
    # all users. multi-domains
    def __init__(self, filename, domain, length, pad_ind, train=True, sparse=False):
        self.domain = domain
        self.train = train
        self.manual_sparse = sparse
        self.pad_ind = pad_ind
        self.length = length  # list of int
        # load data
        self.data = ReadDataParallel(filename, -1)

        self.sparsity = 0
        self.source_domains = np.arange(len(self.data['feature']))
    
    def set_sparsity(self, sparsity):
        self.sparsity = sparsity

    def set_sparse(self, manual=False):
        self.manual_sparse = manual

    def __getitem__(self, index):
        tag_set_all = [self.data['feature'][i][index] for i in range(len(self.data['feature']))]
        age = deal_with_age_qq(self.data['age'][index])
        age = mult_hot(age)
        gender = self.data['gender'][index]
        mask_org = []   # input sample is sparse it self
        sparsity_org = 0 
        for val in tag_set_all:
            if len(val) < 1: 
                sparsity_org += 1
                mask_org.append(1)
            else:
                mask_org.append(0)

        if self.manual_sparse:  # 
            sparsity = np.random.choice(5, 1)[0]   # p=[0.2, 0.2, 0.2, 0.1, 0.1]
        else:
            sparsity = self.sparsity
        
        if sparsity > sparsity_org: 
            sparsity -= sparsity_org
        else:
            sparsity = 0

        domains = copy.deepcopy(self.source_domains)        
        index = np.where(mask_org == 1)            
        np.delete(domains, index)
        masked_domains = np.random.choice(domains, sparsity, replace=False)

        manual_mask = []  # manually masked domains

        for i in range(5):  
            tag_set_all[i] = CutPad(self.length[i], tag_set_all[i], self.pad_ind).astype(int)
            if i in masked_domains:
                manual_mask.append(1)
            else:            
                manual_mask.append(0)

        return {"tag_set": tag_set_all, "mask_org": np.array(mask_org), "age": age, "gender": gender, 
                "mask_man": np.array(manual_mask)}

    def __len__(self):
    	return len(self.data['age'])


##########################################
# prediction data loader for single domain
##########################################
class SingleEmbed(data.Dataset):
    # for dense users
    def __init__(self, filename, domain, length, pad_ind, n_neg, train=True):
        self.domain = domain
        self.train = train
        self.pad_ind = pad_ind
        self.length = length
        self.n_neg = n_neg
        # load data
        self.tag = []
        self.age = []
        self.gender = []
        for file in filename:
            data_tmp = load_pickle(file)
            self.tag.extend(data_tmp['feature'][domain])
            self.age.extend(data_tmp['age'])
            self.gender.extend(data_tmp['gender'])
        if domain > 1:
            self.n_item = 50000
        else:
            self.n_item = 100000

    def __getitem__(self, index):
        tag_set = self.tag[index]
        age = deal_with_age_qq(self.age[index])
        # multi-hot emebdding. 
        age = mult_hot(age)

        gender = self.gender[index]
        if len(tag_set) >= 5:          # including 0
            if 0 in tag_set:
                tag_set.remove(0)
            # pos
            pos = np.random.choice(tag_set, 1, replace=False)[0]
            tag_set.remove(pos)  # Remove or not
            # tag set
            tag_len = len(tag_set)
            if self.length >= tag_len:
                tag_set_f = np.array([self.pad_ind] * (self.length - tag_len) + tag_set)
            else:
                tag_set_f = np.array(tag_set[0:self.length])
            # neg
            neg = random.sample(range(1, self.n_item + 1), self.n_neg)
            if not self.train:  # test
                neg = np.insert(neg, 0, pos)  # insert pos in front of all neg samples
            neg = np.array(neg)
            pos = np.array([pos])
        else:
            print("Wrong user, ")
            if self.train:
                neg = np.zeros(self.n_neg)
            else:
                neg = np.zeros(self.n_neg + 1)
            pos = np.array([0])
            tag_set_f = np.zeros(self.length)

        if self.train:
            return {"tag_set": tag_set_f, "pos": pos, "neg": neg, 'age': age, 'gender': gender}
        else:
            return {"tag_set": tag_set_f, "pos": neg, "neg": np.array([0]), 'age': age, 'gender': gender}

    def __len__(self):
        return len(self.tag)


class SingleEmbedAll(data.Dataset):
    # single domain data for all users. 
    def __init__(self, filename, domain, length, pad_ind, n_neg, train=True):
        self.domain = domain
        self.train = train
        self.pad_ind = pad_ind
        self.length = length
        self.n_neg = n_neg
        # load data
        self.data = ReadDataParallel(filename, domain)

        if domain > 1:
            self.n_item = 50000
        else:
            self.n_item = 100000
        print(len(self.data['age']))

    def __getitem__(self, index):
        tag_set = self.data["feature"][index]
        age = deal_with_age_qq(self.data['age'][index])
        # multi-hot emebdding. 
        age = mult_hot(age)

        gender = self.data['gender'][index]
        if len(tag_set) >= 5:          # including 0
            if 0 in tag_set:
                tag_set.remove(0)
            # pos
            pos = np.random.choice(tag_set, 1, replace=False)[0]
            # tag_set.remove(pos)   # Remove or not 
            # tag set
            tag_len = len(tag_set)
            if self.length >= tag_len:
                tag_set_f = np.array([self.pad_ind] * (self.length - tag_len) + tag_set)
            else:
                tag_set_f = np.array(tag_set[0:self.length])
            # neg
            neg = random.sample(range(1, self.n_item + 1), self.n_neg)
            if not self.train:  # test
                neg = np.insert(neg, 0, pos)  # insert pos in front of all neg samples
            neg = np.array(neg)
            pos = np.array([pos])
        else:
            print("Wrong user, ")
            if self.train:
                neg = np.zeros(self.n_neg)
            else:
                neg = np.zeros(self.n_neg + 1)
            pos = np.array([0])
            tag_set_f = np.zeros(self.length)

        if self.train:
            return {"tag_set": tag_set_f, "pos": pos, "neg": neg, 'age': age, 'gender': gender}
        else:
            return {"tag_set": tag_set_f, "pos": neg, "neg": np.array([0]), 'age': age, 'gender': gender}

    def __len__(self):
        return len(self.data['age'])    
    
    
###########################
# Loaders functions
###########################
def data_file_train_age(hyper_args, sparse=False):
    data_dir = hyper_args.data_dir
    files = [val for val in os.listdir(data_dir)]
    total_num = len(files)
    test_num = int(total_num * 0.2)

    file_ind = np.arange(total_num - test_num)
    train_num = int(len(file_ind) * 0.75)
    np.random.shuffle(file_ind)
    train_files = []
    valid_files = []
    # train
    for val in file_ind[0: train_num]:
        val_tmp = val + 100000
        train_files.append(os.path.join(data_dir, f"part-{str(val_tmp)[1:]}.pickle"))
    # valid
    for val in file_ind[train_num:]:
        val_tmp = val + 100000
        valid_files.append(os.path.join(data_dir, f"part-{str(val_tmp)[1:]}.pickle"))

    print("== loading training data (all users in target domain)")
    train_loader = DataAgeCross(train_files, hyper_args.domain, hyper_args.length,
                                hyper_args.pad_index, train=True, sparse=sparse)
    train_loader.set_sparsity(hyper_args.sparsity)                
    train_batch = DataLoader(train_loader, batch_size=hyper_args.batch_size,
                             num_workers=hyper_args.n_worker, pin_memory=True, shuffle=True)
    print("== loading validation data (all users in target domain)")
    valid_loader = DataAgeCross(valid_files, hyper_args.domain, hyper_args.length,
                                hyper_args.pad_index, train=True, sparse=sparse)
    valid_loader.set_sparsity(hyper_args.sparsity)
    valid_batch = DataLoader(valid_loader, batch_size=hyper_args.batch_size,
                             num_workers=hyper_args.n_worker)
    return train_batch, valid_batch


def data_file_train_age_loader(hyper_args, sparse=False):
    data_dir = hyper_args.data_dir
    files = [val for val in os.listdir(data_dir)]
    total_num = len(files)
    test_num = int(total_num * 0.2)

    file_ind = np.arange(total_num - test_num)
    train_num = int(len(file_ind) * 0.75)
    np.random.shuffle(file_ind)
    train_files = []
    valid_files = []
    # train
    for val in file_ind[0: train_num]:
        val_tmp = val + 100000
        train_files.append(os.path.join(data_dir, f"part-{str(val_tmp)[1:]}.pickle"))
    # valid
    for val in file_ind[train_num:]:
        val_tmp = val + 100000
        valid_files.append(os.path.join(data_dir, f"part-{str(val_tmp)[1:]}.pickle"))

    print("== loading training data (all users in target domain)")
    train_loader = DataAgeCross(train_files, hyper_args.domain, hyper_args.length,
                                hyper_args.pad_index, train=True, sparse=sparse)
    print("== loading validation data (all users in target domain)")
    valid_loader = DataAgeCross(valid_files, hyper_args.domain, hyper_args.length,
                                hyper_args.pad_index, train=True, sparse=sparse)
    return train_loader, valid_loader


def data_file_test_age(hyper_args):
    data_dir = hyper_args.data_dir
    files = [val for val in os.listdir(data_dir)]
    total_num = len(files)
    test_num = int(total_num * 0.2)
    test_files_tmp = []
    for val in range(total_num - test_num, total_num):
        val_tmp = val + 100000
        test_files_tmp.append(os.path.join(data_dir, f"part-{str(val_tmp)[1:]}.pickle"))
        print(os.path.join(data_dir, f"part-{str(val_tmp)[1:]}.gz_{0}.pickle"))
    loader = DataAgeCross(test_files_tmp, hyper_args.domain, hyper_args.length,
                          hyper_args.pad_index, train=False)
    return loader


def all_data_train_test(train=True, num_sparse=5):
    data_dir = '/data/ceph/seqrec/UMMD/data/pickle/q36_age_train_org'
    sampled_file_ids = pickle.load(open("/data/ceph/seqrec/UMMD/data/pickle/q36_age_train_org_sel.pickle", 'rb'))
    print(" === Selected file IDs ...")    
    sample_files_train = []
    
    for i in range(num_sparse):
        test_num = int(len(sampled_file_ids[i]) * 0.2)
        if train:
            sample_files_train.append(sampled_file_ids[i][:-test_num])
        else:
            sample_files_train.append(sampled_file_ids[i][-test_num:])

    return  sample_files_train


def data_file_train_age_all(hyper_args, num_sparse=5):
    file_ind = all_data_train_test(train=True, num_sparse=num_sparse) 
    data_dir = '/data/ceph/seqrec/UMMD/data/pickle/q36_age_train_org'

    files = []
    for sparsity in range(num_sparse):
        for selected in file_ind[sparsity]: 
            num = 100000 + selected
            files.append(os.path.join(data_dir, f"part-{str(num)[1:]}.gz_{sparsity}.pickle")) 

    num_files = len(files)
    num_train = int(num_files * 0.8) 
    np.random.shuffle(files)
    train_files = files[0: num_train]
    valid_files = files[num_train: ]

    print(f"== loading training data (all users in target domain from {num_train} files))")
    start = time.time()
    train_loader = DataAgeCrossAll(train_files, hyper_args.domain, hyper_args.length,
                                   hyper_args.pad_index, train=True, sparse=False)
    train_loader.set_sparsity(hyper_args.sparsity)
    train_batch = DataLoader(train_loader, batch_size=hyper_args.batch_size,
                             num_workers=hyper_args.n_worker, pin_memory=True, shuffle=True)
    print(f" *** using {time.time() - start} seconds. {train_loader.__len__()} samples for training")
    print(f"== loading validation data (all users in target domain) ")
    start = time.time()
    valid_loader = DataAgeCrossAll(valid_files, hyper_args.domain, hyper_args.length,
                                hyper_args.pad_index, train=True, sparse=False)
    valid_loader.set_sparsity(hyper_args.sparsity)
    valid_batch = DataLoader(valid_loader, batch_size=hyper_args.batch_size,
                             num_workers=hyper_args.n_worker)
    print(f" *** using {time.time() - start} seconds. {valid_loader.__len__()} samples for validation")
    return train_batch, valid_batch


def data_file_test_age_all(hyper_args, num_sparse=5):
    file_ind = all_data_train_test(train=False, num_sparse=num_sparse) 
    data_dir = '/data/ceph/seqrec/UMMD/data/pickle/q36_age_train_org'
    files = []
    for sparsity in range(num_sparse):
        file_sparse = []
        for selected in file_ind[sparsity]: 
            num = 100000 + selected
            file_sparse.append(os.path.join(data_dir, f"part-{str(num)[1:]}.gz_{sparsity}.pickle")) 
        files.append(file_sparse)
    loaders = []
    for ind, val in enumerate(files): 
        loader = DataAgeCrossAll(val, hyper_args.domain, hyper_args.length,
                              hyper_args.pad_index, train=False, sparse=False)  # no manual saprse
        loaders.append(loader)                              
        print(f" *** Test loader with sparsity {ind}, {loader.__len__()} samples")
    return loaders


###################################
# AMAZON DATASETS
###################################
class AmazonSingleRec(data.Dataset):
    def __init__(self, filename, n_item, n_neg=1):
        """
        Args:
            filename: xxx.csv file with columns: 'uid', 'item', 'rate'
            n_neg:
            train:
        """
        self.n_neg = n_neg
        self.n_item = n_item
        self.uid = []
        self.pos = []
        self.user_item = {"user": [], "item": []}
        df = pd.read_csv(filename)
        for index, row in df.iterrows():
            if len(eval(row['item'])) > 0:
                self.user_item['user'].append(int(row['uid']))
                self.user_item['item'].append(eval(row['item']))
                for val in eval(row['item']):
                    self.uid.append(int(row['uid']))
                    self.pos.append(int(val))
        print(f'number of interaction {len(self.uid)}')
        self.item_set = list(set(self.pos))
        self.user_set = list(set(self.uid))
        print(f"Number of user {len(self.user_set)}, {np.max(self.user_set)}")
        print(f"Number of item {len(self.item_set)}, {np.max(self.item_set)}")
        # for i in range(20):
        #     print(self.uid[i], self.pos[i])
        # sys.exit()

    def __getitem__(self, index ):
        uid, pos = self.uid[index], self.pos[index]
        uid_index = self.user_item['user'].index(uid)
        while True:
            neg = random.sample(range(self.n_item), 1)[0]
            if neg in self.user_item['item'][uid_index]:
                continue
            else:
                break

        return {"uid": np.array(uid), "pos": np.array(pos), "neg": np.array(neg)}

    def __len__(self):        
        return len(self.uid)    


class AmazonSingleRecTest(data.Dataset):
    def __init__(self, filename, n_item, n_neg=1):
        """
        Args:
            filename: xxx.csv file with columns: 'uid', 'item', 'rate'
                list: [train.csv: test.csv]
            n_neg:
            train:
        """
        self.n_neg = n_neg
        self.n_item = n_item
        self.user_item_test = {"user": [], "item": []}
        self.user_item_train = {"user": [], "item": []}
                
        for file in filename:
            df = pd.read_csv(file)
            for index, row in df.iterrows():
                if "train" in file: 
                    self.user_item_train['user'].append(int(row['uid']))
                    self.user_item_train['item'].append(eval(row['item']))
                else:
                    if len(eval(row['item'])) > 0:
                        self.user_item_test['user'].append(int(row['uid']))
                        self.user_item_test['item'].append(eval(row['item']))

    def getUserPosItems(self, users): 
        test_pos = []
        train_pos = []
        for user_tmp in users:  # TODO, why include user id here? 
            test_index = self.user_item_test['user'].index(user_tmp)
            train_index = self.user_item_train['user'].index(user_tmp)

            test_pos.append(self.user_item_test['item'][test_index])  
            train_pos.append(self.user_item_train['item'][train_index])  
            
        return test_pos, train_pos

    def __getitem__(self, index):    
        uid = self.user_item_test['user'][index]
        # self.user_item_test['item'][index]
        # uid_train_index = self.user_item_train['uid'].index(uid)
        # train_pos = self.user_item_train['item'][uid_train_index]
        
        # differetn length ....
        return {"uid": uid}

    def __len__(self):        
        return len(self.user_item_test['user'])
