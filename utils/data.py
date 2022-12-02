import _pickle as pickle
import random
import sys
import time

import numpy as np
import pandas as pd
import torch.utils.data as data


def load_pickle(filename):
    return pickle.load(open(filename, "rb"))


def save_pickle(filename, data_save, mode='wb'):
    with open(filename, mode) as fid:
        pickle.dump(data_save, fid, -1)


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


def data_transformer_bpr(input_dict):
    """
    BPR: user embedding, pos_embedding, neg_embedding
    "tag_set" padded and with 5w                      : user embedding
    "pos": one sampled from tag_set (not 0 nor 50001) : pos embedding
    "neg":                                            : neg embedding
    """
    for key in ["tag_set"]:
        index = input_dict[key] > 50000
        input_dict[key][index] = 50001
    # select pos and neg
    tag_index = np.nonzero(input_dict['tag_set'] != 0)[0]
    pos_index = np.random.randint(0, len(tag_index), 1)[0]
    input_dict["pos"] = input_dict['tag_set'][tag_index[pos_index]]

    # neg_index = np.random.randint(0, 10, 1)[0]
    # input_dict["neg"] = input_dict['neg_set'][neg_index]
    neg_index = np.random.randint(0, 50000, 1)[0] + 1
    input_dict["neg"] = neg_index
    return input_dict


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


class DataLoaderMul(data.Dataset):
    def __init__(self, filename, length, pad_ind, man_mask=False):
        self.man_mask = man_mask
        self.pad_ind = pad_ind
        self.length = length
        # start = time.time()
        self.data = {"feature": [[], [], [], [], []], 'age': [], 'gender': []}
        for file in filename:
            data_tmp = load_pickle(file)
            for i in range(5):
                self.data['feature'][i].extend(data_tmp['feature'][i])
            self.data['age'].extend(data_tmp['age'])
            self.data['gender'].extend(data_tmp['gender'])
        # print(f' time: {time.time() - start} ...')
        self.sparsity = 0
        self.source_domains = np.arange(len(self.data['feature']))

    def __len__(self):
        return len(self.data['age'])

    def set_sparsity(self, sparsity):
        self.sparsity = sparsity

    def set_mask(self, manual=False):
        self.man_mask = manual

    def __getitem__(self, index):
        tag_set_all = [self.data['feature'][i][index] for i in range(len(self.data['feature']))]
        age = deal_with_age_qq(self.data['age'][index])
        age = mult_hot(age)
        gender = self.data['gender'][index]

        # mask a random number of domain or mask a fixed number of domain
        if self.man_mask:
            self.sparsity = np.random.choice(5, 1)[0]  # p=[0.2, 0.2, 0.2, 0.1, 0.1]
            # self.sparsity = 3
        masked_domains = np.random.choice(self.source_domains, self.sparsity, replace=False)  # fixed manual sparse

        mask = []
        for i in range(5):  # 5 domains.
            tag_set_all[i] = CutPad(self.length[i], tag_set_all[i], self.pad_ind).astype(int)
            # return original data, but masking is suggested in "mask" variable
            if i in masked_domains:
                mask.append(1)
            else:
                mask.append(0)
        return {"tag_set": tag_set_all, "mask": np.array(mask), "age": age, "gender": gender}


##########################################
# data loader for attribute prediction
##########################################
def mult_hot(label):
    """
    Args:
        label (int): 0-34
    Return:
        multi-hot embedding of label: dim:  38 (0-37)
    """
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


class SingleEmbed(data.Dataset):
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


def read_single_file(single_args):
    filename, target_domain = single_args[0], int(single_args[1])

    if target_domain < 0:
        result_data = {"feature": [[], [], [], [], []]}
    else:
        result_data = {"feature": []}
    result_data['age'] = []
    result_data['gender'] = []

    try:
        data_tmp = pickle.load(open(filename, 'rb'))
        if target_domain >= 0 and "0.pickle" not in filename:
            # if we are loading single-domain data and the file does not contain all the dense users.
            for i in range(len(data_tmp['uid'])):
                if len(data_tmp['feature'][target_domain][i]) > 0:
                    result_data['feature'].append(data_tmp['feature'][target_domain][i])
                    result_data['age'].append(data_tmp['age'][i])
                    result_data['gender'].append(data_tmp['gender'][i])
        elif target_domain >= 0:  # single domain but dense file
            result_data['feature'] = data_tmp['feature'][target_domain]
            result_data['age'] = data_tmp['age']
            result_data['gender'] = data_tmp['gender']
        else:  # target_domain < 0, loading multi-domain data
            for key in result_data.keys():
                if key == "feature":
                    for ind in range(5):
                        result_data['feature'][ind] = data_tmp['feature'][ind]
                result_data["age"] = data_tmp['age']
                result_data["gender"] = data_tmp['gender']
    except OSError as e:
        print(filename)
        print(e)
    return filename, result_data


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
        final_result = {"feature": [[], [], [], [], []]}
    else:
        final_result = {"feature": []}
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


class SingleEmbedRec(data.Dataset):
    def __init__(self, data_tmp, domain, n_neg=1, n_item=None, mask='False'):
        self.n_neg = n_neg
        # load data way 1:
        # self.pos = data_tmp['feature']
        # self.user = data_tmp['uid']
        # self.pos_dict = data_tmp['feature']
        # way 2:
        self.pos = []
        self.user = []
        self.pos_dict = dict(zip(data_tmp['uid'], data_tmp['feature']))
        self.user_uni = data_tmp['uid']
        for ind, val in enumerate(data_tmp['feature']):
            user = data_tmp['uid'][ind]
            for item in val:
                self.user.append(user)
                self.pos.append(item)

        if n_item is None:
            if domain > 1:
                self.n_item = 50000
            else:
                self.n_item = 100000
            # assert len(self.user) == np.max(self.user)   # max user id
        else:
            self.n_item = n_item
        # self.allItems = set(range(self.n_item))
        self.n_interaction = 0
        for val in self.pos_dict.values():
            self.n_interaction += len(val)

        self.mask = mask
        self.domain = domain

    def set_mask(self, val='False'):
        self.mask = val

    def __getitem__(self, index):
        if self.mask != 'False':
            user = self.user_uni[index]
            mask = np.zeros(5, dtype=float)
            if self.mask == "rand":
                index = np.random.randint(0, 5)
            else:
                index = np.random.randint(0, 5)
                while index == self.domain:
                    index = np.random.randint(0, 5)
            mask[index] = 1.0
            return {"uid": user, 'mask': mask}
        else:
            user = self.user[index]
            # way 1:
            # pos = np.random.choice(self.pos[index], 1)[0]
            pos = self.pos[index]
            neg = np.random.randint(1, self.n_item + 1)
            while neg in self.pos_dict[user]:
                neg = np.random.randint(1, self.n_item + 1)
            return {"uid": user, "pos": pos, "neg": neg}

    def __len__(self):
        if self.mask != "False":
            return len(self.pos_dict)
        else:
            return len(self.user)


class SingleEmbedRec_mpf(data.Dataset):
    def __init__(self, data_tmp, domain, n_neg=1, n_item=None):
        self.n_neg = n_neg
        # load data way 1:
        # self.pos = data_tmp['feature']
        # self.user = data_tmp['uid']
        # self.pos_dict = data_tmp['feature']
        # way 2:
        self.pos = []
        self.user = []
        self.pos_dicts = dict(zip(data_tmp['uid'],
                                  zip(data_tmp['feature'][0], data_tmp['feature'][1], data_tmp['feature'][2],
                                      data_tmp['feature'][3], data_tmp['feature'][4])
                                  ))
        self.domain = domain
        self.user_uni = data_tmp['uid']

        # for single domain training
        for ind, val in enumerate(data_tmp['feature'][domain]):
            user = data_tmp['uid'][ind]
            for item in val:
                self.user.append(user)
                self.pos.append(item)

        if n_item is None:
            if domain > 1:
                self.n_item = 50000
            else:
                self.n_item = 100000
            # assert len(self.user) == np.max(self.user)   # max user id
        else:
            self.n_item = n_item
        # self.allItems = set(range(self.n_item))
        self.n_interaction = 0
        for val in self.pos_dicts.values():
            try:
                self.n_interaction += len(val[domain])
            except:
                print(domain)
                print(len(val))
                print(type(val))
                sys.exit()

    def __getitem__(self, index):
        user = self.user[index]
        # within domain
        pos = self.pos[index]
        neg = np.random.randint(1, self.n_item + 1)
        while neg in self.pos_dicts[user][self.domain]:
            neg = np.random.randint(1, self.n_item + 1)
        # global
        # randomly chose domain
        domain = np.random.randint(0, 5)
        while len(self.pos_dicts[user][domain]) < 1:
            domain = np.random.randint(0, 5)
        # randomly pos sample.
        pos_g = np.random.choice(self.pos_dicts[user][domain])
        if domain > 1:
            domain_item = 50000
            off_set = 200000 + (domain - 2) * 50000
        else:
            domain_item = 100000
            off_set = domain * 100000

        neg_g = np.random.randint(1, domain_item + 1)
        while neg_g in self.pos_dicts[user][domain]:
            neg_g = np.random.randint(1, domain_item + 1)

        return {"uid": user, "pos": pos, "neg": neg, "pos_g": pos_g + off_set, "neg_g": neg_g + off_set}

    def __len__(self):
        return len(self.user)


class SingleEmbedRec_mpf2(data.Dataset):
    def __init__(self, data_tmp, n_neg=1, n_item=None):
        self.n_neg = n_neg
        # self.pos = []
        self.user = data_tmp['uid']
        self.pos_dicts = dict(zip(data_tmp['uid'],
                                  zip(data_tmp['feature'][0], data_tmp['feature'][1], data_tmp['feature'][2],
                                      data_tmp['feature'][3], data_tmp['feature'][4])
                                  ))
        self.user_uni = data_tmp['uid']

        #
        # for domain_n in range(5):
        #     if domain_n < 1:
        #         off_set = (domain_n - 1) * 100000
        #     else:
        #         off_set = (domain_n - 2) * 50000 + 200000
        #     for ind, val in enumerate(data_tmp['feature'][domain_n]):
        #         user = data_tmp['uid'][ind]
        #         for item in val:
        #             self.user.append(user)
        #             self.pos.append(item + off_set)

        # self.allItems = set(range(self.n_item))
        self.n_interaction = 0
        for val in self.pos_dicts.values():
            try:
                for i in range(5):
                    self.n_interaction += len(val[i])
            except:
                print(len(val))
                print(type(val))
                sys.exit()

        print(self.pos_dicts[12])

    def __getitem__(self, index):
        user = self.user[index]
        # within domain
        # pos = self.pos[index]
        pos = []
        neg = []
        #  pos and neg in each domain
        for i in range(5):
            if i < 1:
                n_item = 100000
                off_set = i * 100000
            else:
                n_item = 50000
                off_set = (i - 2) * 50000 + 200000
            if len(self.pos_dicts[user][i]) > 0:
                pos_d = np.random.choice(self.pos_dicts[user][i])
            else:
                pos_d = 0
            neg_d = np.random.randint(1, n_item + 1)
            while neg_d in set(self.pos_dicts[user][i]):
                neg_d = np.random.randint(1, n_item + 1)
            pos.append(pos_d + off_set)
            neg.append(neg_d + off_set)

        return {"uid": user, "pos": np.array(pos), "neg": np.array(neg)}

    def __len__(self):
        return len(self.user)


class SingleEmbedRec_cmf(data.Dataset):
    def __init__(self, data_tmp, n_neg=1, n_item=None):
        self.n_neg = n_neg
        self.pos = []
        self.user = []
        self.pos_dicts = dict(zip(data_tmp['uid'],
                                  data_tmp['feature'],
                                  ))
        self.user_uni = data_tmp['uid']

        # for single domain training
        for ind, val in enumerate(data_tmp['feature']):
            user = data_tmp['uid'][ind]
            for item in val:
                self.user.append(user)
                self.pos.append(item)
        self.n_item = 200000 + 50000 * 3
        # self.allItems = set(range(self.n_item))
        self.n_interaction = 0
        # for val in self.pos_dicts.values():
        #     self.n_interaction += len(val[domain])

    def __getitem__(self, index):
        user = self.user[index]
        # within domain
        pos = self.pos[index]
        neg = np.random.randint(1, self.n_item + 1)
        while neg in self.pos_dicts[user]:
            neg = np.random.randint(1, self.n_item + 1)
        return {"uid": user, "pos": pos, "neg": neg}

    def __len__(self):
        return len(self.user)


class SingleEmbedRecTest(data.Dataset):
    def __init__(self, data_train, data_valid, data_test, domain):
        # load data
        self.ground_truth = data_test['feature']
        self.user = data_test['uid']
        if domain > 1:
            self.n_item = 50000
        else:
            self.n_item = 100000
        assert len(self.user) == np.max(self.user)  # max user id
        # self.allItems = set(range(self.n_item))
        # remove_pos
        self.train_pos_dict = dict(zip(data_train['uid'], data_train['feature']))
        for ind, u in enumerate(data_valid['uid']):
            self.train_pos_dict[u].extend(data_valid['feature'][ind])

    def __getitem__(self, index):
        user, pos = self.user[index], self.ground_truth[index]
        train_pos = self.train_pos_dict[user]
        return {"uid": user, "pos": pos, "train_pos": train_pos}

    def __len__(self):
        return len(self.user)


class SingleEmbedRec_ama(data.Dataset):
    def __init__(self, data_tmp, domain, n_neg=1):
        self.n_neg = n_neg
        # load data
        self.pos = data_tmp['feature']
        self.user = data_tmp['uid']
        if domain > 1:
            self.n_item = 50000
        else:
            self.n_item = 100000
        assert len(self.user) == np.max(self.user)  # max user id
        # self.allItems = set(range(self.n_item))
        self.n_interaction = 0
        for val in self.pos:
            self.n_interaction += len(val)

    def __getitem__(self, index):
        user = self.user[index]
        pos = np.random.choice(self.pos[index], 1)[0]
        neg = np.random.randint(1, self.n_item + 1)
        while neg in self.pos[index]:
            neg = np.random.randint(1, self.n_item + 1)
        return {"uid": user, "pos": pos, "neg": neg}

    def __len__(self):
        return len(self.user)


class SingleEmbedRecTest_ama(data.Dataset):
    def __init__(self, data_train, data_valid, data_test, domain):
        # load data
        self.ground_truth = data_test['feature']
        self.user = data_test['uid']
        if domain > 1:
            self.n_item = 50000
        else:
            self.n_item = 100000
        assert len(self.user) == np.max(self.user)  # max user id
        # self.allItems = set(range(self.n_item))
        # remove_pos
        self.train_pos_dict = dict(zip(data_train['uid'], data_train['feature']))
        for ind, u in enumerate(data_valid['uid']):
            self.train_pos_dict[u].extend(data_valid['feature'][ind])

    def __getitem__(self, index):
        user, pos = self.user[index], self.ground_truth[index]
        train_pos = self.train_pos_dict[user]
        return {"uid": user, "pos": pos, "train_pos": train_pos}

    def __len__(self):
        return len(self.user)


class MulCrossRec(data.Dataset):
    # TODO
    def __init__(self, filename, length, pad_ind, man_mask=False):
        self.man_mask = man_mask
        self.pad_ind = pad_ind
        self.length = length
        # start = time.time()
        self.data = {"feature": [[], [], [], [], []], 'age': [], 'gender': []}
        for file in filename:
            data_tmp = load_pickle(file)
            for i in range(5):
                self.data['feature'][i].extend(data_tmp['feature'][i])
            self.data['age'].extend(data_tmp['age'])
            self.data['gender'].extend(data_tmp['gender'])
        # print(f' time: {time.time() - start} ...')
        self.sparsity = 0
        self.source_domains = np.arange(len(self.data['feature']))

    def __len__(self):
        return len(self.data['age'])

    def set_sparsity(self, sparsity):
        self.sparsity = sparsity

    def set_mask(self, manual=False):
        self.man_mask = manual

    def __getitem__(self, index):
        tag_set_all = [self.data['feature'][i][index] for i in range(len(self.data['feature']))]
        age = deal_with_age_qq(self.data['age'][index])
        age = mult_hot(age)
        gender = self.data['gender'][index]

        # mask a random number of domain or mask a fixed number of domain
        if self.man_mask:
            self.sparsity = np.random.choice(5, 1)[0]  # p=[0.2, 0.2, 0.2, 0.1, 0.1]
            # self.sparsity = 3
        masked_domains = np.random.choice(self.source_domains, self.sparsity, replace=False)  # fixed manual sparse

        mask = []
        for i in range(5):  # 5 domains.
            tag_set_all[i] = CutPad(self.length[i], tag_set_all[i], self.pad_ind).astype(int)
            # return original data, but masking is suggested in "mask" variable
            if i in masked_domains:
                mask.append(1)
            else:
                mask.append(0)
        return {"tag_set": tag_set_all, "mask": np.array(mask), "age": age, "gender": gender}


class MulColdRec(data.Dataset):
    # TODO
    def __init__(self, filename, length, pad_ind, man_mask=False):
        self.man_mask = man_mask
        self.pad_ind = pad_ind
        self.length = length
        # start = time.time()
        self.data = {"feature": [[], [], [], [], []], 'age': [], 'gender': []}
        for file in filename:
            data_tmp = load_pickle(file)
            for i in range(5):
                self.data['feature'][i].extend(data_tmp['feature'][i])
            self.data['age'].extend(data_tmp['age'])
            self.data['gender'].extend(data_tmp['gender'])
        # print(f' time: {time.time() - start} ...')
        self.sparsity = 0
        self.source_domains = np.arange(len(self.data['feature']))

    def __len__(self):
        return len(self.data['age'])

    def set_sparsity(self, sparsity):
        self.sparsity = sparsity

    def set_mask(self, manual=False):
        self.man_mask = manual

    def __getitem__(self, index):
        tag_set_all = [self.data['feature'][i][index] for i in range(len(self.data['feature']))]
        age = deal_with_age_qq(self.data['age'][index])
        age = mult_hot(age)
        gender = self.data['gender'][index]

        # mask a random number of domain or mask a fixed number of domain
        if self.man_mask:
            self.sparsity = np.random.choice(5, 1)[0]  # p=[0.2, 0.2, 0.2, 0.1, 0.1]
            # self.sparsity = 3
        masked_domains = np.random.choice(self.source_domains, self.sparsity, replace=False)  # fixed manual sparse

        mask = []
        for i in range(5):  # 5 domains.
            tag_set_all[i] = CutPad(self.length[i], tag_set_all[i], self.pad_ind).astype(int)
            # return original data, but masking is suggested in "mask" variable
            if i in masked_domains:
                mask.append(1)
            else:
                mask.append(0)
        return {"tag_set": tag_set_all, "mask": np.array(mask), "age": age, "gender": gender}


###################################
# AMAZON DATASETS
###################################
class AmazonSingleRec(data.Dataset):
    def __init__(self, filename, n_item, n_neg=1):
        """
        Args:
            filename: xxx.csv file with columns: 'uid', 'item', 'rate'
            n_neg:
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

    def __getitem__(self, index):
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


class AmazonCrossRec(data.Dataset):
    def __init__(self, filename, n_item, n_neg=1):
        """ TODO
        Args:
            filename: xxx.csv file with columns: 'uid', 'item', 'rate'
            n_neg:
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

    def __getitem__(self, index):
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


class AmazonColdRec(data.Dataset):
    def __init__(self, filename, n_item, n_neg=1):
        """ TODO
        Args:
            filename: xxx.csv file with columns: 'uid', 'item', 'rate'
            n_neg:
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

    def __getitem__(self, index):
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


class AmazonSingleRecTest_bus(data.Dataset):
    def __init__(self, test_data, n_item):
        """
        Args:
            filename: xxx.csv file with columns: 'uid', 'item', 'rate'
                list: [train.csv: test.csv]
            n_neg:
            train:
        """
        self.n_item = n_item
        self.user_item_test = {"user": test_data['user'], "item": test_data['truth']}
        self.user_item_train = test_data['train_pos']

    def getUserPosItems(self, users):
        test_pos = []
        train_pos = []
        for user_tmp in users:  # TODO, why include user id here?
            test_index = self.user_item_test['user'].index(user_tmp)
            test_pos.append(self.user_item_test['item'][test_index])
            train_pos.append(self.user_item_train[test_index])

        return test_pos, train_pos

    def __getitem__(self, index):
        uid = self.user_item_test['user'][index]
        return {"uid": uid}

    def __len__(self):
        return len(self.user_item_test['user'])
