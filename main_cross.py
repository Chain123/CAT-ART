# -*- coding: utf-8 -*-

"""
Run cross-domain recommendation for
base1, base2 and base3 methods
"""
import argparse
import os
import shutil

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import Data_loader
import run_functions as run_func
import utils
import models
import _pickle as pickle


def main(h_args, print_p=True, train=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = f"Cross-{h_args.cross}-U{h_args.fix_user}-I{h_args.fix_item}-" \
                 f"d{h_args.latent_dim}-r{h_args.num_run}-new"
    out_path = f"{h_args.result_dir}/{model_name}/checkpoint/best_model.pt"
    if train and os.path.isdir(f"{h_args.result_dir}/{model_name}"):
        shutil.rmtree(f"{h_args.result_dir}/{model_name}")
    if not os.path.isdir(f"{h_args.result_dir}/{model_name}"):
        os.mkdir(f"{h_args.result_dir}/{model_name}")
        os.mkdir(f"{h_args.result_dir}/{model_name}/checkpoint")
    # data set
    if h_args.n_users < 20000:
        print("loading amazon dataset")
        train_loader, valid_loader, test_loader = Data_loader.single_domain_loader_ama(h_args.domain, train)
    else:
        train_loader, valid_loader, test_loader = Data_loader.single_domain_loader(h_args.domain, train)
    # build model
    model_config = {'target_domain': h_args.domain,
                    'single_dirs': h_args.single_dirs,
                    'latent_dim': h_args.latent_dim,
                    'device': device,
                    'fix_user': h_args.fix_user,
                    'fix_item': h_args.fix_item}
    # print(model_config)
    model = h_args.cross_models[h_args.cross](model_config)
    if print_p:  # show model parameters
        for name, param in model.named_parameters():
            print(name, param.requires_grad, param.data.size())
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f" *** data parallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    # train single domain prediction model
    if train:
        writer = SummaryWriter(f"{h_args.result_dir}/{model_name}")
        early_stop = utils.EarlyStopping(patience=2, verbose=True, path=out_path)
        opt = optim.Adam(model.parameters(), lr=h_args.lr, weight_decay=0)
        train_batch = DataLoader(train_loader, batch_size=h_args.batch_size,
                                 num_workers=h_args.n_worker, pin_memory=True, shuffle=True)
        valid_batch = DataLoader(valid_loader, batch_size=h_args.batch_size,
                                 num_workers=h_args.n_worker, pin_memory=True)
        round_train = int(train_loader.n_interaction / len(train_loader.user))
        round_valid = int(valid_loader.n_interaction / len(valid_loader.user))
        print(f" *** Number of interactions in train set is: {train_loader.n_interaction}")
        print(f" *** Number of interactions in valid set is: {valid_loader.n_interaction}")
        print(f"train round for all users {round_train}")
        run_config = {"train_loader": train_batch,
                      "valid_loader": valid_batch,
                      "opt": opt,
                      "device": device,
                      "early_stop": early_stop,
                      "epoch": h_args.epoch,
                      "bar_dis": h_args.bar_dis,
                      'train_r': round_train,
                      'valid_r': round_valid,
                      "decay": h_args.weight_decay}
        run_func.train_single_step(model, run_config, writer, cross=True)
    # else:
    # Reload the best model parameters and testing
    print(f"loading model from {out_path}")
    model.load_state_dict(torch.load(out_path, map_location=device))
    # Testing data
    test_result = run_func.test_single_step(model, test_loader, device,
                                            n_process=24,
                                            num_items=h_args.n_items[h_args.domain])
    print(" **** Test result")
    print(test_result)
    # save results
    with open(f"{h_args.result_dir}/{model_name}/result.pickle", 'wb') as fid:
        pickle.dump(test_result, fid, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Unsupervised Recommendation Training')
    # Path Arguments  gender_method
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='batch size')
    parser.add_argument('--domain', type=int, default=1,
                        help='batch size')
    parser.add_argument('--epoch', type=int, default=2000,
                        help='number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='dimensions')
    parser.add_argument('--result_dir', type=str,
                        default="/data/ceph/seqrec/UMMD/www/recommend",
                        help='output dir to store the results')
    parser.add_argument('--n_worker', type=int, default=4,
                        help='number of workers for the train data loader')
    parser.add_argument('--num_run', type=int, default=0,
                        help='number of repeats')
    parser.add_argument('--bpr_neg', type=int, default=1,
                        help='number of negative sample for recommendation task (using bpr loss)')
    parser.add_argument('--train', type=bool, default=False,
                        help='if training')
    parser.add_argument('--bar_dis', type=bool, default=False,
                        help='disable bar plot of tqdm')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay for l2 regularizer')
    parser.add_argument('--n_users', type=int, default=1166552,
                        help='number of users in the dataset')
    parser.add_argument('--amazon', type=bool, default=False,
                        help='amazon datasets ?')
    parser.add_argument('--cross', type=str, default='base1',
                        help='cross method')
    parser.add_argument('--fix_item', type=bool, default=False,
                        help='whether to fix item embeddings in the target domain')
    parser.add_argument('--fix_user', type=bool, default=False,
                        help='whether to fix user embeddings in the target domain')

    params = parser.parse_args()
    params.n_items = [100000, 100000, 50000, 50000, 50000]
    if params.amazon:
        params.n_users = 18347 + 1
        params.n_items = [274552 + 1, 94657 + 1, 41896 + 1, 76172 + 1, 24649 + 1]
        params.result_dir = "/data/ceph/seqrec/UMMD/www/amazon"
    params.result_dir += f"/domain_{params.domain}"

    cross_models = {
        "base1": models.MultiRecommendBase1,
        "base2": models.MultiRecommendBase2,
        "base3": models.MultiRecommendBase3,
    }
    params.cross_models = cross_models
    single_dirs = []
    for i in range(5):
        model_dir = params.result_dir.replace(f"domain_{params.domain}", f"domain_{i}")
        run_name = f"MF-d{params.latent_dim}-r{params.num_run}"
        single_dirs.append(f"{model_dir}/{run_name}/checkpoint/best_model.pt")
    params.single_dirs = single_dirs
    for val in params.single_dirs:
        print(val)
    print(" ======================================")
    print(f" **** Single Recommendation task on {params.domain}; Train a new model? {params.train}")
    print(" =============== running ============== ")
    main(params, print_p=True, train=params.train)

"""
business data: 
number of users: 1166552
number of items: [100000, 100000, 50000, 50000, 50000]

amazon data: 
number of users: 18347 
number of items: ['Books', 'Electronics', 'Movies', 'Sports', 'Video']
                 [274552 ,  94657, 41896, 76172, 24649]

main_cross.py --num_run 1 --domain 0 --epoch 50 --bar_dis True --n_worker 6 --train True --cross "base1"
"""
