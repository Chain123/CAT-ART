# -*- coding: utf-8 -*-

"""
Run single domain embedding
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
from models import PureMF
import _pickle as pickle


def main(h_args, print_p=True, train=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = f"MF-d{h_args.latent_dim}-r{h_args.num_run}"
    out_path = f"{h_args.result_dir}/{run_name}/checkpoint/best_model.pt"
    # if train and os.path.isdir(f"{h_args.result_dir}/{run_name}"):
    #     shutil.rmtree(f"{h_args.result_dir}/{run_name}")
    if not os.path.isdir(f"{h_args.result_dir}"):
        os.mkdir(f"{h_args.result_dir}")
    if not os.path.isdir(f"{h_args.result_dir}/{run_name}"):
        os.mkdir(f"{h_args.result_dir}/{run_name}")
        os.mkdir(f"{h_args.result_dir}/{run_name}/checkpoint")

    # data set
    if h_args.n_users < 20000:
        print("loading amazon dataset")
        train_loader, valid_loader, test_loader = Data_loader.single_domain_loader_ama(h_args.domain, train)
    else:
        train_loader, valid_loader, test_loader = Data_loader.single_domain_loader(h_args.domain, train)
    # build model
    model_config = {'n_users': h_args.n_users,
                    'n_items': h_args.n_items[h_args.domain],
                    'latent_dim': h_args.latent_dim}
    print(model_config)
    model = PureMF(model_config)
    model = model.to(device)
    # if exits, reload and continue training.
    if os.path.isfile(out_path):
        print("Continue training ....")
        model.load_state_dict(torch.load(out_path))
    if print_p:  # show model parameters
        for name, param in model.named_parameters():
            print(name, param.requires_grad, param.data.size())
    # train single domain prediction model
    if train:
        writer = SummaryWriter(f"{h_args.result_dir}/{run_name}")
        early_stop = utils.EarlyStopping(patience=3, verbose=True, path=out_path)
        opt = optim.Adam(model.parameters(), lr=h_args.lr, weight_decay=0)
        train_batch = DataLoader(train_loader, batch_size=h_args.batch_size,
                                 num_workers=h_args.n_worker, pin_memory=True, shuffle=True)
        valid_batch = DataLoader(valid_loader, batch_size=h_args.batch_size,
                                 num_workers=h_args.n_worker, pin_memory=True)
        round_train = max(int(train_loader.n_interaction / len(train_loader.user)), 1)
        round_valid = max(int(valid_loader.n_interaction / len(valid_loader.user)), 1)
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
        run_func.train_single_step(model, run_config, writer)
    # else:
    #     pass
    # Reload the best model parameters and testing
    print(f"loading model from {out_path}")
    model.load_state_dict(torch.load(out_path, map_location=device))
    # Testing data
    test_result = run_func.test_single_step(model, test_loader, device,
                                            n_process=24, num_items=model_config['n_items'])
    print(" **** Test result")
    print(test_result)
    # save results
    with open(f"{h_args.result_dir}/{run_name}/result.pickle", 'wb') as fid:
        pickle.dump(test_result, fid, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Unsupervised Recommendation Training')
    # Path Arguments  gender_method
    parser.add_argument('--batch_size', type=int, default=512,
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
    params = parser.parse_args()
    params.n_items = [100000, 100000, 50000, 50000, 50000]
    if params.amazon:
        params.n_users = 18347
        params.n_items = [274552, 94657, 41896, 76172, 24649]
        params.result_dir = "/data/ceph/seqrec/UMMD/www/amazon"

    if params.domain < 0:
        params.result_dir += "/all_domain"
    else:
        params.result_dir += f"/domain_{params.domain}"
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
"""
