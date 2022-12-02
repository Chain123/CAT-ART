# -*- coding: utf-8 -*-

"""
Run cross-domain recommendation for
base1, base2 and base3 methods
"""
import argparse
import os
import shutil
import sys

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
    model_name = f"Cross-{h_args.cross}_{h_args.knowledge}-U{h_args.fix_user}-I{h_args.fix_item}" \
                 f"-EL{h_args.encoder_loss}-r{h_args.num_run}"
    out_path = f"{h_args.result_dir}/{model_name}/checkpoint/best_model.pt"
    print(out_path)
    # if train and os.path.isdir(f"{h_args.result_dir}/{model_name}"):
    #     shutil.rmtree(f"{h_args.result_dir}/{model_name}")
    if not os.path.isdir(f"{h_args.result_dir}/{model_name}"):
        os.mkdir(f"{h_args.result_dir}/{model_name}")
        os.mkdir(f"{h_args.result_dir}/{model_name}/checkpoint")
    # data set
    if h_args.n_users < 20000:
        print("loading amazon dataset")  # TODO: change accordingly.
        train_loader, valid_loader, test_loader = Data_loader.single_domain_loader_ama(h_args.domain, train)
    else:
        train_loader, valid_loader, test_loader = Data_loader.single_domain_loader(h_args.domain,
                                                                                   train,
                                                                                   mask=h_args.mask)
    # build model
    model_config = {'target_domain': h_args.domain,
                    'single_dirs': h_args.single_dirs,
                    'latent_dim': h_args.latent_dim,
                    'device': device,
                    'fix_user': h_args.fix_user,
                    'fix_item': h_args.fix_item,
                    'final_embed': h_args.knowledge}   # or others

    # print(model_config)
    model = h_args.cross_models[h_args.cross](model_config)
    if print_p:  # show model parameters
        for name, param in model.named_parameters():
            print(name, param.requires_grad, param.data.size())
    model = model.to(device)
    # train single domain prediction model

    # Reload the best model parameters and testing
    print(f"loading model from {out_path}")
    model = h_args.cross_models[h_args.cross](model_config)
    model = model.to(device)
    model_dict = model.state_dict()
    pre_dict = torch.load(out_path)
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)
    if torch.cuda.device_count() > 1:
        print(f" *** data parallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    attn_scores = run_func.test_attn_score(model, test_loader, device, n_process=24)
    print(" **** Test result")
    print(attn_scores)
    # save results
    with open(f"{h_args.result_dir}/{model_name}/attn_score.pickle", 'wb') as fid:
        pickle.dump(attn_scores, fid, -1)


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
    parser.add_argument('--cross', type=str, default='base4',
                        help='cross method')
    parser.add_argument('--fix_item', type=bool, default=False,
                        help='whether to fix item embeddings in the target domain')
    parser.add_argument('--fix_user', type=bool, default=False,
                        help='whether to fix user embeddings in the target domain')
    parser.add_argument('--mask', type=str, default='rand',
                        help='random mask domain for high-level embedding extraction')
    parser.add_argument('--knowledge', type=str, default='d_in',
                        help='what to transfer: Domain-independent representation(d_in), '
                             'domain-specific (d_sp) or both')
    parser.add_argument('--encoder_loss', type=str, default='full',
                        help='loss to train the encoder, full or rec ')

    params = parser.parse_args()
    params.n_items = [100000, 100000, 50000, 50000, 50000]
    if params.amazon:
        params.n_users = 18347
        params.n_items = [274552, 94657, 41896, 76172, 24649]
        params.result_dir = "/data/ceph/seqrec/UMMD/www/amazon"
    params.result_dir += f"/domain_{params.domain}"

    cross_models = {
        "base4": models.MultiRecommendBase4,
    }
    assert params.cross in cross_models
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

main_cross_pre.py --num_run 11 --domain 0 --epoch 50 --bar_dis True --n_worker 6 --train True --cross "base4"
"""
