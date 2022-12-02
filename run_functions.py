import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
from multiprocessing import Pool
import loss_functions as eval_metric
import utils

"""
                    === Single domain functions ===
"""

cross_entropy_func = nn.CrossEntropyLoss()
mult_label_loss = nn.BCEWithLogitsLoss()


def train_single_step(model, param, writer, mpf=False, cross=False):
    """
    run_config = {"train_loader": train_batch,
                  "valid_loader": valid_batch,
                  "opt": opt,
                  "device": device,
                  "early_stop": early_stop,
                  "epoch": h_args.epoch,
                  "bar_dis": h_args.bar_dis,}
    """
    if mpf:
        data_keys = ["uid", "pos", "neg", "pos_g", "neg_g"]
    else:
        data_keys = ["uid", "pos", "neg"]
    model.train()
    for epoch_n in range(param["epoch"] * param["train_r"]):
        train_loss = []
        pbar = tqdm(enumerate(param["train_loader"]), disable=param['bar_dis'])
        pbar.set_description(f'[Train epoch {epoch_n}]')
        for ind, data_batch in pbar:
            for key in data_keys:
                data_batch[key] = data_batch[key].to(param["device"])
            model.zero_grad(set_to_none=True)
            if mpf:
                loss, reg_loss = model(data_batch['uid'], data_batch['pos'], data_batch['neg'],
                                       data_batch['pos_g'], data_batch['neg_g'])
            else:
                loss, reg_loss = model(data_batch['uid'], data_batch['pos'], data_batch['neg'])

            if torch.cuda.device_count() > 1:
                loss, reg_loss = loss.mean(), reg_loss.mean()
            reg_loss = reg_loss * param['decay']
            if not cross:
                loss = loss + reg_loss
            # backward
            loss.backward()
            param["opt"].step()
            pbar.set_postfix(loss=loss.item(), reg_loss=reg_loss.item())
            train_loss.append(loss.item())

        if (epoch_n + 1) % param["train_r"] == 0:
            print(" *** Evaluation ...")
            loss_v = valid_single_step(model, param['valid_loader'], param["device"],
                                       param['bar_dis'], eval_round=param['valid_r'], mpf=mpf)
            param["early_stop"](loss_v, model)
            if param["early_stop"].early_stop:
                print(" ****** Early stopping ...")
                return 1
            writer.add_scalar("train loss", np.mean(train_loss), int((epoch_n + 1) / param["train_r"]))
            writer.add_scalar("valid loss", loss_v.item(), int((epoch_n + 1) / param["train_r"]))
    return 0


def valid_single_step(model, data_loader, device_t, bar_dis=True, eval_round=1, mpf=False):
    if mpf:
        data_keys = ["uid", "pos", "neg", "pos_g", "neg_g"]
    else:
        data_keys = ["uid", "pos", "neg"]
    model.eval()
    valid_loss = []
    with torch.no_grad():
        for epoch_n in range(eval_round):
            pbar = tqdm(enumerate(data_loader), disable=bar_dis)
            pbar.set_description(f'[Eval round {epoch_n}]')
            for ind, data_batch in pbar:
                for key in data_keys:
                    data_batch[key] = data_batch[key].to(device_t)
                if mpf:
                    loss, reg_loss = model(data_batch['uid'], data_batch['pos'], data_batch['neg'],
                                           data_batch['pos_g'], data_batch['neg_g'])
                else:
                    loss, reg_loss = model(data_batch['uid'], data_batch['pos'], data_batch['neg'])

                if torch.cuda.device_count() > 1:
                    loss, reg_loss = loss.mean(), reg_loss.mean()
                pbar.set_postfix(loss=loss.item(), reg_loss=reg_loss.item())
                valid_loss.append(loss.item())
    return np.mean(valid_loss)


def train_single_step_pre(model, param, writer, l='full'):
    """
    run_config = {"train_loader": train_batch,
                  "valid_loader": valid_batch,
                  "opt": opt,
                  "device": device,
                  "early_stop": early_stop,
                  "epoch": h_args.epoch,
                  "bar_dis": h_args.bar_dis,}
    """
    data_keys = ["uid", 'mask']
    model.train()
    for epoch_n in range(param["epoch"] * param["train_r"]):
        train_loss = []
        pbar = tqdm(enumerate(param["train_loader"]), disable=param['bar_dis'])
        pbar.set_description(f'[Train epoch {epoch_n}]')
        for ind, data_batch in pbar:
            for key in data_keys:
                data_batch[key] = data_batch[key].to(param["device"])
            model.zero_grad(set_to_none=True)
            contrastive_loss, loss_rec1, loss_rec2 = model.GeneralForward(data_batch['uid'], data_batch['mask'])
            if l == 'full':
                # full loss
                loss = contrastive_loss * 0.2 + loss_rec1 * 0.4 + loss_rec2 * 0.4
            else:
                # only with reconstruction
                loss = loss_rec1
            # backward
            loss.backward()
            param["opt"].step()
            pbar.set_postfix(loss=loss.item(), contrastive_loss=contrastive_loss.item(),
                             rec_l=loss_rec1.item(), rec_l2=loss_rec2.item())
            train_loss.append(loss.item())

        if (epoch_n + 1) % param["train_r"] == 0:
            print(" *** Evaluation ...")
            loss_v = valid_single_step_pre(model, param['valid_loader'], param["device"],
                                           param['bar_dis'], eval_round=param['valid_r'], l=l)
            param["early_stop"](loss_v, model)
            if param["early_stop"].early_stop:
                print(" ****** Early stopping ...")
                return 1
            writer.add_scalar("train loss", np.mean(train_loss), int((epoch_n + 1) / param["train_r"]))
            writer.add_scalar("valid loss", loss_v.item(), int((epoch_n + 1) / param["train_r"]))
    return 0


def valid_single_step_pre(model, data_loader, device_t, bar_dis=True, eval_round=1, l='full'):
    data_keys = ["uid", "mask"]
    model.eval()
    valid_loss = []
    with torch.no_grad():
        for epoch_n in range(eval_round):
            pbar = tqdm(enumerate(data_loader), disable=bar_dis)
            pbar.set_description(f'[Eval round {epoch_n}]')
            for ind, data_batch in pbar:
                for key in data_keys:
                    data_batch[key] = data_batch[key].to(device_t)
                contrastive_loss, loss_rec1, loss_rec2 = model.GeneralForward(data_batch['uid'], data_batch['mask'])
                if l == "full":
                    loss = contrastive_loss * 0.2 + loss_rec1 * 0.4 + loss_rec2 * 0.4
                else:
                    loss = loss_rec1
                pbar.set_postfix(loss=loss.item(), contrastive_loss=contrastive_loss.item(),
                                 rec_l=loss_rec1.item(), rec_l2=loss_rec2.item())
                valid_loss.append(loss.item())
    return np.mean(valid_loss)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size')
    tensors = tensors[0]
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple([x[i:i + batch_size] for x in tensors])


def test_single_step(model, data_loader, device_t,
                     top_k=None, n_process=0, num_items=0):
    if top_k is None:
        top_k = [5, 10, 20, 50]
    model.eval()
    result = {"precision": np.zeros(len(top_k)),
              "recall": np.zeros(len(top_k)),
              "ndcg": np.zeros(len(top_k))}
    user_list = []
    rating_list = []
    groundTrue_list = []
    if n_process > 0:
        pool = Pool(n_process)
    n_users = 0
    max_k = np.max(top_k)
    n_user = len(data_loader['user'])
    # test_loader = utils.AmazonSingleRecTest_bus(data_loader, num_items)
    # test_batch = DataLoader(test_loader, batch_size=4096, num_workers=6)
    # pbar = tqdm(enumerate(test_batch))
    # pbar.set_description(f'[Testing]')
    # with torch.no_grad():
    #     for ind, data_batch in pbar:
    #         user_pos_test, user_pos_train = test_loader.getUserPosItems(data_batch["uid"])
    #         data_batch["uid"] = data_batch["uid"].to(device_t)
    #
    #         n_users += data_batch["uid"].size(0)
    #         rating = model.getUsersRating(data_batch["uid"])
    #
    #         exclude_index = []
    #         exclude_items = []
    #         for range_i, items in enumerate(user_pos_train):
    #             exclude_index.extend([range_i] * len(items))
    #             exclude_items.extend(items)
    #         rating[exclude_index, exclude_items] = -(1 << 10)
    #         # get top k
    #         _, score_k = torch.topk(rating, k=max_k)  # [B, max_k]
    #         rating_list.append(score_k.cpu())
    #         groundTrue_list.append(user_pos_test)
    if torch.cuda.device_count() > 1:
        multi = True
    else:
        multi = False
    batch_size = 4096
    n_batch = n_user // batch_size + 1
    All_data = [data_loader['user'], data_loader['truth'], data_loader['train_pos']]
    with torch.no_grad():
        for batch_data in minibatch(All_data, batch_size=batch_size):
            batch_user = torch.Tensor(batch_data[0]).long().to(device_t)
            if multi:
                rating = model.Module.getUsersRating(batch_user)
            else:
                rating = model.getUsersRating(batch_user)
            exclude_index = []
            exclude_item = []
            for i, items in enumerate(batch_data[2]):
                exclude_index.extend([i] * len(items))
                exclude_item.extend(items)
            rating[exclude_index, exclude_item] = -(1 << 10)
            _, rating_k = torch.topk(rating, k=max_k)
            user_list.append(batch_data[0])
            rating_list.append(rating_k.cpu())
            groundTrue_list.append(batch_data[1])
            n_users += len(batch_data[0])
    X = zip(rating_list, groundTrue_list)
    if n_process > 0:
        pre_results = pool.map(eval_metric.metrics_batch, X)
        pool.close()
        pool.join()
    else:
        pre_results = []
        for x in X:
            pre_results.append(eval_metric.metrics_batch(x))
    # collect result from all batches
    for batch_result in pre_results:
        result['precision'] += batch_result["precision"]
        result['recall'] += batch_result["recall"]
        result['ndcg'] += batch_result["ndcg"]
    result['precision'] /= float(n_users)
    result['recall'] /= float(n_users)
    result['ndcg'] /= float(n_users)

    return result


def test_single_step_cmf(model, data_loader, device_t,
                         top_k=None, n_process=0, domain=0):
    if top_k is None:
        top_k = [5, 10, 20, 50]
    model.eval()
    result = {"precision": np.zeros(len(top_k)),
              "recall": np.zeros(len(top_k)),
              "ndcg": np.zeros(len(top_k))}
    user_list = []
    rating_list = []
    groundTrue_list = []
    if n_process > 0:
        pool = Pool(n_process)
    n_users = 0
    max_k = np.max(top_k)
    n_user = len(data_loader['user'])
    if torch.cuda.device_count() > 1:
        multi = True
    else:
        multi = False
    batch_size = 8192
    All_data = [data_loader['user'], data_loader['truth'], data_loader['train_pos']]

    # item_range
    item_range_dict = {
        0: [1, 100000 + 1],  # [1 to 100000]
        1: [100000 + 1, 200000 + 1],
        2: [200000 + 1, 250000 + 1],
        3: [250000 + 1, 300000 + 1],
        4: [300000 + 1, 350000 + 1],
    }
    item_id = torch.from_numpy(np.arange(item_range_dict[domain][0],
                                         item_range_dict[domain][1])).long().to(device_t)
    with torch.no_grad():
        for batch_data in minibatch(All_data, batch_size=batch_size):
            batch_user = torch.Tensor(batch_data[0]).long().to(device_t)
            if multi:
                rating = model.Module.getUsersRating_cmf(batch_user, item_id)
            else:
                rating = model.getUsersRating_cmf(batch_user, item_id)
            exclude_index = []
            exclude_item = []
            for i, items in enumerate(batch_data[2]):
                exclude_index.extend([i] * len(items))
                exclude_item.extend(items)
            rating[exclude_index, exclude_item] = -(1 << 10)
            _, rating_k = torch.topk(rating, k=max_k)
            user_list.append(batch_data[0])
            rating_list.append(rating_k.cpu())
            groundTrue_list.append(batch_data[1])
            n_users += len(batch_data[0])
    X = zip(rating_list, groundTrue_list)
    if n_process > 0:
        pre_results = pool.map(eval_metric.metrics_batch, X)
        pool.close()
        pool.join()
    else:
        pre_results = []
        for x in X:
            pre_results.append(eval_metric.metrics_batch(x))
    # collect result from all batches
    for batch_result in pre_results:
        result['precision'] += batch_result["precision"]
        result['recall'] += batch_result["recall"]
        result['ndcg'] += batch_result["ndcg"]
    result['precision'] /= float(n_users)
    result['recall'] /= float(n_users)
    result['ndcg'] /= float(n_users)

    return result


"""
            === Multi-domains function ===
            main_multi_base.py
            main_multi_pro.py 
"""


def train_mul_base(model, param, writer, recon=None, mul_ratio=0.15):
    """
    recon: whether there is a reconstruction loss from model.forward()
        None: no reconstruction
        'pre': this is a pre-training procedure of the model.generator, only rec_loss is used for back propagation
        'ce': only cross entropy is used: end-end fine-tune
        "mul": ce + rec_loss for multi-task manner fine-tuning
    """
    domain_list = np.arange(5).reshape((1, 5))
    data_keys = ["tag_set", 'mask', 'age', 'gender']
    model.train()
    step = 0
    train_loss = []
    if torch.cuda.device_count() > 1:
        latent_dim = model.module.latent_dim
        zeros_tensor = torch.zeros(torch.cuda.device_count(),
                                   latent_dim).to(param['device'])
        domain_tensor = torch.from_numpy(np.repeat(domain_list,
                                                   torch.cuda.device_count(),
                                                   axis=0)).to(param['device'])  # multi-gpus
        label_name = model.module.label_name
    else:
        latent_dim = model.latent_dim
        label_name = model.label_name
        zeros_tensor = torch.zeros(1, latent_dim).to(param['device'])
        domain_tensor = torch.from_numpy(domain_list[0]).to(param['device'])  # single-gpus

    for epoch_n in range(param["epoch"]):
        pbar = tqdm(enumerate(param["train_loader"]), disable=param['bar_dis'])
        pbar.set_description(f'[Train epoch {epoch_n}]')
        for ind, data_batch in pbar:
            for key in data_keys:
                if key == "tag_set":
                    for i in range(5):
                        data_batch[key][i] = data_batch[key][i].to(param["device"])
                else:
                    data_batch[key] = data_batch[key].to(param["device"])
            # forward
            model.zero_grad(set_to_none=True)
            if recon is None:
                logits = model(data_batch["tag_set"],
                               data_batch["mask"],
                               zero=zeros_tensor,
                               domain_tensor=domain_tensor)
                rec_loss = 0
            else:
                logits, rec_loss = model(data_batch["tag_set"],
                                         data_batch["mask"],
                                         zero=zeros_tensor,
                                         domain_tensor=domain_tensor)
            if label_name == "gender":
                loss = cross_entropy_func(logits, data_batch[label_name])
                pred = logits.argmax(axis=1)
                acc = torch.sum(pred == data_batch[label_name]) / data_batch[label_name].size(0)
            else:
                loss = mult_label_loss(logits, data_batch[label_name])
                pred = logits.argmax(axis=1)
                correct = torch.sum(data_batch[label_name][range(pred.size(0)), pred] == 1)
                acc = correct / data_batch[label_name].size(0)
            # backward
            if recon == "pre":
                rec_loss = rec_loss.mean()  # for multi-gpus case
                rec_loss.backward()
            elif recon == "ce":
                loss.backward()
            elif recon == "mul":
                rec_loss = rec_loss.mean()
                total_loss = loss + rec_loss * mul_ratio  # TODO to be tuned
                total_loss.backward()
            else:  # is None
                loss.backward()

            param["opt"].step()
            if recon is None or recon == "ce":
                pbar.set_postfix(CE_loss=loss.item(), train_acc=acc.item())
            else:
                pbar.set_postfix(CE_loss=loss.item(), train_acc=acc.item(), recons_loss=rec_loss.item())
            step += 1
            train_loss.append(loss.item())
        print(" *** Evaluation ...")
        loss_v, acc_v = valid_mul_base(model, param['valid_loader'],
                                       bar_dis=param['bar_dis'],
                                       param=param,
                                       recon=recon)
        param["early_stop"](loss_v, model)
        if param["scheduler"] is not None:
            param["scheduler"].step()
        if param["early_stop"].early_stop:
            print("Early stopping")
            break
        writer.add_scalar("train loss", np.mean(train_loss), epoch_n)
        writer.add_scalar("valid loss", loss_v.item(), epoch_n)
        writer.add_scalar("valid acc", acc_v.item(), epoch_n)
        writer.add_scalar("valid acc", acc_v.item(), epoch_n)


def valid_mul_base(model, data_loader, bar_dis, param, recon=None):
    data_keys = ["tag_set", 'mask', 'age', 'gender']
    domain_list = np.arange(5).reshape((1, 5))
    model.eval()
    pbar = tqdm(enumerate(data_loader), disable=bar_dis)
    pbar.set_description(f'[eval]')
    all_loss = []
    acc_all = []
    if torch.cuda.device_count() > 1:
        latent_dim = model.module.latent_dim
        zeros_tensor = torch.zeros(torch.cuda.device_count(),
                                   latent_dim).to(param['device'])
        domain_tensor = torch.from_numpy(np.repeat(domain_list,
                                                   torch.cuda.device_count(),
                                                   axis=0)).to(param['device'])  # multi-gpus
        label_name = model.module.label_name
    else:
        latent_dim = model.latent_dim
        label_name = model.label_name
        zeros_tensor = torch.zeros(1, latent_dim).to(param['device'])
        domain_tensor = torch.from_numpy(domain_list[0]).to(param['device'])  # single-gpus

    with torch.no_grad():
        for ind, data_batch in pbar:
            for key in data_keys:
                if key == "tag_set":
                    for i in range(5):
                        data_batch[key][i] = data_batch[key][i].to(param['device'])
                else:
                    data_batch[key] = data_batch[key].to(param['device'])
            # forward
            if recon is None:
                logits = model(data_batch["tag_set"],
                               data_batch["mask"],
                               zero=zeros_tensor,
                               domain_tensor=domain_tensor)
                rec_loss = 0
            else:
                logits, rec_loss = model(data_batch["tag_set"],
                                         data_batch["mask"],
                                         zero=zeros_tensor,
                                         domain_tensor=domain_tensor)
            if label_name == "gender":
                loss = cross_entropy_func(logits, data_batch[label_name])
                pred = logits.argmax(axis=1)
                acc = torch.sum(pred == data_batch[label_name]) / data_batch[label_name].size(0)
            else:
                loss = mult_label_loss(logits, data_batch[label_name])
                pred = logits.argmax(axis=1)
                correct = torch.sum(data_batch[label_name][range(pred.size(0)), pred] == 1)
                acc = correct / data_batch[label_name].size(0)

            acc_all.append(acc.item())
            # record
            if recon == "pre":
                rec_loss = rec_loss.mean()  # for multi-gpus case
                all_loss.append(rec_loss.item())
            elif recon == "ce":
                all_loss.append(loss.item())
            elif recon == "mul":
                rec_loss = rec_loss.mean()
                total_loss = loss + rec_loss * 0.15  # TODO to be tuned
                all_loss.append(total_loss.item())  # TODO: use loss only?
            else:
                all_loss.append(loss.item())
    model.train()
    return np.mean(all_loss), np.mean(acc_all)


def test_mul_base(model, data_loader, device, bar_dis, recon=None):
    """
    data_loader:
    device:
    top_k (list of int):
    """
    data_keys = ["tag_set", 'mask', 'age', 'gender']
    domain_list = np.arange(5).reshape((1, 5))
    model.eval()
    pbar = tqdm(enumerate(data_loader), disable=bar_dis)
    pbar.set_description(f'[test]')
    acc_all = []

    if torch.cuda.device_count() > 1:
        latent_dim = model.module.latent_dim
        zeros_tensor = torch.zeros(torch.cuda.device_count(),
                                   latent_dim).to(device)
        domain_tensor = torch.from_numpy(np.repeat(domain_list,
                                                   torch.cuda.device_count(),
                                                   axis=0)).to(device)  # multi-gpus
        label_name = model.module.label_name
    else:
        latent_dim = model.latent_dim
        label_name = model.label_name
        zeros_tensor = torch.zeros(1, latent_dim).to(device)
        domain_tensor = torch.from_numpy(domain_list[0]).to(device)  # single-gpus

    with torch.no_grad():
        for ind, data_batch in pbar:
            for key in data_keys:
                if key == "tag_set":
                    for i in range(5):
                        data_batch[key][i] = data_batch[key][i].to(device)
                else:
                    data_batch[key] = data_batch[key].to(device)
            # TODO BUG on multiple GPUs.
            if recon is not None:
                logits, rec_loss = model(data_batch["tag_set"],
                                         data_batch["mask"],
                                         zero=zeros_tensor,
                                         domain_tensor=domain_tensor)
            else:
                logits = model(data_batch["tag_set"],
                               data_batch["mask"],
                               zero=zeros_tensor,
                               domain_tensor=domain_tensor)
            pred = logits.argmax(axis=1)
            if label_name == "gender":
                acc = torch.sum(pred == data_batch[label_name]) / data_batch[label_name].size(0)
            else:
                correct = torch.sum(data_batch[label_name][range(pred.size(0)), pred] == 1)
                acc = correct / data_batch[label_name].size(0)
            acc_all.append(acc.item())
    return np.mean(acc_all)


"""
            +++++++ Multi-domains VQ ++++++++
            main_multi_pro_vq.py
"""


def train_mul_vq(model, param, writer, recon='pre'):
    """
    recon: whether there is a reconstruction loss from model.forward()
        'pre': this is a pre-training procedure of the model.generator, only rec_loss is used for back propagation
        'ce': only cross entropy is used: end-end fine-tune
        "mul": ce + rec_loss for multi-task manner fine-tuning
    """
    domain_list = np.arange(5).reshape((1, 5))
    data_keys = ["tag_set", 'mask', 'age', 'gender']
    model.train()
    step = 0
    train_loss = []
    if torch.cuda.device_count() > 1:
        latent_dim = model.module.latent_dim
        zeros_tensor = torch.zeros(torch.cuda.device_count(),
                                   latent_dim).to(param['device'])
        domain_tensor = torch.from_numpy(np.repeat(domain_list,
                                                   torch.cuda.device_count(),
                                                   axis=0)).to(param['device'])  # multi-gpus
        label_name = model.module.label_name
        vq_method = model.module.vq_method
    else:
        latent_dim = model.latent_dim
        label_name = model.label_name
        zeros_tensor = torch.zeros(1, latent_dim).to(param['device'])
        domain_tensor = torch.from_numpy(domain_list[0]).to(param['device'])  # single-gpus
        vq_method = model.vq_method
    for epoch_n in range(param["epoch"]):
        pbar = tqdm(enumerate(param["train_loader"]), disable=param['bar_dis'])
        pbar.set_description(f'[Train epoch {epoch_n}]')
        for ind, data_batch in pbar:
            for key in data_keys:
                if key == "tag_set":
                    for i in range(5):
                        data_batch[key][i] = data_batch[key][i].to(param["device"])
                else:
                    data_batch[key] = data_batch[key].to(param["device"])
            model.zero_grad(set_to_none=True)
            # forward
            if vq_method == "vq_l2":
                logits, rec_loss, diff = model(data_batch["tag_set"],
                                               data_batch["mask"],
                                               zero=zeros_tensor,
                                               domain_tensor=domain_tensor)
                id_acc = 0
            else:
                logits, rec_loss, diff, id_acc = model(data_batch["tag_set"],
                                                       data_batch["mask"],
                                                       zero=zeros_tensor,
                                                       domain_tensor=domain_tensor)
            if label_name == "gender":
                loss = cross_entropy_func(logits, data_batch[label_name])
                pred = logits.argmax(axis=1)
                acc = torch.sum(pred == data_batch[label_name]) / data_batch[label_name].size(0)
            else:
                loss = mult_label_loss(logits, data_batch[label_name])
                pred = logits.argmax(axis=1)
                correct = torch.sum(data_batch[label_name][range(pred.size(0)), pred] == 1)
                acc = correct / data_batch[label_name].size(0)
            # backward
            rec_loss = rec_loss.mean()  # for multi-gpus case
            diff = diff.mean()
            if recon == "pre":
                rec_loss.backward()
            elif recon == "ce":
                loss.backward()
            elif recon == "mul":
                total_loss = loss + rec_loss * 0.15  # TODO to be tuned
                total_loss.backward()
            else:  # otherwise
                loss.backward()
            param["opt"].step()
            if vq_method == "vq_l2":
                pbar.set_postfix(CE_loss=loss.item(), train_acc=acc.item(),
                                 rec_loss=rec_loss.item(), diff=diff.item())
            else:
                pbar.set_postfix(CE_loss=loss.item(), train_acc=acc.item(),
                                 rec_loss=rec_loss.item(), diff=diff.item(), id_acc=id_acc.mean().item())
            step += 1
            train_loss.append(loss.item())
        print(" *** Evaluation ...")
        loss_v, acc_v = valid_mul_vq(model, param['valid_loader'],
                                     bar_dis=param['bar_dis'],
                                     param=param,
                                     recon=recon)
        param["early_stop"](loss_v, model)
        if param["scheduler"] is not None:
            param["scheduler"].step()
        if param["early_stop"].early_stop:
            print("Early stopping")
            break
        writer.add_scalar("train loss", np.mean(train_loss), epoch_n)
        writer.add_scalar("valid loss", loss_v.item(), epoch_n)
        writer.add_scalar("valid acc", acc_v.item(), epoch_n)
        writer.add_scalar("valid acc", acc_v.item(), epoch_n)


def valid_mul_vq(model, data_loader, bar_dis, param, recon=None):
    data_keys = ["tag_set", 'mask', 'age', 'gender']
    domain_list = np.arange(5).reshape((1, 5))
    model.eval()
    pbar = tqdm(enumerate(data_loader), disable=bar_dis)
    pbar.set_description(f'[eval]')
    all_loss = []
    acc_all = []
    if torch.cuda.device_count() > 1:
        latent_dim = model.module.latent_dim
        zeros_tensor = torch.zeros(torch.cuda.device_count(),
                                   latent_dim).to(param['device'])
        domain_tensor = torch.from_numpy(np.repeat(domain_list,
                                                   torch.cuda.device_count(),
                                                   axis=0)).to(param['device'])  # multi-gpus
        label_name = model.module.label_name
        vq_method = model.module.vq_method
    else:
        latent_dim = model.latent_dim
        label_name = model.label_name
        zeros_tensor = torch.zeros(1, latent_dim).to(param['device'])
        domain_tensor = torch.from_numpy(domain_list[0]).to(param['device'])  # single-gpus
        vq_method = model.vq_method
    with torch.no_grad():
        for ind, data_batch in pbar:
            for key in data_keys:
                if key == "tag_set":
                    for i in range(5):
                        data_batch[key][i] = data_batch[key][i].to(param['device'])
                else:
                    data_batch[key] = data_batch[key].to(param['device'])
            # forward
            if vq_method == "vq_l2":
                logits, rec_loss, diff = model(data_batch["tag_set"],
                                               data_batch["mask"],
                                               zero=zeros_tensor,
                                               domain_tensor=domain_tensor)
                # id_acc = 0
            else:
                logits, rec_loss, diff, id_acc = model(data_batch["tag_set"],
                                                       data_batch["mask"],
                                                       zero=zeros_tensor,
                                                       domain_tensor=domain_tensor)
            if label_name == "gender":
                loss = cross_entropy_func(logits, data_batch[label_name])
                pred = logits.argmax(axis=1)
                acc = torch.sum(pred == data_batch[label_name]) / data_batch[label_name].size(0)
            else:
                loss = mult_label_loss(logits, data_batch[label_name])
                pred = logits.argmax(axis=1)
                correct = torch.sum(data_batch[label_name][range(pred.size(0)), pred] == 1)
                acc = correct / data_batch[label_name].size(0)
            acc_all.append(acc.item())
            # record
            rec_loss = rec_loss.mean()  # for multi-gpus case
            if recon == "pre":
                all_loss.append(rec_loss.item())
            elif recon == "ce":
                all_loss.append(loss.item())
            elif recon == "mul":
                total_loss = loss + rec_loss * 0.15  # TODO to be tuned
                all_loss.append(total_loss.item())  # TODO: use loss only?
            else:
                all_loss.append(loss.item())
    model.train()
    return np.mean(all_loss), np.mean(acc_all)


def test_mul_vq(model, data_loader, device, bar_dis, vq_method="vq_l2"):
    """
    data_loader:
    device:
    top_k (list of int):
    """
    data_keys = ["tag_set", 'mask', 'age', 'gender']
    domain_list = np.arange(5).reshape((1, 5))
    model.eval()
    pbar = tqdm(enumerate(data_loader), disable=bar_dis)
    pbar.set_description(f'[test]')
    acc_all = []

    if torch.cuda.device_count() > 1:
        latent_dim = model.module.latent_dim
        zeros_tensor = torch.zeros(torch.cuda.device_count(),
                                   latent_dim).to(device)
        domain_tensor = torch.from_numpy(np.repeat(domain_list,
                                                   torch.cuda.device_count(),
                                                   axis=0)).to(device)  # multi-gpus
        label_name = model.module.label_name
        # vq_method = model.module.vq_method
    else:
        latent_dim = model.latent_dim
        label_name = model.label_name
        zeros_tensor = torch.zeros(1, latent_dim).to(device)
        domain_tensor = torch.from_numpy(domain_list[0]).to(device)  # single-gpus
        # vq_method = model.vq_method
    with torch.no_grad():
        for ind, data_batch in pbar:
            for key in data_keys:
                if key == "tag_set":
                    for i in range(5):
                        data_batch[key][i] = data_batch[key][i].to(device)
                else:
                    data_batch[key] = data_batch[key].to(device)
            if vq_method == "vq_l2":
                logits, rec_loss, diff = model(data_batch["tag_set"],
                                               data_batch["mask"],
                                               zero=zeros_tensor,
                                               domain_tensor=domain_tensor)
            else:
                logits, rec_loss, diff, id_acc = model(data_batch["tag_set"],
                                                       data_batch["mask"],
                                                       zero=zeros_tensor,
                                                       domain_tensor=domain_tensor)
            pred = logits.argmax(axis=1)
            if label_name == "gender":
                acc = torch.sum(pred == data_batch[label_name]) / data_batch[label_name].size(0)
            else:
                correct = torch.sum(data_batch[label_name][range(pred.size(0)), pred] == 1)
                acc = correct / data_batch[label_name].size(0)
            acc_all.append(acc.item())
    return np.mean(acc_all)
