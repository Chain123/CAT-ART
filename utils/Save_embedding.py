import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import utils


def get_embedding(data_loader, model, device, name, bpr):
    model.eval()
    data_keys = ["tag_set_0", "tag_set_1", "tag_set_2", "tag_set_3",
                 "tag_set_4", "label", "mask", 'pos_0', 'pos_1', 'pos_2', 'pos_3', 'pos_4',
                 "neg_0", "neg_1", "neg_2", "neg_3", "neg_4"]
    num_user = 0
    train_embeddings = []
    train_mask = []
    train_labels = []
    train_acc = []
    with torch.no_grad():
        for step, data_batch in data_loader:
            for key in data_keys:
                data_batch[key] = data_batch[key].to(device)
            data_all = [data_batch['tag_set_0'], data_batch['tag_set_1'], data_batch['tag_set_2'],
                        data_batch['tag_set_3'], data_batch['tag_set_4']]
            if bpr is not None:
                pos = [data_batch["pos_0"], data_batch["pos_1"], data_batch["pos_2"], data_batch["pos_3"],
                       data_batch["pos_4"]]
                neg = [data_batch["neg_0"], data_batch["neg_1"], data_batch["neg_2"], data_batch["neg_3"],
                       data_batch["neg_4"]]
                if bpr == "bpr":
                    _, embedding = model.get_embeddings(data_all, pos, neg, data_batch['mask'].to(torch.bool))
                else:   # VQ
                    _, embedding, _ = model.get_embeddings(data_all, pos, neg, data_batch['mask'].to(torch.bool))
            else:
                embedding = model.get_embeddings(data_all, data_batch['mask'].to(torch.bool))

            logits = model.layers(embedding)
            acc = utils.eval_metrics.Accuracy(logits, data_batch["label"].type_as(logits))
            train_acc.append(acc.item())

            train_mask.append(data_batch['mask'].cpu().numpy())
            train_embeddings.append(embedding.detach().cpu().numpy())
            train_labels.append(data_batch['label'].cpu().numpy())
            num_user += data_batch['mask'].size()[0]
            # if num_user > 2000000:
            #     break
    print(f"Training acc {np.mean(train_acc)}, number of users: {num_user}")

    # np.save(name + "_embed.npy", np.concatenate(train_embeddings, axis=0))
    # np.save(name + "_mask.npy", np.concatenate(train_mask, axis=0))
    # np.save(name + "_label.npy", np.concatenate(train_labels, axis=0))
    # save zipped
    np.savez_compressed(name + "_embed", embed=np.concatenate(train_embeddings, axis=0))
    np.savez_compressed(name + "_mask", mask=np.concatenate(train_mask, axis=0))
    np.savez_compressed(name + "_label", label=np.concatenate(train_labels, axis=0))


def user_embedding(model, test_data, train_loader_dense, train_loader_sparse, out_dir, device, bpr=None):
    train_loader_dense = DataLoader(train_loader_dense, batch_size=2048, num_workers=6)
    train_loader_sparse = DataLoader(train_loader_sparse, batch_size=2048, num_workers=6)
    train_dense = tqdm(enumerate(train_loader_dense))    # one tfrecord file.
    train_sparse = tqdm(enumerate(train_loader_sparse))  # one tfrecord file.
    # dense training users
    name = os.path.join(out_dir, f"train_dense")
    get_embedding(train_dense, model, device, name, bpr)
    # sparse training users
    name = os.path.join(out_dir, f"train_sparse")
    get_embedding(train_sparse, model, device, name, bpr)

    # test data
    for i in range(len(test_data)):
        test_loader = DataLoader(test_data[i], batch_size=2048, num_workers=6)
        data_iter = tqdm(enumerate(test_loader),
                         desc=f"Testing {i}",
                         bar_format="{l_bar}{r_bar}")
        name = os.path.join(out_dir, f"test_miss_{i}")
        get_embedding(data_iter, model, device, name, bpr)


def user_scores(model, test_data, train_loader_dense, train_loader_sparse, out_dir, device, bpr=None):
    train_loader_dense = DataLoader(train_loader_dense, batch_size=2048, num_workers=6)
    train_loader_sparse = DataLoader(train_loader_sparse, batch_size=2048, num_workers=6)
    train_dense = tqdm(enumerate(train_loader_dense))    # one tfrecord file.
    train_sparse = tqdm(enumerate(train_loader_sparse))  # one tfrecord file.
    # dense training users
    name = os.path.join(out_dir, f"train_dense")
    get_scores(train_dense, model, device, name, bpr)
    # sparse training users
    name = os.path.join(out_dir, f"train_sparse")
    get_scores(train_sparse, model, device, name, bpr)

    # test data
    for i in range(len(test_data)):
        test_loader = DataLoader(test_data[i], batch_size=2048, num_workers=6)
        data_iter = tqdm(enumerate(test_loader),
                         desc=f"Testing {i}",
                         bar_format="{l_bar}{r_bar}")
        name = os.path.join(out_dir, f"test_miss_{i}")
        get_scores(data_iter, model, device, name, bpr)


def get_scores(data_loader, model, device, name, bpr):
    model.eval()
    data_keys = ["tag_set_0", "tag_set_1", "tag_set_2", "tag_set_3",
                 "tag_set_4", "label", "mask", 'pos_0', 'pos_1', 'pos_2', 'pos_3', 'pos_4',
                 "neg_0", "neg_1", "neg_2", "neg_3", "neg_4"]
    num_user = 0
    train_logits = []
    user_ids = []
    train_acc = []
    train_labels = []
    with torch.no_grad():
        for step, data_batch in data_loader:
            for key in data_keys:
                data_batch[key] = data_batch[key].to(device)
            data_all = [data_batch['tag_set_0'], data_batch['tag_set_1'], data_batch['tag_set_2'],
                        data_batch['tag_set_3'], data_batch['tag_set_4']]
            if bpr is not None:
                pos = [data_batch["pos_0"], data_batch["pos_1"], data_batch["pos_2"], data_batch["pos_3"],
                       data_batch["pos_4"]]
                neg = [data_batch["neg_0"], data_batch["neg_1"], data_batch["neg_2"], data_batch["neg_3"],
                       data_batch["neg_4"]]
                if bpr == "bpr":
                    # _, embedding = model.get_embeddings(data_all, pos, neg, data_batch['mask'].to(torch.bool))
                    logits, _ = model(data_all, pos, neg, data_batch['mask'].to(torch.bool))
                else:   # VQ
                    # _, embedding, _ = model.get_embeddings(data_all, pos, neg, data_batch['mask'].to(torch.bool))
                    logits, _, _ = model(data_all, pos, neg, data_batch['mask'].to(torch.bool))
            else:
                logits = model(data_all, data_batch['mask'].to(torch.bool))

            acc = utils.eval_metrics.Accuracy(logits, data_batch["label"].type_as(logits))
            train_acc.append(acc.item())
            # save info
            train_labels.append(data_batch['label'].cpu().numpy())
            train_logits.append(logits.detach().cpu().numpy())
            user_ids.append(data_batch['uid'])
            num_user += data_batch['mask'].size()[0]

    print(f"Training acc {np.mean(train_acc)}, number of users: {num_user}")
    user_ids = np.concatenate(user_ids, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_logits = np.concatenate(train_logits, axis=0)
    np.savetxt(name + "scores.csv", np.stack([user_ids, train_labels, train_logits], axis=0))
