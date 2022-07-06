# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import os

import numpy as np
import json
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import dgl
from dgllife.utils.eval import Meter
from dgl.data.utils import Subset, extract_archive
from dgl.data.utils import load_graphs

from sklearn.model_selection import train_test_split
from utils import load_dataset, load_model, rand_hyperparams, set_random_seed

import pandas as pd
import boto3


def update_msg_from_scores(msg, scores):
    for metric, score in scores.items():
        msg += ", {} {:.4f}".format(metric, score)
    return msg


def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter(args["train_mean"], args["train_std"])
    epoch_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        bg, labels = batch_data
        labels = labels.to(args["device"])
        if args["model"] == "PotentialNet":
            bigraph_canonical, knn_graph = bg  # unpack stage1_graph, stage2_graph
            bigraph_canonical = bigraph_canonical.to(args["device"])
            knn_graph = knn_graph.to(args["device"])
            prediction = model(bigraph_canonical, knn_graph)
        elif args["model"] == "ACNN":
            bg = bg.to(args["device"])
            prediction = model(bg)
        loss = loss_criterion(prediction, (labels - args["train_mean"]) / args["train_std"])
        epoch_loss += loss.data.item() * len(labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels)
    avg_loss = epoch_loss / len(data_loader.dataset)
    # if (args['num_epochs'] - epoch) <= 6: # print only the last 5 epochs
    total_scores = {
        metric: train_meter.compute_metric(metric, "mean") for metric in args["metrics"]
    }
    msg = "epoch {:d}/{:d}, training | loss {:.4f}".format(epoch + 1, args["num_epochs"], avg_loss)
    msg = update_msg_from_scores(msg, total_scores)
    print(msg)
    return total_scores


def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter(args["train_mean"], args["train_std"])
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            bg, labels = batch_data
            labels = labels.to(args["device"])
            if args["model"] == "PotentialNet":
                bigraph_canonical, knn_graph = bg  # unpack
                bigraph_canonical = bigraph_canonical.to(args["device"])
                knn_graph = knn_graph.to(args["device"])
                prediction = model(bigraph_canonical, knn_graph)
            elif args["model"] == "ACNN":
                bg = bg.to(args["device"])
                prediction = model(bg)
            eval_meter.update(prediction, labels)
    total_scores = {metric: eval_meter.compute_metric(metric, "mean") for metric in args["metrics"]}
    return total_scores

class MyDataset(Dataset):
    def __init__(self, lst_graph1_paths):
        super().__init__()
        
        self.len = len(lst_graph1_paths)
        self.lst = lst_graph1_paths
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        graphs1, label_dict = load_graphs(self.lst[index])
        graphs2, label_dict = load_graphs(self.lst[index].replace('_g1.bin', '_g2.bin'))
        label = label_dict['glabel']
        
        graphs1_batch = [dgl.batch([graphs1[i],graphs1[i+1]]) for i in range(0, int(len(graphs1)), 2)]
        bg = [tuple([graphs1_batch[i],graphs2[i]]) for i in range(0, int(len(graphs2)), 1)]
        
        return bg[0], label
    
def collate(data):
    graphs, labels = map(list, zip(*data))
    if (type(graphs[0]) == tuple):
        bg1 = dgl.batch([g[0] for g in graphs])
        bg2 = dgl.batch([g[1] for g in graphs])
        bg = (bg1, bg2) # return a tuple for PotentialNet
    else:
        bg = dgl.batch(graphs)
        for nty in bg.ntypes:
            bg.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
        for ety in bg.canonical_etypes:
            bg.set_e_initializer(dgl.init.zero_initializer, etype=ety)

    labels = torch.stack(labels, dim=0)
    return bg, labels


def main(args):

    torch.multiprocessing.set_sharing_strategy("file_system")
    args["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_random_seed(args["random_seed"])
    
    lst_csv = []
    lst_csv += glob.glob(args["data_dir"] +"/"+ "**.csv")
    assert len(lst_csv) <= 1 # Make sure there aren't more than 1 csv files. 
    print(lst_csv)
    
    if len(lst_csv) == 1: # There is a csv file for split instruction 
        _lst_g1_train = pd.read_csv(lst_csv[0])[pd.read_csv(lst_csv[0])['split'] == 'train']['g1_bin'].to_list()
        _lst_g1_val = pd.read_csv(lst_csv[0])[pd.read_csv(lst_csv[0])['split'] == 'val']['g1_bin'].to_list()
        _lst_g1_test = pd.read_csv(lst_csv[0])[pd.read_csv(lst_csv[0])['split'] == 'test']['g1_bin'].to_list()
        
        lst_g1_train = [args["data_dir"] +"/"+ pdb for pdb in _lst_g1_train]
        lst_g1_val = [args["data_dir"] +"/"+ pdb for pdb in _lst_g1_val]
        lst_g1_test = [args["data_dir"] +"/"+ pdb for pdb in _lst_g1_test]
        
        train_set = MyDataset(lst_g1_train)
        val_set = MyDataset(lst_g1_val)
        test_set = MyDataset(lst_g1_test)
        
    else: # If there is no csv file, use random split. 
        lst_g1 = glob.glob(args["data_dir"] +"/"+ "**_g1.bin")
        print(lst_g1)
        
        lst_g1_train, lst_g1_val = train_test_split(lst_g1, test_size=0.33, random_state=42)
        
        train_set = MyDataset(lst_g1_train)
        val_set = MyDataset(lst_g1_val)
        test_set = MyDataset(lst_g1_val)
    
    train_labels = torch.stack([g[1] for g in train_set])
    train_set.labels_mean = train_labels.mean(dim=0)
    train_set.labels_std = train_labels.std(dim=0)
    
    
    args["train_mean"] = train_set.labels_mean.to(args["device"])
    args["train_std"] = train_set.labels_std.to(args["device"])
    

    print("Data Loader")
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args["batch_size"],
        shuffle=args["shuffle"],
        collate_fn=collate,
        pin_memory=True,
        num_workers=args["num_workers"],
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args["batch_size"],
        collate_fn=collate,
        pin_memory=True,
        num_workers=args["num_workers"],
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args["batch_size"],
        collate_fn=collate,
        pin_memory=True,
        num_workers=args["num_workers"],
    )
    
    # Get distance_bins length
    for i0, g in enumerate(train_set):
        g2_edge_dim = g[0][1].edata['e'].shape[1]
        break

    print(f"g2_edge_dim : {g2_edge_dim - 5}")
    fake_distance_bins = [i1 for i1 in range(g2_edge_dim - 5)]
    args['distance_bins'] = fake_distance_bins
    
    model = load_model(args)
    if args['fine_tune']:
        print('--Fine Tune--')
        s3_bucket = args['pretrained_model'].split('/')[2]
        object_name = '/'.join(args['pretrained_model'].split('/')[3:])
        file_name = args['pretrained_model'].split('/')[-1]
        s3 = boto3.client('s3')
        with open(file_name, 'wb') as f:
            s3.download_fileobj(s3_bucket, object_name, f)
        extract_archive('model.tar.gz', '')
        model.load_state_dict(torch.load('best_test_model.pth'))
        
        ## Freeze Param
        print('--Freeze Params--')
        for i1, child in enumerate(model.children()):
            print(child)
            for param in child.parameters():
                param.requires_grad = False
            if i1 == 1:
                break
        
    
    if args["num_gpus"] > 1:
        print("Gpu count: {}".format(args["num_gpus"]))
        model = nn.DataParallel(model)
        
    ### This is something we can work with mask ###
    
    loss_fn = nn.MSELoss()
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["wd"])
    model.to(args["device"])
    n_epochs = args["num_epochs"]
    
    train_r2, val_r2, test_r2 = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)
    for epoch in range(n_epochs):
        train_scores = run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)
        train_r2[epoch] = train_scores["r2"]
        if len(val_set) > 0:
            val_scores = run_an_eval_epoch(args, model, val_loader)
            val_msg = update_msg_from_scores("validation results", val_scores)
            print(val_msg)
            print(f"------")
            print(f"val mae:{val_msg.split('mae')[-1]}")
            print(f"val r2:{val_msg.split('r2')[-1].split(',')[0]}")
            print(f"------")
            val_r2[epoch] = val_scores["r2"]
        if len(test_set) > 0:
            test_scores = run_an_eval_epoch(args, model, test_loader)
            test_msg = update_msg_from_scores("test results", test_scores)
            print(test_msg)
            print(f"------")
            print(f"test mae:{test_msg.split('mae')[-1]}")
            print(f"test r2:{test_msg.split('r2')[-1].split(',')[0]}")
            print(f"------")
            test_r2[epoch] = test_scores["r2"]
            print("")
                   
        best_epoch = np.argmax(val_r2)
        if epoch == best_epoch and best_epoch != 0 and not np.isnan(val_r2[best_epoch]) :
            print("Best val epoch: ", best_epoch + 1)
        
            print("Saving the model")
            path = os.path.join(args["model_dir"], "best_val_model.pth")
            # recommended way from http://pytorch.org/docs/master/notes/serialization.html
            # This model artifact is on GPU      
            torch.save(model.state_dict(), path)

        best_epoch = np.argmax(test_r2)
        if epoch == best_epoch and best_epoch != 0 and not np.isnan(test_r2[best_epoch]) :
            print("Best test epoch: ", best_epoch + 1)
        
            print("Saving the model")
            path = os.path.join(args["model_dir"], "best_test_model.pth")
            # recommended way from http://pytorch.org/docs/master/notes/serialization.html
            # This model artifact is on GPU 
            torch.save(model.state_dict(), path)
                  
        print("")
                  
    # save model r2 at each epoch
    if args["save_r2"]:
        os.makedirs(args["save_r2"], exist_ok=True)
        save_path = args["save_r2"] + "/{}_{}_{}_{}.npz".format(
            args["model"], args["version"], args["subset"], args["split"]
        )
        np.savez(save_path, train_r2=train_r2, val_r2=val_r2, test_r2=test_r2)

        # save results on the epoch with best validation r2
        best_epoch = np.argmax(val_r2)
        print("Best test epoch: ", best_epoch + 1)
                  
    print("Saving the model")
    path = os.path.join(args["model_dir"], "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


if __name__ == "__main__":
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description="Protein-Ligand Binding Affinity Prediction")
    parser.add_argument(
        "-m", "--model", type=str, choices=["ACNN", "PotentialNet"], help="Model to use"
    )
    parser.add_argument(
        "-d",
        "--dataset_option",
        type=str,
        choices=[
            "PDBBind_core_pocket_random",
            "PDBBind_core_pocket_scaffold",
            "PDBBind_core_pocket_stratified",
            "PDBBind_core_pocket_temporal",
            "PDBBind_refined_pocket_random",
            "PDBBind_refined_pocket_scaffold",
            "PDBBind_refined_pocket_stratified",
            "PDBBind_refined_pocket_temporal",
            "PDBBind_refined_pocket_structure",
            "PDBBind_refined_pocket_sequence",
        ],
        help="Data subset and split to use",
    )
    #parser.add_argument(
    #    "--pdb_path", type=str, default=None, help="local path of custom PDBBind dataset"
    #)
    parser.add_argument("-v", "--version", type=str, choices=["v2007", "v2015"], default="v2015")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of processes for loading PDBBind molecules and Dataloader",
    )
    parser.add_argument(
        "--save_r2", type=str, default="", help="path to save r2 at each epoch, default not save"
    )
    parser.add_argument(
        "--test_on_core",
        type=bool,
        default=True,
        help="whether to use the whole core set as test set when training on refined set, default True",
    )

    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    #parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    

    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--wd", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--num_epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--distance_bins", type=lambda s: json.loads(s.replace("'", '"')),default=argparse.SUPPRESS)
    parser.add_argument("--fine_tune", type=bool, default=False)
    parser.add_argument("--pretrained_model", type=str, default="")

    args = parser.parse_args().__dict__

    args["model"] = "PotentialNet"
    args["dataset_option"] = "PDBBind_refined_pocket_scaffold"
    args["exp"] = "_".join([args["model"], args["dataset_option"]])
    #args["exp"] = "PotentialNet_PDBBind_refined_pocket_scaffold"
    
    if os.environ.get("SM_CHANNEL_TRAIN"):
        args["data_dir"] = os.environ["SM_CHANNEL_TRAIN"]

    default_exp = get_exp_configure(args["exp"])
    for i in default_exp.keys():
        args.setdefault(i, default_exp[i])

    if args["num_workers"] == 0:
        args["num_workers"] = multiprocessing.cpu_count()

    #### We don't need to use this #### 
    #if args["split"] == "sequence" or args["split"] == "structure":
    #    args["version"] = "v2007"
    #    args["test_on_core"] = False
    #    args["remove_coreset_from_refinedset"] = False
    #
    #if args["subset"] == "core":
    #    args["remove_coreset_from_refinedset"] = False
    #    args["test_on_core"] = False
    #### We don't need to use this #### 
    
    #if args["pdb_path"]:
    #    args["pdb_path"] = os.environ["SM_CHANNEL_TRAIN"] + args["pdb_path"]

    rand_hyper_search = False
    if rand_hyper_search:  # randomly initialize hyperparameters
        customized_hps = rand_hyperparams()
        args.update(customized_hps)
    for k, v in args.items():
        print(f"{k}: {v}")

    print("")
    main(args)
