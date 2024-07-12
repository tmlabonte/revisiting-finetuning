# Ignores nuisance warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

# Imports Python builtins.
from copy import deepcopy
from glob import glob
import os.path as osp
import pickle

# Imports Python packages.
from configargparse import Parser
import numpy as np

# Imports PyTorch packages.
import torch
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer

# Imports milkshake packages.
from exps.finetune import *
from milkshake.args import add_input_args
from milkshake.main import main, load_weights

def reset_fc_hook(model):
    """Resets model classifier parameters."""

    try:
        for layer in model.model.classifier:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    except:
        model.model.classifier.reset_parameters()

def train_fc_only(model):
    """Freezes model parameters except for last layer."""
    for p in model.model.parameters():
        p.requires_grad = False

    try:
        for p in model.model.classifier.parameters():
            p.requires_grad = True
    except:
        for p in model.model.fc.parameters():
            p.requires_grad = True

def set_llr_args(args, train_type):
    """Sets args for last-layer retraining."""

    new_args = deepcopy(args)

    # Note that LLR is run for the same amount of epochs as ERM.
    new_args.ckpt_every_n_epochs = new_args.max_epochs + 1
    new_args.check_val_every_n_epoch = new_args.max_epochs
    new_args.lr = 1e-2
    new_args.lr_scheduler = "step"
    new_args.lr_steps = []
    new_args.optimizer = "sgd"
    new_args.train_fc_only = True
    new_args.wandb = False

    if train_type == "llr":
        new_args.train_type = "llr"
        new_args.retrain_type = "group-unbalanced retraining"
    elif train_type == "dfr":
        new_args.train_type = "dfr"
        new_args.retrain_type = "group-balanced retraining"

    return new_args

def compute_spectral_imbalance_metrics(args, model, datamodule):
    """Computes eigenvalues of group covariance matrices."""

    model.cuda().eval()
    if args.model == "bert":
        version = args.bert_version
    elif args.model == "convnextv2":
        version = args.convnextv2_version
    elif args.model == "resnet":
        version = args.resnet_version

    # Defines a hook to get model features for a given input.
    model.features = None
    def get_features():
        def hook(m, x, y):
            model.features = y
        return hook
    
    grp_eigvals = []
    class_eigvals = []
    with torch.no_grad():
        if args.model == "bert":
            model.model.bert.register_forward_hook(get_features())
        elif args.model == "convnextv2":
            model.model.convnextv2.register_forward_hook(get_features())
        elif args.model == "resnet":
            model.model.resnet.register_forward_hook(get_features())

        for class_idx in list(range(args.num_classes)):
            print(f"Computing features for {args.model} {version} class {class_idx}")

            # Creates DataLoaders for each of the groups.
            class_indices = [
                i for i, (_, label) in enumerate(datamodule.dataset_train_no_aug)
                if label[0] == class_idx
            ]
            class_subset = Subset(datamodule.dataset_train_no_aug, class_indices)
            class_dataloader = DataLoader(
                class_subset,
                batch_size=datamodule.batch_size,
                shuffle=False,
            )

            class_features = None
            for batch in class_dataloader:
                data = batch[0].cuda()

                model(data)
                features = model.features.last_hidden_state
                features = features.detach().cpu().flatten(start_dim=1)

                if class_features is None:
                    class_features = features
                else:
                    class_features = torch.concat((class_features, features))

            print(f"Computing covariance for {args.model} {version} class {class_idx}")
            class_mean = torch.mean(class_features, 0)
            class_cov = torch.matmul((class_features - class_mean).T, (class_features - class_mean))
            class_cov /= len(class_features) # Normalize by num of samples in group

            print(f"Computing eigenvalues for {args.model} {version} class {class_idx}")
            eigvals = torch.lobpcg(class_cov, k=50)[0] # Already sorted in descending order
            class_eigvals.append(list(eigvals.numpy()))

        for grp_idx in list(range(args.num_groups)):
            print(f"Computing features for {args.model} {version} group {grp_idx}")

            # Creates DataLoaders for each of the groups.
            grp_indices = [
                i for i, (_, label) in enumerate(datamodule.dataset_train_no_aug)
                if label[1] == grp_idx
            ]
            grp_subset = Subset(datamodule.dataset_train_no_aug, grp_indices)
            grp_dataloader = DataLoader(
                grp_subset,
                batch_size=datamodule.batch_size,
                shuffle=False,
            )

            grp_features = None
            for batch in grp_dataloader:
                data = batch[0].cuda()

                model(data)
                features = model.features.last_hidden_state
                features = features.detach().cpu().flatten(start_dim=1)

                if grp_features is None:
                    grp_features = features
                else:
                    grp_features = torch.concat((grp_features, features))

            print(f"Computing covariance for {args.model} {version} group {grp_idx}")
            grp_mean = torch.mean(grp_features, 0)
            grp_cov = torch.matmul((grp_features - grp_mean).T, (grp_features - grp_mean))
            grp_cov /= len(grp_features) # Normalize by num of samples in group

            print(f"Computing eigenvalues for {args.model} {version} group {grp_idx}")
            eigvals = torch.lobpcg(grp_cov, k=50)[0] # Already sorted in descending order
            grp_eigvals.append(list(eigvals.numpy()))

    return {"group_eigvals": grp_eigvals, "class_eigvals": class_eigvals}

def experiment(args, model_class, datamodule_class):
    args.no_test = True
    args.train_type = "erm"

    if not osp.isfile(args.results_pkl):
        raise ValueError(f"Results file {args.results_pkl} not found.")

    results = load_results(args)

    s = args.seed

    if args.model == "bert":
        v = args.bert_version
    elif args.model == "convnextv2":
        v = args.convnextv2_version
    elif args.model == "resnet":
        v = args.resnet_version

    c = args.balance_erm
    d = args.balance_retrain
    if "mixture" in c:
        c += str(args.mixture_ratio)
    if "mixture" in d:
        d += str(args.mixture_ratio)
    e = args.max_epochs

    wandb_version = results[s][v][c]["erm"][e]["version"]
    if not wandb_version:
        raise ValueError(f"Model version {wandb_version} not found.")

    # Finds model weights in wandb dir.
    fpath = "epoch=" + f"{e - 1:02d}" + "*"
    ckpt_path = osp.join(
        args.wandb_dir, "lightning_logs", wandb_version, "checkpoints", fpath)
    args.weights = glob(ckpt_path)[0]

    datamodule = datamodule_class(args)
    datamodule.setup()

    args.num_classes = datamodule.num_classes
    args.num_groups = datamodule.num_groups

    model = model_class(args)
    model = load_weights(args, model)

    spectral_imbalance_metrics = compute_spectral_imbalance_metrics(
        args, model, datamodule)
    dump_results(args, e, spectral_imbalance_metrics)
    
    """
    # Performs LLR.
    new_args = set_llr_args(args, "llr")
    train_fc_only(model)
    model.hparams.train_type = "llr"
    model, _, _ = main(
        new_args, model, datamodule_class, model_hooks=[reset_fc_hook])

    # Performs DFR.
    # new_args = set_llr_args(args, "dfr")
    # train_fc_only(model)
    # model.hparams.train_type = "dfr"
    # model, _, _ = main(
    #     new_args, model, datamodule_class, model_hooks=[reset_fc_hook])
    """

if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # Arguments imported from retrain.py.
    parser.add("--balance_erm", choices=["none", "subsetting", "upsampling", "upweighting", "mixture"], default="none",
               help="Which type of class-balancing to perform during ERM training.")
    parser.add("--balance_retrain", choices=["none", "subsetting", "upsampling", "upweighting", "mixture"], default="none",
               help="Which type of class-balancing to perform during retraining.")
    parser.add("--mixture_ratio", type=float, default=2,
               help="The largest acceptable class imbalance ratio for the mixture balancing strategy.")
    parser.add("--save_retrained_model", action="store_true",
               help="Whether to save the retrained model outputs.")
    parser.add("--split", choices=["combined", "train"], default="train",
               help="The split to train on; either the train set or the combined train and held-out set.")
    parser.add("--train_pct", default=100, type=int,
               help="The percentage of the train set to utilize (for ablations)")

    datamodules = {
        "celeba": CelebARetrain,
        "civilcomments": CivilCommentsRetrain,
        "multinli": MultiNLIRetrain,
        "waterbirds": WaterbirdsRetrain,
    } 
    models = {
        "bert": BERTWithLogging,
        "convnextv2": ConvNeXtV2WithLogging,
        "resnet": ResNetWithLogging,
    }

    args = parser.parse_args()
    args.results_pkl = f"{args.datamodule}_{args.model}.pkl"
    experiment(args, models[args.model], datamodules[args.datamodule])
