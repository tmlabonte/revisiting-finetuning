"""Utility functions for milkshake."""

# Imports Python packages.
import os
import os.path as osp
from glob import glob
import math
import numpy as np
import warnings

# Imports PyTorch packages.
import torch
from torch._utils import _accumulate

# Imports milkshake packages.
from milkshake.datamodules.dataset import Subset


def get_weights_helper(ckpt_path, version, best=None, idx=None):
    """Returns weights path given model version and checkpoint index.

    Args:
        ckpt_path: The root directory for checkpoints.
        version: The model's PyTorch Lightning version.
        best: Whether to return the best model checkpoint.
        idx: The model's checkpoint index (-1 selects the latest checkpoint).

    Returns:
        The filepath of the desired model weights.
    """

    # Selects the right naming convention for PL versions based on
    # whether the version input is an int or a string.
    if isinstance(version, int):
        ckpt_path = osp.join(ckpt_path, f"lightning_logs/version_{version}/checkpoints/*")
    else:
        ckpt_path = osp.join(ckpt_path, f"lightning_logs/{version}/checkpoints/*")

    list_of_weights = glob(osp.join(os.getcwd(), ckpt_path))

    if best:
        return [w for w in list_of_weights if "best" in w][0]
    else:
        list_of_weights = sorted([w for w in list_of_weights if "best" not in w])
        return list_of_weights[idx]

def get_weights(args, version, best=None, idx=None):
    """Returns weights path given model version and checkpoint index.

    Args:
        args: The configuration dictionary.
        version: The model's PyTorch Lightning version.
        best: Whether to return the best model checkpoint.
        idx: The model's checkpoint index (-1 selects the latest checkpoint).

    Returns:
        The filepath of the desired model weights.
    """

    try:
        return get_weights_helper("", version, best=best, idx=idx)
    except:
        return get_weights_helper(args.wandb_dir, version, best=best, idx=idx)

def compute_accuracy(probs, targets, num_classes, num_groups):
    """Computes top-1 and top-5 accuracies.

    Computes top-1 and top-5 accuracies by total and by class, and also
    includes the number of correct predictions and number of samples.
    The latter is useful for, e.g., collating metrics over an epoch.
    If groups are provided (as a second column in targets), then
    also returns group accuracies.

    Args:
        probs: A torch.Tensor of prediction probabilities.
        targets: A torch.Tensor of classification targets.
        num_classes: The total number of classes.
        num_groups: The total number of groups.

    Returns:
        A dictionary of metrics including top-1 and top-5 accuracies, number of
        correct predictions, and number of samples, each by total, class, and group.
    """

    # TODO: Clean up group metrics.

    # Splits apart targets and groups if necessary.
    groups = None
    if num_groups:
        groups = targets[:, 1]
        targets = targets[:, 0]

    if num_classes == 1:
        preds1 = (probs >= 0.5).int()
    else:
        preds1 = torch.argmax(probs, dim=1)

    correct1 = (preds1 == targets).float()

    acc1_by_class = []
    correct1_by_class = []
    total_by_class = []
    for j in range(num_classes):
        correct_nums = correct1[targets == j]
        acc1_by_class.append(correct_nums.mean())
        correct1_by_class.append(correct_nums.sum())
        total_by_class.append((targets == j).sum())

    acc1_by_group = [torch.tensor(1., device=targets.device)]
    correct1_by_group = [torch.tensor(1., device=targets.device)]
    total_by_group = [torch.tensor(1., device=targets.device)]
    if num_groups:
        acc1_by_group = []
        correct1_by_group = []
        total_by_group = []
        for j in range(num_groups):
            correct_nums = correct1[groups == j]
            acc1_by_group.append(correct_nums.mean())
            correct1_by_group.append(correct_nums.sum())
            total_by_group.append((groups == j).sum())

    correct5 = torch.tensor([1.] * len(targets), device=targets.device)
    acc5_by_class = [torch.tensor(1., device=targets.device)] * num_classes
    correct5_by_class = total_by_class
    if num_classes > 5:
        _, preds5 = torch.topk(probs, k=5, dim=1)
        correct5 = torch.tensor([t in preds5[j] for j, t in enumerate(targets)], device=targets.device)
        correct5 = correct5.float()

        acc5_by_class = []
        correct5_by_class = []
        for j in range(num_classes):
            correct_nums = correct5[targets == j]
            acc5_by_class.append(correct_nums.mean())
            correct5_by_class.append(correct_nums.sum())

    acc5_by_group = [torch.tensor(1., device=targets.device)]
    correct5_by_group = [torch.tensor(1., device=targets.device)]
    if num_groups:
        acc5_by_group = [torch.tensor(1., device=targets.device)] * num_groups
        correct5_by_group = total_by_group
        if num_classes > 5:
            acc5_by_group = []
            correct5_by_group = []
            for j in range(num_groups):
                correct_nums = correct5[groups == j]
                acc5_by_group.append(correct_nums.mean())
                correct5_by_group.append(correct_nums.sum())

    accs = {
        "acc": correct1.mean(),
        "acc5": correct5.mean(),
        "acc_by_class": torch.stack(acc1_by_class),
        "acc5_by_class": torch.stack(acc5_by_class),
        "acc_by_group": torch.stack(acc1_by_group),
        "acc5_by_group": torch.stack(acc5_by_group),
        "correct": correct1.sum(),
        "correct5": correct5.sum(),
        "correct_by_class": torch.stack(correct1_by_class),
        "correct5_by_class": torch.stack(correct5_by_class),
        "correct_by_group": torch.stack(correct1_by_group),
        "correct5_by_group": torch.stack(correct5_by_group),
        "total": len(targets),
        "total_by_class": torch.stack(total_by_class),
        "total_by_group": torch.stack(total_by_group),
    }

    return accs

def get_vectorized_features(targets, features, num_classes, num_groups, retraining):
    """Gets vectorized features for each group, class and training split

    Args:
        targets: A torch.Tensor of classification targets.
        features: A torch.Tensor of model features.
        num_classes: The total number of classes.
        num_groups: The total number of groups.

    Returns:
        A dictionary of vectorized features broken down by group, class, and training split.
    """

    # # print(features)
    # print(features.shape)

    # print(targets)

    # TODO: Clean up group metrics.

    # Splits apart targets and groups if necessary.
    groups = None
    if num_groups:
        groups = targets[:, 1]
        targets = targets[:, 0]

    # Vectorize Features

    # Get the batch size
    batch_size = features.size(0)

    # Vectorize the features
    batch_vectorized_features = features.view(batch_size, -1)

    # # Print the shape of the vectorized tensor
    # print(batch_vectorized_features.shape)

    features_by_class = {}

    for j in range(num_classes):
        class_features = batch_vectorized_features[targets == j]
        if j in features_by_class:
            features_by_class[j] = torch.concat((features_by_class[j], class_features))
        else:
            features_by_class[j] = class_features

    

    # print(features_by_class.keys())

    if num_groups:
        features_by_group = {}
        for j in range(num_groups):
            group_features = batch_vectorized_features[groups == j]
            if j in features_by_group:
                features_by_group[j] = torch.concat((features_by_group[j], group_features))
            else:
                features_by_group[j] = group_features

    # print(features_by_group.keys())

    # features_by_split = {}

    # if retraining == False:
    #     ### This means that features should be saved for the training split
    #     features_by_split[False] = batch_vectorized_features
    # else:
    #     ### This means that features should be saved for the last layer retraining split
    #     features_by_split[True] = batch_vectorized_features


    vectorized_features = {
        "features_by_group": features_by_group,
        "features_by_class": features_by_class,
        "features": batch_vectorized_features
        # "features_by_split": features_by_split
    }


    return vectorized_features

def compute_collapse_metrics(features, num_classes, num_groups):
    """Computes feature collapse metrics between groups and classes

    Computes the intra-group, inter-group, intra-class, and 
    inter-class feature covariances.

    Args:
        features: A dictionary of dictionaries containing torch.Tensor of model features.
        num_classes: The total number of classes.
        num_groups: The total number of groups.

    Returns:
        A dictionary of metrics including intra-group, inter-group, intra-class, and 
    inter-class feature covariances.
    """

    ### Compute means used in covariance computation

    global_mean = torch.mean(features["features"], dim=0)
    class_means = []
    group_means = []
    
    for j in range(num_classes):
        class_means.append(torch.mean(features["features_by_class"][j], dim=0))

    for j in range(num_groups):
        group_means.append(torch.mean(features["features_by_group"][j], dim=0))

    ### Center features for covariance computation

    centered_features = features["features"] - global_mean
    centered_features = centered_features

    ### Calculate overall covariance

    N = centered_features.shape[0]

    # print(centered_features.shape)
    
    total_cov = (1/N) * torch.matmul(centered_features.t(), centered_features)
    # print(total_cov.size())

    total_cov_norm = torch.frobenius_norm(total_cov)

    ### Calculate covariance metrics for the classes

    inter_class_cov = (1/num_classes)*torch.matmul((torch.stack(class_means, dim=0) - global_mean).t(), (torch.stack(class_means, dim=0) - global_mean))
    # print(inter_class_cov.size())
    
    inter_class_cov_norm = torch.frobenius_norm(inter_class_cov)

    centered_classes = []    
    for j in range(num_classes):
        centered_classes.append(features["features_by_class"][j] - class_means[j])
    centered_classes = torch.cat(centered_classes, dim=0)
    intra_class_cov = (1/N)* torch.matmul(centered_classes.t(), centered_classes)

    # print(intra_class_cov.size())

    intra_class_cov_norm = torch.frobenius_norm(intra_class_cov)

    class_trace = torch.trace((1/num_classes)*torch.matmul(intra_class_cov, torch.pinverse(inter_class_cov)))

    ### Calculate covariance metrics for the groups

    inter_group_cov = (1/num_groups)*torch.matmul((torch.stack(group_means, dim=0) - global_mean).t(), (torch.stack(group_means, dim=0) - global_mean))
    inter_group_cov_norm = torch.frobenius_norm(inter_group_cov)

    # print(inter_group_cov.size())

    centered_groups = []    
    for j in range(num_groups):
        centered_groups.append(features["features_by_group"][j] - group_means[j])
    centered_groups = torch.cat(centered_groups, dim=0)
    intra_group_cov = (1/N)* torch.matmul(centered_groups.t(), centered_groups)

    # print(intra_group_cov.size())

    intra_group_cov_norm = torch.frobenius_norm(intra_group_cov)

    group_trace = torch.trace((1/num_groups)*torch.matmul(intra_group_cov, torch.pinverse(inter_group_cov)))

    collapse_metrics = {
        "global_cov": total_cov_norm,
        "inter_class_cov": inter_class_cov_norm,
        "intra_class_cov": intra_class_cov_norm,
        "inter_group_cov": inter_group_cov_norm,
        "intra_group_cov": intra_group_cov_norm,
        "class_trace": class_trace,
        "group_trace": group_trace
    }

    return collapse_metrics

def _to_np(x):
    """Converts torch.Tensor input to numpy array."""

    return x.cpu().detach().numpy()

def to_np(x):
    """Converts input to numpy array.

    Args:
        x: A torch.Tensor, np.ndarray, or list.

    Returns:
        The input converted to a numpy array.

    Raises:
        ValueError: The input cannot be converted to a numpy array.
    """

    if not len(x):
        return np.array([])
    elif isinstance(x, torch.Tensor):
        return _to_np(x)
    elif isinstance(x, (np.ndarray, list)):
        if isinstance(x[0], torch.Tensor):
            return _to_np(torch.stack(x))
        else:
            return np.asarray(x)
    else:
        raise ValueError("Input cannot be converted to numpy array.")

def random_split(dataset, lengths, generator):
    """Random split function from PyTorch adjusted for milkshake.Subset.

    Args:
        dataset: The milkshake.Dataset to be randomly split.
        lengths: The lengths or fractions of splits to be produced.
        generator: The generator used for the random permutation.

    Returns:
        A list of milkshake.Subsets with the desired splits.

    Raises:
        ValueError: The sum of input lengths does not equal the length of the input dataset.
    """

    # Handles the case where lengths is a list of fractions.
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)

        # Adds 1 to all the lengths in round-robin fashion until the remainder is 0.
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                print(f"Length of split at index {i} is 0. "
                      f"This might result in an empty dataset.")

    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths), generator=generator).tolist()
    return [Subset(dataset, indices[offset - length : offset])
            for offset, length in zip(_accumulate(lengths), lengths)]

def ignore_warnings():
    """Adds nuisance warnings to ignore list.

    Should be called before import pytorch_lightning.
    """

    warnings.filterwarnings(
        "ignore",
        message=r"The feature ([^\s]+) is currently marked under review",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"In the future ([^\s]+)",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"Lazy modules ([^\s]+)",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"There is a wandb run already in progress ([^\s]+)",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"`resume_download` ([^\s]+)",
    )


    original_filterwarnings = warnings.filterwarnings
    def _filterwarnings(*xargs, **kwargs):
        return original_filterwarnings(*xargs, **{**kwargs, "append": True})
    warnings.filterwarnings = _filterwarnings
