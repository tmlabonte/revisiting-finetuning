# Imports Python packages.
import numpy as np
from numpy.random import default_rng

# Imports PyTorch packages.
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

# Imports milkshake packages.
from milkshake.datamodules.dataset import Subset
from milkshake.datamodules.datamodule import DataModule


class Retrain(DataModule):
    def __init__(self, args, *xargs):
        super().__init__(args, *xargs)
        
        self.heldout_pct = 0.5

        self.balance_erm = args.balance_erm
        self.balance_retrain = args.balance_retrain
        self.mixture_ratio = args.mixture_ratio
        self.retrain_type = args.retrain_type if hasattr(args, "retrain_type") else "erm"
        self.split = args.split
        self.train_pct = args.train_pct

    def _shuffle_in_unison(self, indices, targets):
        p = default_rng(seed=self.seed).permutation(len(indices))
        return indices[p], targets[p]

    def _make_groups_array(self, indices):
        groups = np.zeros(len(indices), dtype=np.int32)
        for i, x in enumerate(indices):
            for j, group in enumerate(self.dataset_train.groups):
                if x in group:
                    groups[i] = j
        return groups

    def _make_balanced_subset(
        self,
        indices,
        targets_or_groups,
        min_count,
        len_targets_or_groups,
    ):
        subset = []
        counts = [0] * len_targets_or_groups
        for idx, target_or_group in zip(indices, targets_or_groups):
            if counts[target_or_group] < min_count:
                subset.append(idx)
                counts[target_or_group] += 1
        print(f"Data subset: {counts}")

        return subset

    def _split_dataset(self, dataset_train, dataset_val):
        if dataset_train.train_indices is None or dataset_val.val_indices is None:
            len_dataset = len(dataset_train)
            splits = self._get_splits(len_dataset)
            all_inds = list(range(len_dataset))
            dataset_train.train_indices = all_inds[:splits[0]]
            dataset_val.val_indices = all_inds[-splits[1]:]

        train_inds = dataset_train.train_indices
        val_inds = dataset_val.val_indices
        default_rng(seed=self.seed).shuffle(val_inds)

        retrain_num = int(self.heldout_pct * len(val_inds))
        new_val_inds = val_inds[retrain_num:]

        if self.split == "train" and self.train_pct == 100:
            new_train_inds = train_inds
            new_retrain_inds = val_inds[:retrain_num]
        elif self.split == "train":
            default_rng(seed=self.seed).shuffle(train_inds)
            train_num = int(len(train_inds) * self.train_pct / 100)
            new_train_inds = train_inds[:train_num]
            new_retrain_inds = train_inds[train_num:]
        elif self.split == "combined":
            combined_inds = np.concatenate((train_inds, val_inds))
            train_num = int((len(train_inds) + retrain_num) * self.train_pct / 100)
            new_retrain_num = len(combined_inds) - \
                    int((1 - self.heldout_pct) * len(val_inds))
            new_combined_inds = combined_inds[:new_retrain_num]

            default_rng(seed=self.seed).shuffle(new_combined_inds)
            new_train_inds = new_combined_inds[:train_num]
            new_retrain_inds = new_combined_inds[train_num:]

        dataset_train = Subset(dataset_train, new_train_inds)
        dataset_retrain = Subset(dataset_val, new_retrain_inds)
        dataset_val = Subset(dataset_val, new_val_inds)

        dataset_retrain.val_indices = new_retrain_inds
        dataset_val.val_indices = new_val_inds

        return dataset_train, dataset_retrain, dataset_val

    def group_unbalanced_dataloader(self, balance):
        if balance == "upsampling":
            self.balanced_sampler = True
        elif balance == "subsetting":
            indices = self.dataset_train.train_indices
            targets = self.dataset_train.targets[indices][:, 0] # Removes group dimension
            indices, targets = self._shuffle_in_unison(indices, targets)
            min_count = min(np.unique(targets, return_counts=True)[1])
            subset = self._make_balanced_subset(
                indices,
                targets,
                min_count,
                self.num_classes,
            )
            
            self.dataset_train = Subset(self.dataset_train, subset)
            self.balanced_sampler = False
        elif balance == "mixture":
            indices = self.dataset_train.train_indices
            targets = self.dataset_train.targets[indices][:, 0] # Removes group dimension
            indices, targets = self._shuffle_in_unison(indices, targets)
            min_count = min(np.unique(targets, return_counts=True)[1]) * self.mixture_ratio
            subset = self._make_balanced_subset(
                indices,
                targets,
                min_count,
                self.num_classes,
            )
            
            self.dataset_train = Subset(self.dataset_train, subset)
            self.balanced_sampler = True
        else:
            self.balanced_sampler = False
        return super().train_dataloader()

    def group_balanced_dataloader(self, balance):
        indices = self.dataset_train.train_indices
        groups = self._make_groups_array(indices)
        
        if balance == "upsampling":
            counts = np.bincount(groups)
            label_weights = 1. / counts
            weights = label_weights[groups]
            sampler = WeightedRandomSampler(weights, len(weights))
            return self._data_loader(self.dataset_train, sampler=sampler)
        elif balance == "subsetting":
            indices, targets = self._shuffle_in_unison(indices, targets)
            min_count = min([len(x) for x in self.dataset_train.groups])
            subset = self._make_balanced_subset(
                indices,
                targets,
                min_count,
                len(self.dataset_train.groups),
            )
            
            self.dataset_train = Subset(self.dataset_train, subset)
            self.balanced_sampler = False
            return super().train_dataloader()
        elif balance == "mixture":
            indices, targets = self._shuffle_in_unison(indices, targets)
            min_count = min([len(x) for x in self.dataset_train.groups]) * self.mixture_ratio
            subset = self._make_balanced_subset(
                indices,
                targets,
                min_count,
                len(self.dataset_train.groups),
            )
            
            self.dataset_train = Subset(self.dataset_train, subset)
            self.balanced_sampler = True
            return super().train_dataloader()
        else:
            raise ValueError("Must set balance for group-balanced training.")

    def train_dataloader(self, shuffle=True):
        if "retraining" in self.retrain_type:
            new_set = self.dataset_class(
                self.data_dir,
                train=True,
                transform=self.train_transforms,
            )
            new_set.train_indices = np.arange(len(new_set))
            self.dataset_train = Subset(
                new_set,
                self.dataset_retrain.val_indices
            )

        if self.retrain_type == "group-unbalanced retraining":
            return self.group_unbalanced_dataloader(self.balance_retrain)
        elif self.retrain_type == "group-balanced retraining":
            return self.group_balanced_dataloader(self.balance_retrain)
        else: # For ERM training.
            return self.group_unbalanced_dataloader(self.balance_erm)

    def val_dataloader(self):
        #dataloaders = super().val_dataloader()
                
        # Hack to get train/test metrics at each val epoch.
        # Data augmentation and balanced sampling are turned off.
        dataloaders = [
            self._data_loader(self.dataset_train_no_aug), # VAL ON TRAIN SET
            super().test_dataloader(), # VAL ON TEST SET
        ]

        return dataloaders

    def test_dataloader(self):
        dataloaders = super().test_dataloader()
        return dataloaders

    def setup(self, stage=None):
        dataset_train = self.dataset_class(
            self.data_dir,
            train=True,
            transform=self.train_transforms,
        )

        dataset_val = self.dataset_class(
            self.data_dir,
            train=True,
            transform=self.val_transforms,
        )

        dataset_test = self.dataset_class(
            self.data_dir,
            train=False,
            transform=self.test_transforms,
        )

        dataset_train = self.train_preprocess(dataset_train)
        dataset_val = self.val_preprocess(dataset_val)
        dataset_test = self.test_preprocess(dataset_test)

        self.dataset_train, self.dataset_retrain, self.dataset_val = \
            self._split_dataset(dataset_train, dataset_val)
        self.dataset_test = dataset_test

        # Hack for validating on the training set. See self.val_dataloader().
        dataset_train_no_aug = self.train_preprocess(self.dataset_class(
            self.data_dir,
            train=True,
            transform=self.default_transforms(),
        ))
        self.dataset_train_no_aug, _, _ = \
            self._split_dataset(dataset_train_no_aug, dataset_val)

        if self.balance_erm == "subsetting":
            indices = self.dataset_train_no_aug.train_indices
            targets = self.dataset_train_no_aug.targets[indices][:, 0] # Removes group dimension
            indices, targets = self._shuffle_in_unison(indices, targets)
            min_count = min(np.unique(targets, return_counts=True)[1])
            subset = self._make_balanced_subset(
                indices,
                targets,
                min_count,
                self.num_classes,
            )
            
            self.dataset_train_no_aug = Subset(self.dataset_train_no_aug, subset)
        elif self.balance_erm == "mixture":
            indices = self.dataset_train.train_indices
            targets = self.dataset_train.targets[indices][:, 0] # Removes group dimension
            indices, targets = self._shuffle_in_unison(indices, targets)
            min_count = min(np.unique(targets, return_counts=True)[1]) * self.mixture_ratio
            subset = self._make_balanced_subset(
                indices,
                targets,
                min_count,
                self.num_classes,
            )
            
            self.dataset_train_no_aug = Subset(self.dataset_train_no_aug, subset)

        print(self.load_msg())
