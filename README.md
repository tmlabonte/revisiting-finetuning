# The Group Robustness is in the Details: Revisiting Finetuning under Spurious Correlations
### Official codebase for the NeurIPS 2024 paper: [https://arxiv.org/abs/2407.13957](https://arxiv.org/abs/2407.13957)
### Installation
```
conda update -n base -c defaults conda
conda create -n milkshake python==3.10
conda activate milkshake
conda install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install -e .
```
### Instructions
To run an experiment, specify the config with `-c`. For example,
```
python exps/finetune.py -c cfgs/waterbirds.yaml
```

By default, the program will run ERM finetuning with no class-balancing. Here is an example of a run with a different class-balancing method and model size:
```
python exps/finetune.py -c cfgs/waterbirds.yaml --convnextv2_version nano --balance_erm mixture --mixture_ratio 2
```

After models are finetuned, run eigenvalue computations with `exps/postprocess.py`.

### Citation and License
This codebase uses [Milkshake](https://github.com/tmlabonte/milkshake) as a template and inherits its MIT License. Please consider using the following citation:
```
@inproceedings{labonte24revisiting,
  author={Tyler LaBonte and John C. Hill and Xinchen Zhang and Vidya Muthukumar and Abhishek Kumar},
  title={The Group Robustness is in the Details: Revisiting Finetuning under Spurious Correlations},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2024},
}
```
