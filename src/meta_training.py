import argparse
from cProfile import label
from cgi import test
from json import load
import os
from datetime import datetime
from core.config import load_config_from_yaml

parser = argparse.ArgumentParser(description='PyTorch Lightning ResNet Training')
parser.add_argument('--dataset-path', metavar='DIR',
    default='/preprocessed/meta', 
    type=str, 
    dest="dataset_path",help='path to the source domain dataset', 
)
parser.add_argument('--num-workers',           default=16, type=int, metavar='N', help='number of data loading workers (default: 4)', dest='num_workers')
parser.add_argument('--wandb-key',             default=None, type=str, dest='wandb_key')
parser.add_argument('--wandb-entity',          default='mpss22', type=str, dest='wandb_entity')
parser.add_argument('--wandb-project',         default='patch_size_analysis', type=str, dest='wandb_project')
parser.add_argument('--wandb-run',             default=f'meta_training_resnet18_{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}', type=str, dest='wandb_run')
parser.add_argument('--seed',                  default=None, type=int, dest='seed')
parser.add_argument('--save-path',             default='/saved_models/meta_run', type=str, dest='save_path')


def main():
    args = parser.parse_args()
    
    hyper_params = load_config_from_yaml('experiment_yaml_files/pretraining_meta.yml')
    
    os.system(f'python core/training_lightning.py \
        --mode training_meta \
        --dataset-path /language_identification/meta/ \
        --wandb-key {args.wandb_key} \
        --wandb-entity {args.wandb_entity} \
        --wandb-project {args.wandb_project} \
        --wandb-run {args.wandb_run} \
        --save-path {args.save_path}'
    )

    save_path = os.path.join(args.save_path, 'resnet18_meta_training.pt')
    
    os.system(f'python core/training_lightning.py \
        --mode training_few_shot \
        --dataset-path /language_identification/wpi/ \
        --pretrained-path {save_path} \
        --wandb-key {args.wandb_key} \
        --wandb-entity {args.wandb_entity} \
        --wandb-project {args.wandb_project} \
        --wandb-run {args.wandb_run}'
    )

if __name__ == '__main__':
    main()