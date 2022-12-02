import argparse
from cProfile import label
from cgi import test
from json import load
import os
import torch
import random
import pytorch_lightning as pl
import wandb
from tqdm import tqdm
import itertools
from pathlib import Path
from dataset import get_datasets

from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
from model import ResNetModule
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from config import load_config_from_yaml

parser = argparse.ArgumentParser(description='PyTorch Lightning ResNet Training')
parser.add_argument('--dataset-path', metavar='DIR',
    default='/preprocessed/meta', 
    type=str, 
    dest="dataset_path",help='path to the source domain dataset', 
)
parser.add_argument('--num-workers',           default=16, type=int, metavar='N', help='number of data loading workers (default: 4)', dest='num_workers')
parser.add_argument('--mode',                  default="training_impact", type=str, dest='mode', choices=['training_meta', 'training_impact', 'training_wpi', 'training_few_shot'])
parser.add_argument('--wandb-key',             default=None, type=str, dest='wandb_key')
parser.add_argument('--wandb-entity',          default='mpss22', type=str, dest='wandb_entity')
parser.add_argument('--wandb-project',         default='new_data_loader_training', type=str, dest='wandb_project')
parser.add_argument('--wandb-run',             default=None, type=str, dest='wandb_run')
parser.add_argument('--seed',                  default=None, type=int, dest='seed')
parser.add_argument('--pretrained-path',       default=None, type=str, dest='pretrained_path')
parser.add_argument('--save-path',             default=None, type=str, dest='save_path')


def main():
    args = parser.parse_args()
    
    if args.mode == 'training_impact':
        hyper_params = load_config_from_yaml('experiment_yaml_files/pretraining_impact.yml')
    elif args.mode == 'training_wpi':
        hyper_params = load_config_from_yaml('experiment_yaml_files/pretraining_wpi.yml')
    elif args.mode == 'training_meta':
        hyper_params = load_config_from_yaml('experiment_yaml_files/pretraining_meta.yml')
    elif args.mode == 'training_few_shot':
        hyper_params = load_config_from_yaml('experiment_yaml_files/few_shot_training.yml')
    
    hyper_params['mode'] = args.mode
    hyper_params['num_workers'] = args.num_workers
    hyper_params['seed'] = args.seed
    hyper_params['pretrained_path'] = args.pretrained_path
    hyper_params['save_path'] = args.save_path

    if args.wandb_key is not None:
        wandb.login(key=args.wandb_key)
        if args.wandb_run is None:
            run_name = f'resnet18_{args.mode}{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}'
        else:
            run_name = args.wandb_run
        logger = WandbLogger(
            name=run_name,
            project=args.wandb_project,
            entity=args.wandb_entity
        )
    
    train_dataset, val_dataset, test_dataset, class_names = get_datasets(args.dataset_path, hyper_params=hyper_params)

    # randomly log 10 images from the training set for later manual supervision/evaluation
    if args.wandb_key is not None:
        log_images = []
        for i in range(10):
            randindex = random.randint(0, len(train_dataset)-1)
            log_images.append(
                wandb.Image(
                    train_dataset[randindex]['image_transformed'], 
                    caption=f'image_filename_{train_dataset[randindex]["filename"]}'
                )
            )
        wandb.log({f'train images': log_images})
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize a ResNet18 model
    model = ResNetModule(
        datasets={'train': train_dataset, 'val': val_dataset, 'test': test_dataset},
        hyper_params=hyper_params, wandb_key=args.wandb_key).to(device)

    if hyper_params['seed'] is not None:
        seed_everything(hyper_params['seed'], workers=True)

    # run the training
    print(f"###### STARTING TRAINING ######")

    trainer = pl.Trainer(
        logger=logger if args.wandb_key is not None else None,
        auto_lr_find=True,
        #callbacks=[early_stop_callback],
        # auto_scale_batch_size="power",
        # weights_summary=None,
        log_every_n_steps=10,
        max_epochs=hyper_params["epochs"],
        #gpus=1 if torch.cuda.is_available() else 0,
        deterministic=True,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else 0
    )
    
    # train the model
    trainer.fit(model=model)
    print(f"###### FINISHED TRAINING ######")

    # Model Evaluation
    print(f"###### STARTING EVALUATION ######")

    # setting the model into evaluation mode
    model.eval()
    model.to("cpu")

    y_pred, y_true = [], []
    # evaluate the model predictions for each language and then print the confusion matrix
    for i in tqdm(range(len(test_dataset))):
        out = model.make_prediction(test_dataset[i]['image'].to("cpu"))
        y_pred.append(int(torch.max(torch.mean(out, 0),0)[1]))
        y_true.append(int(torch.max(test_dataset[i]["label"],0)[1]))
    

    # print the confusion matrix
    print(f"###### PRINTING CONFUSION MATRIX ######")
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(conf_matrix)
    
    print(f"###### PRINTING ADDITIONAL CLASSIFICATION RESULT DATA")
    classification_rep = classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names)
    print(classification_rep)

    model.to(device)
    trainer.test(model)
    
    # save the model
    print(f"###### SAVING THE MODEL ######")
    
    # create target directory if it does not exist already
    if hyper_params['save_path'] is not None:
        base_path = hyper_params["save_path"]
        save_directory = os.path.join(base_path, 'resnet18_meta_training.pt')
    else:
        base_path = f'/saved_models/{hyper_params["mode"]}'
        result_metric = round(trainer.callback_metrics["test_acc_epoch"].item(), 2)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
        save_directory = os.path.join(base_path, f'resnet18_{result_metric}_{timestamp}.pt')
    Path(base_path).mkdir(parents=True, exist_ok=True)

    torch.save({
        'feature_extractor' : model.resnet18.state_dict(),
        'classifier': model.lin_classifier.state_dict()}, 
        save_directory
    )
    
    # finally upload the test data images and the prediction/label pairs
    if args.wandb_key is not None:
        for lang_index in tqdm(range(len(class_names))):
            images = []
            for i in range(len(test_dataset)):
                if (y_true[i] == lang_index):
                    images.append(
                        wandb.Image(
                            test_dataset[i]['image_transformed'], 
                            caption=f'IMAGE: {test_dataset[i]["filename"]},\
                                PREDICTION: {class_names[y_pred[i]]},\
                                GROUNDTRUTH: {class_names[y_true[i]]}'
                        )
                    )
                if i == 20:
                    break
            wandb.log({f'images_{class_names[lang_index]}': images})
    print(f"RUN COMPLETED!")
    
if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    main()