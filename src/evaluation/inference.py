import argparse
import os
from unittest.mock import patch
import numpy as np
import cv2
import torch
from core.model import ResNetModule
import torch.nn.functional as F
from utils.dataset import ArtHistoricalDocumentsDataset
from pathlib import Path, PurePath
from utils.preprocessing import preprocessing_image
import os, os.path
from pathlib import Path
import json
from utils.config import load_config_from_yaml
from utils.language_classes import get_language_targets
import pytorch_lightning as pl

parser = argparse.ArgumentParser(description='PyTorch Lightning Inference')
parser.add_argument('--image-path',            
    default='/preprocessed/demo_examples/de/286116_12264478.jpg',
    metavar='DIR', 
    type=str, 
    dest="image_path", 
    help='path to the image to generate a prediction'
)
parser.add_argument('--pretrained-path',
    type=str, 
    default='/saved_models/pretraining_wpi/pretraining_resnet18_test_acc_0.932022-09-21_15:59:29.pt', 
    dest="pretrained_path", 
    help='path to pretrained model'
)
parser.add_argument('--mode',
    default='evaluation', 
    type=str, 
    dest='mode',
)
parser.add_argument('--evaluation-dataset',
    default='evaluation_wpi', 
    type=str, 
    dest='evaluation_dataset', 
    choices=['evaluation_impact', 'evaluation_wpi', 'wpi700k', 'evaluation_meta', 'evaluation_few_shot']
)
parser.add_argument('--num-workers', 
    default=16, 
    type=int, 
    metavar='N',
    help='number of data loading workers (default: 4)', 
    dest='num_workers'
)


def main():
    args = parser.parse_args()

    if args.evaluation_dataset == 'evaluation_impact':
        hyper_params = load_config_from_yaml('experiment_yaml_files/inference_impact.yml')
    elif args.evaluation_dataset in ['evaluation_wpi', 'wpi700k']:
        hyper_params = load_config_from_yaml('experiment_yaml_files/inference_wpi.yml')
    elif args.evaluation_dataset  == 'evaluation_few_shot':
        hyper_params = load_config_from_yaml('experiment_yaml_files/inference_few_shot.yml')
    else:
        hyper_params = load_config_from_yaml('experiment_yaml_files/inference_meta.yml')

    hyper_params['mode'] = args.mode
    hyper_params['pretrained_path'] = args.pretrained_path
    hyper_params['image_path'] = args.image_path
    hyper_params['evaluation_dataset'] = args.evaluation_dataset
    hyper_params['shapes'] = (int(hyper_params['shape_x']), int(hyper_params['shape_y']))
    hyper_params['num_workers'] = args.num_workers
    
    languages, _ = get_language_targets(hyper_params['evaluation_dataset'])
    languages = list(languages.keys())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # create the model
    model = ResNetModule(
        datasets={'train': None,'val': None, 'test': None}, 
        hyper_params=hyper_params).to(device)

    model.to(device)

    # this branch is only used to generate the project submission (i.e. the labels for the +700k images from WPI)
    if hyper_params['evaluation_dataset'] == 'wpi700k':
        results = []
        filenames = [Path(name) for name in os.listdir('/preprocessed/de/')]
        for filename in filenames:
            print(filename)
            image_raw = cv2.imread(os.path.join('/preprocessed/de/', str(filename)), cv2.IMREAD_UNCHANGED)

            image_preprocessed = preprocessing_image(image_raw, hyper_params['shapes'])
            image_preprocessed = np.reshape(image_preprocessed, newshape=(image_preprocessed.shape[0], image_preprocessed.shape[1], 1))
            image_preprocessed = np.broadcast_to(image_preprocessed, shape=(image_preprocessed.shape[0], image_preprocessed.shape[1], 3))

            image_transformed = ArtHistoricalDocumentsDataset(
                data=[np.array(image_preprocessed, copy=True)], 
                label_list=torch.tensor([0]).float(),
                filenames=PurePath(hyper_params['image_path']).parts[-1],
                mode=hyper_params['mode'],
                patch_size_x=hyper_params['patch_size_x'],
                patch_size_y=hyper_params['patch_size_y'],
                num_patches=hyper_params['patch_size']
            )
            y_pred = model.make_prediction(image_transformed[0]['image'].to(device))
            y_pred_language = languages[int(y_pred.argmax())]        
            results.append(
                {
                    str(filename): {
                        'prediction': y_pred_language,
                        'confidence': float(y_pred[0][int(y_pred.argmax())]*100)
                    }
                }
            )

        json_data = json.dumps(
            results,
            #indent=4
        )

        save_file = open("/preprocessed/results_wpi700k.json", mode='w')
        json.dump(json_data, save_file)
            
    else:
        # first, load the raw .tiff/.jpeg image
        image_raw = cv2.imread(hyper_params['image_path'], cv2.IMREAD_UNCHANGED)
        print(image_raw)
        # then apply the preprocessing steps on the image and then return it as a PIL image
        image_preprocessed = preprocessing_image(image_raw, hyper_params['shapes'])
        image_preprocessed = np.reshape(image_preprocessed, newshape=(image_preprocessed.shape[0], image_preprocessed.shape[1], 1))
        image_preprocessed = np.broadcast_to(image_preprocessed, shape=(image_preprocessed.shape[0], image_preprocessed.shape[1], 3))

        image_transformed = ArtHistoricalDocumentsDataset(
            data=[np.array(image_preprocessed, copy=True)], 
            label_list=torch.tensor([0]).float(),
            filenames=PurePath(hyper_params['image_path']).parts[-1],
            mode=hyper_params['mode'],
            patch_size_x=hyper_params['patch_size_x'],
            patch_size_y=hyper_params['patch_size_y'],
            num_patches=hyper_params['patch_size']
        )  
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1 if torch.cuda.is_available() else 0
        )
        data_loader = torch.utils.data.DataLoader(
            image_transformed[0]['image'],
            batch_size=hyper_params['patch_size'],
            num_workers=hyper_params['num_workers'],
        )
        y_pred = trainer.predict(model, data_loader)
        y_pred = y_pred[0].squeeze()
        y_pred_language = languages[int(y_pred.argmax())]
        
        print(f'######## PREDICTED LANGUAGE: {y_pred_language} ########')
        print(f'######## CONFIDENCE: {y_pred[int(y_pred.argmax())]*100}%')
        
if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    main()