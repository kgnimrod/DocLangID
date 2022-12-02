import argparse
import os
import torch
from tqdm import tqdm 
import itertools

from sklearn.metrics import confusion_matrix, classification_report
from core.model import ResNetModule
from utils.prepare_datasets import prepare_datasets, prepare_datasets_wpi_few_shot, create_datasets
import torch.nn.functional as F
from utils.config import load_config_from_yaml
from utils.language_classes import get_language_targets

# Use this argument to specify the path to the source domain dataset -
# If evaluating on IMPACT, use the first path from the list of choices.
# Otherwise, use the second path from the list of choices to load the WPI dataset
parser = argparse.ArgumentParser(description='PyTorch Lightning ResNet Evaluation')
parser.add_argument('--dataset-source-domain', 
    metavar='DIR', 
    default='/impact/preprocessed/images_resized/', 
    type=str, 
    dest="data",
    help='path to source domain dataset',
    choices=[
        '/impact/preprocessed/images_resized/',
        '/wpi/language_identification/preprocessed/cat_appr/evaluation/images_resized/',
        ]
    )

# this argument only needs to be used when evaluation a Meta trained model to additionally load -
# the target domain datast. Therefore, there is only one path which needs to be considered
parser.add_argument('--dataset-target-domain', 
    metavar='DIR', 
    default='/wpi/language_identification/preprocessed/cat_appr/', 
    type=str, 
    dest="data_meta",
    help='path to target domain dataset dataset',
    choices=[
        '/wpi/language_identification/preprocessed/cat_appr/'
    ]
)
parser.add_argument('--pretrained-path',       metavar='DIR', default='/saved_models/pretraining_meta/pretraining_resnet18_test_acc_0.72022-09-21_19:45:11.pt', type=str, dest="pretrained_path",help='path to resnet model')
parser.add_argument('--mode',                  default="evaluation_few_shot", type=str, dest='mode', choices=['evaluation_wpi', 'evaluation_impact', 'evaluation_meta', 'evaluation_few_shot', 'evaluation_few_shot_pre'])

def main():
    args = parser.parse_args()

    if args.mode == 'evaluation_impact':
        hyper_params = load_config_from_yaml('experiment_yaml_files/pretraining_impact.yml')
    elif args.mode == 'evaluation_wpi':
        hyper_params = load_config_from_yaml('experiment_yaml_files/pretraining_wpi.yml')
    elif args.mode == 'evaluation_meta':
        hyper_params = load_config_from_yaml('experiment_yaml_files/pretraining_meta.yml')
    elif args.mode == 'evaluation_few_shot_pre':
        hyper_params = load_config_from_yaml('experiment_yaml_files/pretraining_few_shot.yml')
    elif args.mode == 'evaluation_few_shot':
        hyper_params = load_config_from_yaml('experiment_yaml_files/few_shot_training.yml')


    hyper_params['mode'] = args.mode
    hyper_params['pretrained_path'] = args.pretrained_path

    print(
        f'###### STARTING EVALUATION on {hyper_params["mode"].split("_")[1]} \
        data using a {"Pretrained" if hyper_params["few_shot_mode"] is None else "Finetuned"} model ######' )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # prepare the datasets depending on the setting on how the model was trained before
    languages_dict, _ = get_language_targets(hyper_params['mode'])
    if hyper_params['few_shot_mode'] is None:
        if hyper_params['mode'] in ['evaluation_wpi', 'evaluation_impact']:
            languages_dict, _ = get_language_targets(hyper_params['mode'])
            _, _, test_dataset = prepare_datasets(
                languages=languages_dict, 
                path=args.data, 
                hyper_params=hyper_params
            )
        else:
            print(f"###### LOADING META DATASETS ######")
            _, _, test_dataset = prepare_datasets(
                languages=dict(itertools.islice(languages_dict.items(),0,6)), 
                path=args.data, 
                hyper_params=hyper_params
            )
            _, _, test_dataset_meta = prepare_datasets_wpi_few_shot(
                languages=dict(itertools.islice(languages_dict.items(),6,10)), 
                path=args.data_meta, 
                hyper_params=hyper_params
            )
            _, _, test_dataset = create_datasets(
                train_data=None,
                val_data=None,
                test_data=[
                    test_dataset.data+test_dataset_meta.data, 
                    test_dataset.label_list+test_dataset_meta.label_list, 
                    test_dataset.filenames+test_dataset_meta.filenames
                ],
                hyper_params=hyper_params
            )
    # used to evaluate a few-shot tuned model on the target domain
    else:
        _, _, test_dataset = prepare_datasets_wpi_few_shot(
            languages=languages_dict, 
            path=args.data,
            hyper_params=hyper_params
        )

    # instantiate a ResNet18 model
    model = ResNetModule(datasets={'train': None, 'val': None, 'test': test_dataset},
     hyper_params=hyper_params).to(device)

    print(f'{torch.cuda.get_device_name(0)}')
    print(f'{torch.cuda.Device}')

    languages = list(languages_dict.keys())

    # Model Evaluation
    print(f'###### USING model: {hyper_params["pretrained_path"]}')
        
    # setting the model into evaluation mode
    model.eval()
    y_pred, y_true = [], []
    # evaluate the model predictions for each language and then print the confusion matrix
    for i in tqdm(range(len(test_dataset))):
        out = model.make_prediction(test_dataset[i]["image"].to(device))
        y_pred.append(int(torch.max(torch.mean(out, 0),0)[1]))
        y_true.append(int(torch.max(test_dataset[i]["label"],0)[1]))

        print(f'PREDICTION: {y_pred[i]}, LABEL: {y_true[i]}')
        
    print(f"###### PRINTING CONFUSION MATRIX ######")
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(f'{conf_matrix}')

    print(f"###### PRINTING ADDITIONAL CLASSIFICATION RESULT DATA")
    classification_rep = classification_report(y_true=y_true, y_pred=y_pred, target_names=languages)
    print(classification_rep)

    print(f"RUN COMPLETED!")

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
