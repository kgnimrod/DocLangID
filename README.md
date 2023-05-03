This repository contains the code for our paper: "DocLangID: Improving Few-Shot Training for Language Identification of Historical Documents" (Arxiv Upload soon).



# Overview
This rest of this README is strucutured in the following way: 

1. [**Project Files and Structure**](#project-files-and-structure)
2. [**Reproducing our Results using SLURM**](#reproducing-our-results-using-slurm)
    - [**Enroot Image Installation**](#enroot-image-installation)
    - [**Setup and Configurations**](#setup-and-configurations)
    - [**Running the experiments**](#running-the-experiments)
        - [**Fully-Supervised Pretraining and Meta Training**](#1-fully-supervised-pretraining-and-meta-training)
        - [**Few-Shot Training on the WPI datasets**](#2-few-shot-training-on-the-wpi-datasets)
3. [**References**](#references)

# Project Files and Structure
First, we used the [Project template](https://gitlab.hpi.de/deeplearning/students/project-template) provided for the usage with SLURM. Our implementations can be found in the `/src` directory and mainly consist of Python files. Our deep learning approaches were implemented using PyTorch and PyTorch Lightning as main frameworks and SKLearn as well as TorchMetrics to evaluate the performance of our trained models.

In the following, we give a brief introduction of the main files:
- `meta_training.py`: Can be used to easily perform the two-stage training of our main mehtod. This script will first train a ResNet-18 model on the joint dataset created from the IMPACT and WPI datasets. Then, this model will be used in the second stage to further finetune its classification head by performing few-shot training on the few-shot subsets of WPI. After this two stage training, the model will be saved and classification performance will be evaluated. 
- `training_lightning.py`: Start a fully-supervised pretraining or meta training of a ResNet-18 model. Additionally, start a few-shot training on an already pretrained model. After the training, the model will be saved and classification performance will be evaluated. 
- `model.py`: Core implementation of the supervised pretraining and few-shot training approaches. Contains model instantiation, optimization, and evaluation steps.
- `eval.py`: Can be used in case a saved model should be re-evaluated on a dataset.
- `dataset.py`: Core implementation of the pytorch dataset module used to encompass the data access and augmentation processes.
- `preprocessing.py`: OpenCV preprocessing pipeline used to initially reduce the files sizes of the large source images.
- `inference.py`: Given a path to a saved model and the path to an input image, performs inference on the image. Starts from preprocessing the image using the OpenCV pipeline.
- `augmentations.py`: Implementation of the augmentation pipeline used to randomly augment train images during training.
- `create_wpi_data/`: Contains scripts that were used to create our WPI dataset 
- `tesseract_script.py`: A script that can be used to determine the language with our OCR/Tesseract approach (see [Language Detection using OCR](https://gitlab.hpi.de/till.nowakowski/masterproject-art-historical-documents/-/wikis/Language-Detection-using-OCR))


# Reproducing our Results using SLURM

## Enroot Image Installation

We used an enroot image to run our experiments on the Cluster.

The script `./scripts/build-image-enroot.sh` (or in short `./scripts/build-image.sh`) builds 
the enroot image in enroot.
It starts from the configured `BASE_IMAGE` and runs [`00_install.sh`](install/00_install.sh)
(this file is very similar to the [Dockerfile](Dockerfile)).

All scripts start from a docker base image (defined by `BASE_IMAGE`) in some way and
create a .sqsh file named `TARGET_SQSH` (default is `${TARGET_TAG}.sqsh`).

**You have to build the images on a host other than the `slurmsubmit` host.** (Building on this machine is disabled to prevent overusing the limited resources)

- In case you already have a different enroot image under `/enroot_share/` in usage, you might need to remove it first
- in a terminal, allocate a node by running `salloc`
- in a terminal, navigate to `/scripts` and run `./build-image-enroot.sh`
- When the installations successfully finishes, there should be a new image file under `/enroot_share/${USER}` on the cluster

## Setup and Configurations

- 1. Clone this repository to your cluster
- 2. Under `/scripts`, open [customize.sh](scripts/customize.sh) and scroll down to the function called `container_mounts()`
    - You will need to replace the paths to the container mounts with your environment paths

- 3. In order to save the models after the training create a folder named `/saved_models` under the same directory to where you cloned the repository to

## Running the experiments
In the following you can find the commands needed to prepare and then submit experiments to SLURM. Regarding the configurability of the experiments and the hyper-parameters, we used YAML files that we load at the beginning of each script. These files can be found in the [`experiment_yaml_files`](https://gitlab.hpi.de/till.nowakowski/masterproject-art-historical-documents/-/tree/main/src/experiment_yaml_files) directory. For pretraining and few-shot training, you should use the respective YAML files (starting with "pretraining_" or "few_shot_"). Each of these files represent the different possible training settings of this project and will be loaded depending on which value you set for the command line argument `mode`. To run `eval.py` to re-evaluate a model on a test dataset, you can use the same YAML file which you used for the previous pre/few-shot training of that model. To run `inference.py`, you should use one of the YAML files starting with "inference_" and additionally matches the setting the model was trained on. 

For instance, you train a model on IMPACT using `pretraining_lightning.py` with corresponding mode `training_impact`. To re-evaluate this trained model on IMPACT, run `eval.py` with mode `evaluation_impact`. In both of these steps, the same YAML file `pretraining_impact.yml` will be used. Lastly, to perform inference on this trained model, run `inference.py` with mode `evaluation` and set the argument `evaluation-dataset` to `evaluation_impact`. This will load the YAML file `inference_impact.yml`.

Regarding the datasets needed for the different training settings, we will list the locations of the datasets in the following:
- 1. Pretraining on IMPACT: 
    - dataset location on the cluster: `/home/datasets/vision/impact/preprocessed/images_resized/`
    - command line argument to set: `--dataset-source-domain`

- 2. Pretraining on the full WPI set (used for the demo):
    - dataset location on the cluster: `/home/datasets/vision/wpi/language_identification/preprocessed/cat_appr/evaluation/images_resized/`
    - command line argument to set: `--dataset-source-domain`

- 3. Pretraining on the few-shot subdatasets of the WPI dataset (used to compare with meta training)
    - dataset location on the cluster: `/home/datasets/vision/wpi/language_identification/preprocessed/cat_appr/`
    - command line argument to set: `--dataset-source-domain`

- 4. Joint meta pretraining on IMPACT and on one of the few-shot subdatasets of WPI.
    - dataset location for IMPACT:  `/home/datasets/vision/impact/preprocessed/images_resized/`
    - command line argument to set: `--dataset-source-domain`
    - dataset location for the few-shot subdatasets on the cluster: `/home/datasets/vision/wpi/language_identification/preprocessed/cat_appr/`
    - command line argument to set: `--dataset-target-domain`

### 1. **Fully-Supervised Pretraining and Meta Training**

1.1 Pretraining on the IMPACT dataset
```bash
NAME="pretraining_impact" ./scripts/prepare.sh python pretraining_lightning.py --mode training_impact --num-workers 16 --dataset-source-domain <PATH TO SOURCE DOMAIN DATA> --wandb-key <OPTIONAL> --wandb-entity <OPTIONAL> --wandb-project <OPTIONAL>
```

1.2 Pretraining on the WPI dataset (Using large amounts of labelled WPI data)
```bash
NAME="pretraining_wpi" ./scripts/prepare.sh python pretraining_lightning.py --mode training_wpi --num-workers 16 --dataset-source-domain <PATH TO SOURCE DOMAIN DATA> --wandb-key <OPTIONAL> --wandb-entity <OPTIONAL> --wandb-project <OPTIONAL>
```

1.3 Meta Training on the IMPACT & WPI datasets
```bash
NAME="pretraining_meta" ./scripts/prepare.sh python pretraining_lightning.py --mode training_meta --num-workers 16 --dataset-source-domain <PATH TO SOURCE DOMAIN DATA> --dataset-target-domain <PATH TO few WPI data> --wandb-key <OPTIONAL> --wandb-entity <OPTIONAL> --wandb-project <OPTIONAL>
```

1.4 Pretraining only on the few WPI samples (subsets)
```bash
NAME="pretraining_few_shot" ./scripts/prepare.sh python pretraining_lightning.py --mode training_few_shot --num-workers 16 --dataset-source-domain <PATH TO few WPI data> --wandb-key <OPTIONAL> --wandb-entity <OPTIONAL> --wandb-project <OPTIONAL>
```

- This will create the experiment under your `/runs` folder. To run the experiment you will need to type the folder name of the created experiment into the next command. The experiment folder names look like this for instance: "2022_08_08__14_21_48"
```bash
NUM_GPU=1 ./scripts/submit.sh "runs/default/pretraining_impact/<EXPERIMENT DATE CREATED IN PREVIOUS COMMAND, e.g. 2022_08_08__14_21_48>"  
```

### 2. **Few-Shot Training on the WPI datasets**
Here, you will need to provide the path to the model that you want to use for the few-shot training using the command line argument: `pretrained-path`. Additionally, you will need to provide the path to whatever WPI subset you want to use for few-shot training in the command line argument: `dataset-directory`.

2.1 Few-Shot training using `N=50` (to change this, set `few-shot-n` accordingly (possible values: 5, 10, 25 and 50))
```bash
NAME="few_shot_training" ./scripts/prepare.sh python few_shot_training_lightning.py --pretrained-path <YOUR PATH TO THE SAVED MODEL> --mode training_few_shot --num-workers 16 --dataset-source-domain <PATH TO FEW WPI DATA> --wandb-key <OPTIONAL> --wandb-entity <OPTIONAL> --wandb-project <OPTIONAL>
```

- This will create the experiment under your `/runs` folder. To run the experiment you will need to type the folder name of the created experiment into the next command. The experiment folder names look like this for instance: "2022_08_08__14_21_48"
```bash
NUM_GPU=1 ./scripts/submit.sh "runs/default/few_shot_training/<EXPERIMENT DATE CREATED IN PREVIOUS COMMAND, e.g. 2022_08_08__14_21_48>"  
```

# References
[1] Shah, S., Joshi, M.V. (2021). Document Languag
e Classification: Hierarchical Model with Deep Learning Approach. In: Tsapatsoulis, N., Panayides, A., Theocharides, T., Lanitis, A., Pattichis, C., Vento, M. (eds) Computer Analysis of Images and Patterns. CAIP 2021. Lecture Notes in Computer Science(), vol 13052. Springer, Cham. https://doi.org/10.1007/978-3-030-89128-2_36

[2] Karimpour, M., Noori Saray, S., Tahmoresnezhad, J. et al. Multi-source domain adaptation for image classification. Machine Vision and Applications 31, 44 (2020). https://doi.org/10.1007/s00138-020-01093-2

[3] Y. Zhao et al., "Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification," 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 6273-6282, doi: 10.1109/CVPR46437.2021.00621.

[4] Vatsal, S. et al. (2021). On-Device Language Identification of Text in Images Using Diacritic Characters. In: Singh, S.K., Roy, P., Raman, B., Nagabhushan, P. (eds) Computer Vision and Image Processing. CVIP 2020. Communications in Computer and Information Science, vol 1377. Springer, Singapore. https://doi.org/10.1007/978-981-16-1092-9_42

[5] Chen, W.Y., Liu, Y.C., Kira, Z., Wang, Y.C., & Huang, J.B.. (2019). A Closer Look at Few-shot Classification. https://arxiv.org/abs/1904.04232

[6] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G.. (2020). A Simple Framework for Contrastive Learning of Visual Representations. https://arxiv.org/abs/2002.05709

[7] Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D.. (2020). Supervised Contrastive Learning. https://arxiv.org/abs/2004.11362v4
