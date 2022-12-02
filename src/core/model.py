from cProfile import label
import torch
import wandb
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.nn.utils.weight_norm import WeightNorm
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image


class ResNetModule(pl.LightningModule):
    def __init__(self, datasets, hyper_params=None, wandb_key=None):
        super().__init__()
        
        self.wandb_key = wandb_key
        self.criterion = torch.nn.CrossEntropyLoss()
        self.use_torch_cam = False
        self.hyper_params = hyper_params
        self.datasets = {'train': datasets['train'], 'val': datasets['val'], 'test': datasets['test']}
        self.resnet18 = models.resnet18(pretrained=False, progress=True)
        self.resnet18.fc = nn.Identity()
        self.lin_classifier = None
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # training mode
        if self.hyper_params['mode'].split('_')[0] == "training":
            
            # IMPACT/WPI/META data pretraining
            if self.hyper_params["pretrained_path"] is None:
                self.lin_classifier = nn.Linear(in_features=512, out_features=self.hyper_params['num_languages'], bias=True) 
                pass

            # Fine Tuning on WPI data
            else:
                # load the pretrained model trained on IMPACT/META data and then adjust the last layer
                feature_extractor = torch.load(f'{self.hyper_params["pretrained_path"]}', map_location=device)['feature_extractor']
                self.resnet18.load_state_dict(feature_extractor)
                self.lin_classifier = nn.Linear(in_features=512, out_features=self.hyper_params['num_languages'], bias=True)

                for m in self.resnet18.parameters():
                    m.requires_grad = False

                if self.hyper_params['few_shot_mode'] == "baseline++":
                    WeightNorm.apply(self.lin_classifier, name='weight', dim=0)
                    self.scale_factor = 2
                    
        # evaluation mode
        else:
            # load a pretrained model trained on IMPACT/WPI/META data
            self.resnet18.load_state_dict(torch.load(f'{self.hyper_params["pretrained_path"]}', map_location=device)['feature_extractor'])
            self.lin_classifier = nn.Linear(in_features=512, out_features=self.hyper_params['num_languages'], bias=True)
            if self.hyper_params['few_shot_mode'] == "baseline++":
                WeightNorm.apply(self.lin_classifier, name='weight', dim=0)
                self.scale_factor = 2
            self.lin_classifier.load_state_dict(torch.load(f'{self.hyper_params["pretrained_path"]}', map_location=device)['classifier'])
            
        self.train_acc = torchmetrics.Accuracy(threshold=0.5, subset_accuracy=False)
        self.val_acc = torchmetrics.Accuracy(threshold=0.5, subset_accuracy=False)
        self.test_acc = torchmetrics.Accuracy(threshold=0.5, subset_accuracy=False)

    def forward(self, x):
        # the batch x has shape: (batch_size X patch_size X 3 X patch_size_x X patch_size_y)
        # since we process the batch in one forward pass, we reshape the batch such that the first dimension of the new tensor equals the product
        # of the previous first two dimensions, e.g. (10 X 8 X 3 X 300 X 300) becomes (80 X 3 X 300 X 300).
        batch_size = x.shape[0]
        x = x.reshape((-1,)+x.shape[2:])
        out_features = None

        # extract features
        if self.use_torch_cam and self.hyper_params['few_shot_mode'] is None:
            out_features = []
            for patch in x:
                out_features.append(self.resnet18(patch.unsqueeze(0)))         
            out_features = torch.stack(out_features)
            self.use_torch_cam = False
        else:
            out_features = self.resnet18(x) 
        
        if self.hyper_params['few_shot_mode'] == 'baseline++':
            x_norm = torch.norm(out_features, p=2, dim=1)

            x_norm = x_norm.unsqueeze(1).expand_as(out_features)
            x_normalized = out_features.div(x_norm + 0.00001)

            L_norm = torch.norm(self.lin_classifier.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.lin_classifier.weight.data)
            self.lin_classifier.weight.data = self.lin_classifier.weight.data.div(L_norm + 0.00001)

            cos_dist = self.lin_classifier(x_normalized) #matrix product by linear function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm
            out = self.scale_factor * cos_dist
        else:
            out = self.lin_classifier(out_features)
        
        if self.hyper_params['few_shot_mode'] is None:
            out_features = out.reshape((batch_size, self.hyper_params['patch_size'],self.hyper_params['num_languages']))

        # Then, the output shape will be (80 X num_languages) for instance. Each subsequent patch_size rows make up one image, so the predictions of each blocks of size patch_size
        # need to be considered separately and the majority vote needs to be performed on each block
        # Reshaping the output tensor to shape (batch_size X patch_size X num_languages) gives us batch_size many blocks on which we can perform the majority vote in order to yield one single
        # prediction per block, or in other words, one single prediction per image
        
        out = out.reshape((batch_size, self.hyper_params['patch_size'],self.hyper_params['num_languages']))

        # The final probability distribution is obtained over the mean over the columns of the blocks, i.e. the class that got the highest probabilities over the majority of the patches will also have the highest mean
        # compared to the mean of the other classes and will therefore be selected as predicted class for a given image
        out = torch.mean(out, 1)

        return out, out_features

    def make_prediction(self, x):
        out = x.unsqueeze(0)
        out, _ = self.forward(out)
        softm = nn.Softmax(dim=1)
        out = softm(out)
        return out
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = batch.unsqueeze(0)
        out, _ = self.forward(out)
        softm = nn.Softmax(dim=1)
        out = softm(out)
        return out

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]
        images_transformed = batch["image_transformed"]

        if batch_idx == 0 and self.hyper_params['few_shot_mode'] is None:
            cams = []
            results = []
            self.use_torch_cam = True
            if self.wandb_key is not None:
                cam_extractor = SmoothGradCAMpp(model=self.resnet18, target_layer=self.resnet18.layer4 if self.hyper_params['few_shot_mode'] is None else self.resnet18.fc)
                out, out_feats = self.forward(images)
                for i in range(out_feats.shape[0]):
                    cams.append(cam_extractor(int(labels[i].argmax()), out_feats[i]))
                    results.append(wandb.Image(overlay_mask(to_pil_image(images_transformed[i]), to_pil_image(cams[i][0].squeeze(0), mode='F'), alpha=0.5)))
                wandb.log({f'Epoch: {self.current_epoch}, images_with_torchcam_visualization': results})
                cam_extractor.clear_hooks()
            else:
                out, out_feats = self.forward(images)
        else:
            out, out_feats = self.forward(images)
        loss = self.criterion(out, labels)
        preds = torch.tensor([x.argmax() for x in out])
        labels = torch.tensor([x.argmax() for x in labels])
        self.train_acc(preds, labels)
        self.log("train_acc", self.train_acc, prog_bar=True, logger=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss, "train_acc": self.train_acc}

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]
        out, _ = self.forward(images)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(out, labels)
        preds = torch.tensor([x.argmax() for x in out])
        labels = torch.tensor([x.argmax() for x in labels])
        self.val_acc(preds, labels)
        self.log("val_acc", self.val_acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_acc": self.val_acc}
            
    def test_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]
        out, _ = self.forward(images)
        loss = self.criterion(out, labels)
        preds = torch.tensor([x.argmax() for x in out])
        labels = torch.tensor([x.argmax() for x in labels])
        self.test_acc(preds, labels)
        self.log("test_acc", self.test_acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"test_loss": loss, "test_acc": self.test_acc}

    def configure_optimizers(self):
        if self.hyper_params['few_shot_mode'] is None:
            print("USING OPTIMIZER FOR PRETRAINING")
            optimizer = torch.optim.Adam(self.resnet18.parameters(), self.hyper_params["lr"],
                weight_decay=self.hyper_params['weight_decay'])
        else:
            print("USING OPTIMIZER FOR FEW-SHOT TRAINING")
            optimizer = torch.optim.Adam(self.lin_classifier.parameters(), self.hyper_params["lr"],
                weight_decay=self.hyper_params['weight_decay'])

        return optimizer

    def train_dataloader(self):
        return DataLoader(
                self.datasets["train"], 
                shuffle=True, 
                batch_size=self.hyper_params["batch_size"],
                num_workers=self.hyper_params['num_workers'], 
                drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
                self.datasets["val"], 
                shuffle=False, 
                batch_size=self.hyper_params["batch_size"], 
                num_workers=self.hyper_params['num_workers'], 
                drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
                self.datasets["test"], 
                shuffle=False, 
                batch_size=self.hyper_params['batch_size'],
                num_workers=self.hyper_params['num_workers'], 
                drop_last=True
        )
