from semantic_aug.datasets.caltech101 import CalTech101Dataset
from semantic_aug.datasets.flowers102 import Flowers102Dataset
from semantic_aug.datasets.pets import PetsDataset
from semantic_aug.augmentations.dalda import DALDAAugmentation

from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from transformers import CLIPModel, CLIPTokenizer
from itertools import product
from tqdm import trange
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import pandas as pd
import numpy as np
import random
import os

import importlib.util

os.environ["TOKENIZERS_PARALLELISM"] = "true"

DEFAULT_MODEL_PATH = "CompVis/stable-diffusion-v1-4"
DEFAULT_BASE_PROMPT = "a photo of a {name}"
DEFAULT_SYNTHETIC_DIR = "aug/{dataset}-{aug}-{seed}-{examples_per_class}"
DEFAULT_CONFIGS_DIR = "configs"

DATASETS = {
    "caltech": CalTech101Dataset,
    "flowers": Flowers102Dataset,
    "pets" : PetsDataset
}

AUGMENTATIONS = {
    "dalda": DALDAAugmentation
}

def run_experiment(examples_per_class: int = 0, 
                   seed: int = 0, 
                   dataset: str = "pets", 
                   num_synthetic: int = 10,
                   iterations_per_epoch: int = 200, 
                   num_epochs: int = 50, 
                   batch_size: int = 32, 
                   aug: str = None,
                   guidance_scale: float = None,
                   synthetic_probability: float = 0.5, 
                   synthetic_dir: str = DEFAULT_SYNTHETIC_DIR, 
                   model_path: str = DEFAULT_MODEL_PATH,
                   prompt_mode: str = "base",
                   llm_prompt_path: str = None,
                   image_size: int = 224,
                   classifier_backbone: str = "clip",
                   scale: str = None,
                   num_inference_steps: int = 30,
                   num_workers: int = 4,
                   device: str = 'cuda:0',
                   resume: List[int] = [None, None, None, None]): # seed, examples_per_class, idx, num

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if aug is not None:

        aug = AUGMENTATIONS[aug](
                model_path=model_path,
                prompt_mode=prompt_mode,
                llm_prompt_path=llm_prompt_path,
                guidance_scale=guidance_scale,
                scale=scale,
                num_inference_steps=num_inference_steps,
                examples_per_class = examples_per_class,
                device=device
            )

    train_dataset = DATASETS[dataset](
        split="train", examples_per_class=examples_per_class, 
        synthetic_probability=synthetic_probability, 
        synthetic_dir=synthetic_dir,
        generative_aug=aug, seed=seed,
        image_size=(image_size, image_size))

    if num_synthetic > 0 and aug is not None:
        aug_idx = resume[2]
        aug_num = resume[3]
        train_dataset.generate_augmentations(num_synthetic, aug_idx, aug_num)

    train_sampler = torch.utils.data.RandomSampler(
        train_dataset, replacement=True, 
        num_samples=batch_size * iterations_per_epoch)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler=train_sampler, num_workers=num_workers)

    val_dataset = DATASETS[dataset](
        split="val", seed=seed,
        image_size=(image_size, image_size))

    val_sampler = torch.utils.data.RandomSampler(
        val_dataset, replacement=True, 
        num_samples=batch_size * iterations_per_epoch)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, 
        sampler=val_sampler, num_workers=num_workers)

    model = ClassificationModel(
        train_dataset.num_classes, 
        backbone=classifier_backbone,
        train_dataset = train_dataset,
        device = device
    ).to(device)
    
    if classifier_backbone == 'clip':
        optimizer_dict = [
        {
            'params': filter(lambda p: p.requires_grad, model.parameters()),
            'lr': 0.0002
        },
        ]
        optim = torch.optim.AdamW(optimizer_dict, weight_decay=0.1)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    records = []

    for epoch in trange(num_epochs, desc="Training Classifier"):

        model.train()

        epoch_loss = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device=device)
        epoch_accuracy = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device=device)
        epoch_size = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device=device)

        for i, (image, label) in enumerate(train_dataloader):
            image, label = image.to(device), label.to(device)

            logits = model(image)
            prediction = logits.argmax(dim=1)

            loss = F.cross_entropy(logits, label, reduction="none")
            if len(label.shape) > 1: label = label.argmax(dim=1)

            accuracy = (prediction == label).float()

            optim.zero_grad()
            loss.mean().backward()
            optim.step()

            with torch.no_grad():
            
                epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                epoch_loss.scatter_add_(0, label, loss)
                epoch_accuracy.scatter_add_(0, label, accuracy)

        training_loss = epoch_loss / epoch_size.clamp(min=1)
        training_accuracy = epoch_accuracy / epoch_size.clamp(min=1)

        training_loss = training_loss.cpu().numpy()
        training_accuracy = training_accuracy.cpu().numpy()

        model.eval()

        epoch_loss = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device=device)
        epoch_accuracy = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device=device)
        epoch_size = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device=device)

        for image, label in val_dataloader:
            image, label = image.to(device), label.to(device)

            logits = model(image)
            prediction = logits.argmax(dim=1)

            loss = F.cross_entropy(logits, label, reduction="none")
            accuracy = (prediction == label).float()

            with torch.no_grad():
            
                epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                epoch_loss.scatter_add_(0, label, loss)
                epoch_accuracy.scatter_add_(0, label, accuracy)

        validation_loss = epoch_loss / epoch_size.clamp(min=1)
        validation_accuracy = epoch_accuracy / epoch_size.clamp(min=1)

        validation_loss = validation_loss.cpu().numpy()
        validation_accuracy = validation_accuracy.cpu().numpy()

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=training_loss.mean(), 
            metric="Loss", 
            split="Training"
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=validation_loss.mean(), 
            metric="Loss", 
            split="Validation"
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=training_accuracy.mean(), 
            metric="Accuracy", 
            split="Training"
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=validation_accuracy.mean(), 
            metric="Accuracy", 
            split="Validation"
        ))

        for i, name in enumerate(train_dataset.class_names):

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=training_loss[i], 
                metric=f"Loss {name.title()}", 
                split="Training"
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=validation_loss[i], 
                metric=f"Loss {name.title()}", 
                split="Validation"
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=training_accuracy[i], 
                metric=f"Accuracy {name.title()}", 
                split="Training"
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=validation_accuracy[i], 
                metric=f"Accuracy {name.title()}", 
                split="Validation"
            ))

    return records

class ClassificationModel(nn.Module):
    
    def __init__(self, num_classes: int, backbone: str = "clip", train_dataset = None, device = 'cuda:0'):
        
        super(ClassificationModel, self).__init__()

        self.backbone = backbone
        self.image_processor  = None

        if backbone == "resnet50":
        
            self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.out = nn.Linear(2048, num_classes)

        elif backbone == "clip":
            self.base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
            self.clip_labels = [f"a photo of a {label}" for label in train_dataset.class_names]
        
            label_tokens = self.tokenizer(self.clip_labels, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                label_emb = self.base_model.get_text_features(**label_tokens)
                        
            self.out = torch.nn.Linear(label_emb.shape[1], len(self.clip_labels)).to(device)
            self.out.weight.data = label_emb.clone()

    def forward(self, image):
        
        x = image

        if self.backbone == "resnet50":
            
            with torch.no_grad():

                x = self.base_model.conv1(x)
                x = self.base_model.bn1(x)
                x = self.base_model.relu(x)
                x = self.base_model.maxpool(x)

                x = self.base_model.layer1(x)
                x = self.base_model.layer2(x)
                x = self.base_model.layer3(x)
                x = self.base_model.layer4(x)

                x = self.base_model.avgpool(x)
                x = torch.flatten(x, 1)

            return self.out(x)

        elif self.backbone == "clip":
            with torch.no_grad():
                img_emb = self.base_model.get_image_features(image)

            logits_per_image = self.out(img_emb)

            similarity = logits_per_image

            return similarity
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser("Few-Shot Baseline")

    parser.add_argument("--config", type=str, required=True, default=None)
    
    args = parser.parse_args()

    config_name = args.config
    config_file_path = os.path.join(DEFAULT_CONFIGS_DIR, f"{config_name}.py")

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file {config_file_path} not found")
    
    module_name = os.path.splitext(os.path.basename(config_file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, config_file_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    torch.cuda.set_device(0)

    os.makedirs(cfg.logdir, exist_ok=True)

    all_trials = []

    options = product(range(cfg.num_trials), cfg.examples_per_class)
    options = np.array(list(options))

    option_list = options.tolist()

    if cfg.resume != [None, None, None, None]:
        res_seed = cfg.resume[0]
        res_examples_per_class = cfg.resume[1]

        if (res_seed, res_examples_per_class) != (0, 1):
            prev_option = option_list[option_list.index([res_seed, res_examples_per_class]) - 1]
            csv_path = os.path.join(cfg.logdir, f"results_{prev_option[0]}_{prev_option[1]}.csv")
            all_trials = pd.read_csv(csv_path, index_col = 0).to_dict(orient="records")

        option_list = option_list[option_list.index([res_seed, res_examples_per_class]) : ]

    if hasattr(cfg, "num_workers"):
        num_workers = cfg.num_workers
    else:
        num_workers = 4

    if hasattr(cfg, "device"):
        device = cfg.device
    else:
        device = 'cuda'

    if cfg.prompt_mode == "llm":
        if not hasattr(cfg, "llm_prompt_path"):
            raise ValueError("Configuration Error: 'llm_prompt_path' must be specified when 'prompt_mode' is set to 'llm'.")

    for seed, examples_per_class in option_list:

        hyperparameters = dict(
            examples_per_class=examples_per_class,
            seed=seed, 
            dataset=cfg.dataset,
            num_epochs=cfg.num_epochs,
            iterations_per_epoch=cfg.iterations_per_epoch, 
            batch_size=cfg.batch_size,
            model_path=cfg.model_path,
            synthetic_probability=cfg.synthetic_probability, 
            num_synthetic=cfg.num_synthetic, 
            prompt_mode=cfg.prompt_mode,
            llm_prompt_path=cfg.llm_prompt_path,
            aug=cfg.aug,
            guidance_scale=cfg.guidance_scale,
            image_size=cfg.image_size,
            classifier_backbone=cfg.classifier_backbone,
            scale=cfg.scale,
            num_inference_steps=cfg.num_inference_steps,
            num_workers=num_workers,
            device=device,
            resume=cfg.resume)

        synthetic_dir = cfg.synthetic_dir.format(**hyperparameters)

        all_trials.extend(run_experiment(synthetic_dir=synthetic_dir, **hyperparameters))

        path = f"results_{seed}_{examples_per_class}.csv"
        path = os.path.join(cfg.logdir, path)

        pd.DataFrame.from_records(all_trials).to_csv(path)
        print(f"examples_per_class = {examples_per_class} saved to: {path}")

        cfg.resume = [None, None, None, None]