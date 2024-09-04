from semantic_aug.generative_augmentation import GenerativeAugmentation
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapter
from transformers import CLIPProcessor, CLIPModel

from diffusers.utils import logging
from PIL import Image

from typing import Tuple, Callable, List

import os
import json
import math
import torch
import random
import numpy as np
import math

def format_name(name):
    return f"{name.replace(' ', '_')}"

DEFALUT_SD14_MODEL_PATH = "CompVis/stable-diffusion-v1-4"

DEFAULT_SD15_IMAGE_ENCODER_PATH = "models/image_encoder"
DEFAULT_SD15_IP_CKPT_PATH = "models/ip-adapter_sd15.bin"
DEFAULT_SD15_VAE_MODEL_PATH = "stabilityai/sd-vae-ft-mse"

DEFAULT_BASE_PROMPT = "a photo of a {name}"

class DALDAAugmentation(GenerativeAugmentation):

    def __init__(self,
                 model_path: str = DEFALUT_SD14_MODEL_PATH,
                 image_encoder_path: str = DEFAULT_SD15_IMAGE_ENCODER_PATH,
                 ip_ckpt_path: str = DEFAULT_SD15_IP_CKPT_PATH,
                 vae_model_path: str = DEFAULT_SD15_VAE_MODEL_PATH,
                 prompt_mode: str = "base",
                 llm_prompt_path: str = None,
                 format_name: Callable = format_name,
                 guidance_scale: float = 7.5,
                 num_inference_steps: int = 30,
                 scale: List = 0.5,
                 device: str = 'cuda:0',
                 examples_per_class: int = None,
                 **kwargs):
        super(DALDAAugmentation, self).__init__()

        self.device = device

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype = torch.float16,
            scheduler = noise_scheduler,
            vae = vae,
            feature_extractor = None,
            safety_checker = None
        )

        self.ip_model = IPAdapter(self.pipe, image_encoder_path, ip_ckpt_path, self.device)

        logging.disable_progress_bar()
        self.pipe.set_progress_bar_config(disable=True)

        if type(scale) == float or type(scale) == int:
            self.scale_mode = "fixed"
            self.scale = scale
        elif scale == "adaptive":
            self.scale_mode = scale
            self.sigma = 0.05 * examples_per_class
            clip_model_id = "openai/clip-vit-base-patch16"
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
            self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(self.device)
        elif scale == "all_random":
            self.scale_mode = scale

        if prompt_mode == "base":
            self.prompt = DEFAULT_BASE_PROMPT
        elif prompt_mode == "llm":
            with open(llm_prompt_path, 'r', encoding='utf-8') as file:
                self.prompt = json.load(file)

        self.prompt_mode = prompt_mode
        self.guidance_scale = guidance_scale
        self.format_name = format_name
        self.num_inference_steps = num_inference_steps
        self.examples_per_class = examples_per_class

    def forward(self, image: Image.Image, label: int, metadata: dict, class_names: list, num: int) -> Tuple[Image.Image, int]:

        class_name = metadata.get("name", "")

        canvas = image.resize((512, 512), Image.BILINEAR)
        name = self.format_name(class_name)
        
        if self.prompt_mode == "base":
            prompt = "best quality, high quality, " + self.prompt.format(name=name)
        elif self.prompt_mode == "llm":
            num_prompt = len(self.prompt[class_name])
            if num % num_prompt == 0:
                random.shuffle(self.prompt[class_name])

            prompt = self.prompt[class_name][num % num_prompt]
            prompt = "best quality, high quality, " + f'{prompt.format(name=name)}'

        if self.scale_mode == "fixed":
            scale = random.choice(self.scale)
            clip_score = None
            threshold_clipscore = None
        elif self.scale_mode == "all_random":
            clip_score = None
            threshold_clipscore = None
            scale = random.uniform(0, 1)
        elif self.scale_mode == "adaptive":
            clip_score = self.get_clip_score(image, class_name, class_names)
            threshold_clipscore = 0.3

            if clip_score <= threshold_clipscore:
                min_val = 0.7
                max_val = 0.9
                mu = min_val + (max_val - min_val) * (1 - clip_score)
                scale = min(max(np.random.normal(mu, self.sigma, 1)[0], min_val), max_val)
            else:
                min_val = 0.1
                max_val = 0.4
                mu = min_val + (max_val - min_val) * (1 - clip_score)
                scale = min(max(np.random.normal(mu, self.sigma, 1)[0], min_val), max_val + 0.1 * math.sqrt(self.examples_per_class))
            threshold_clipscore = mu


        kwargs = dict(
            pil_image = canvas,
            num_samples = 1,
            prompt = prompt, 
            guidance_scale=self.guidance_scale, 
            num_inference_steps = self.num_inference_steps,
            scale = scale
        )

        outputs = self.ip_model.generate(**kwargs)

        canvas = outputs[0].resize(image.size, Image.BILINEAR)

        return canvas, label, prompt, scale, clip_score, threshold_clipscore
    
    def get_clip_score(self, image, class_name, class_names):
        clip_labels = [f"a photo of a {name}" for name in class_names]

        inputs = self.clip_processor(
            text = clip_labels,
            images = image,
            return_tensors = 'pt',
            padding = True
        ).to(self.device)

        with torch.no_grad():
            similarity = self.clip_model(**inputs)
            score = similarity.logits_per_image.softmax(dim=1)

        clip_score_list = score.tolist()[0]
        clip_score_list = list(map(lambda x: round(x, 5), clip_score_list))

        clip_score = clip_score_list[class_names.index(class_name)]

        return clip_score