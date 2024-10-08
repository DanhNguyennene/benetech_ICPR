import json
import os
from copy import deepcopy
from operator import itemgetter

import albumentations as A
import numpy as np
from PIL import Image
from tokenizers import AddedToken
from torch.utils.data import Dataset
from transformers import Pix2StructProcessor

TOKEN_MAP = {
  "axes": ["[<axes>]", "[</axes>]"],
  "chart-type": ["[<chart-type>]", "[</chart-type>]"],
  "data-series": ["[<data-series>]", "[</data-series>]"],
  "plot-bb": ["[<plot-bb>]", "[</plot-bb>]"],
  "source": ["[<source>]", "[</source>]"],
  "text": ["[<text>]", "[</text>]"],
  "visual-elements": ["[<visual-elements>]", "[</visual-elements>]"],
  "x-axis": ["[<x-axis>]", "[</x-axis>]"],
  "y-axis": ["[<y-axis>]", "[</y-axis>]"],
  "tick-type": ["[<tick-type>]", "[</tick-type>]"],
  "ticks": ["[<ticks>]", "[</ticks>]"],
  "values-type": ["[<values-type>]", "[</values-type>]"],
  "tick_pt": ["[<tick_pt>]", "[</tick_pt>]"],
  "x": ["[<x>]", "[</x>]"],
  "y": ["[<y>]", "[</y>]"],
  "height": ["[<height>]", "[</height>]"],
  "width": ["[<width>]", "[</width>]"],
  "x0": ["[<x0>]", "[</x0>]"],
  "y0": ["[<y0>]", "[</y0>]"],
  "id": ["[<id>]", "[</id>]"],
  "polygon": ["[<polygon>]", "[</polygon>]"],
  "role": ["[<role>]", "[</role>]"],
  "bos_token" : ["[</s>]"]
}

def tokenize_dict(data: dict, token_mapping: dict):
    def recursive_tokenizer(d):
        if isinstance(d, dict):
            result = ""
            for key, value in d.items():
                start_token, end_token = token_mapping.get(key, (f"<{key}>", f"</{key}>"))
                value_string = recursive_tokenizer(value)
                
                result += f"{start_token}{value_string}{end_token}"
            return result
        elif isinstance(d, list):
            return ''.join(recursive_tokenizer(item) for item in d)
        else:
            return str(d)
    return recursive_tokenizer(data)

def get_processor(cfg):
    """
    load the processor
    """
    processor_path = cfg.model.backbone_path
    print(f"loading processor from {processor_path}")
    processor = Pix2StructProcessor.from_pretrained(processor_path)
    processor.image_processor.is_vqa = False
    processor.image_processor.patch_size = {
        "height": cfg.model.patch_size,
        "width": cfg.model.patch_size
    }
    print("adding new tokens...")
    new_tokens = []
    for _, this_tok in TOKEN_MAP.items():
        for tok in this_tok:
            new_tokens.append(tok)
    new_tokens = sorted(new_tokens)
    tokens_to_add = []
    for this_tok in new_tokens:
        tokens_to_add.append(AddedToken(this_tok, lstrip=False, rstrip=False))
    processor.tokenizer.add_tokens(tokens_to_add)
    return processor

class ICPRDataset(Dataset):
    def __init__(self, cfg, graph_ids,transform=None):

        self.cfg = cfg
        self.data_dir = cfg.competition_dataset.data_dir.rstrip("/")
        self.image_dir = os.path.join(self.data_dir, "train", "images")
        self.annotation_dir = os.path.join(self.data_dir, "train", "annotations")
        self.transform = transform
        self.load_processor()
        self.graph_ids = deepcopy(graph_ids)
    
    def load_processor(self):
        self.processor = get_processor(self.cfg)
    
    def load_image(self, graph_id):
        image_path = os.path.join(self.image_dir, f"{graph_id}.jpg")
        image = Image.open(image_path).convert('RGB')
        return image
    
    def build_output(self, graph_id):

        annotation_path = os.path.join(self.annotation_dir, f"{graph_id}.json")

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        chart_type = ['chart-type']
        
        text = tokenize_dict(annotation, TOKEN_MAP)
        e_string = self.processor.tokenizer.eos_token
        res_text = f"{text}{e_string}"
        return res_text, chart_type
    
    def __len__(self):
        return len(self.graph_ids)

    def __str__(self):
        string = 'ICPR Dataset'
        string += f'\tlen = {len(self)}\n'
        return string
    
    def __getitem__(self, index):
        graph_id = self.graph_ids[index]
        image = self.load_image(graph_id)

        if self.transform is not None:
            image = np.array(image)
            image = self.transform(image=image)["image"]

        try:
            text, chart_type = self.build_output(graph_id)
        except Exception as e:
            print(f"Error in {graph_id}")
            print(e)  # e
            text, chart_type = 'error', 'error_chart'

        p_img = self.processor(
            images=image,
            max_patches=self.cfg.model.max_patches,
            add_special_tokens=True,
        )

     
        p_txt = self.processor(
            text=text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.cfg.model.max_length,
        )

        r = {}
        r['id'] = graph_id
        r['chart_type'] = chart_type
        r['image'] = image
        r['text'] = text
        r['flattened_patches'] = p_img['flattened_patches']
        r['attention_mask'] = p_img['attention_mask']

        try:
            r['decoder_input_ids'] = p_txt['decoder_input_ids']
        except KeyError:
            r['decoder_input_ids'] = p_txt['input_ids']

        try:
            r['decoder_attention_mask'] = p_txt['decoder_attention_mask']
        except KeyError:
            r['decoder_attention_mask'] = p_txt['attention_mask']

        return r
    
def create_train_transforms():
    """
    Returns transformations.

    Returns:
        albumentations transforms: transforms.
    """

    transforms = A.Compose(
        [
            A.OneOf(
                [
                    A.RandomToneCurve(scale=0.3),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.1, 0.2),
                        contrast_limit=(-0.4, 0.5),
                        brightness_by_max=True,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=(-20, 20),
                        sat_shift_limit=(-30, 30),
                        val_shift_limit=(-20, 20)
                    )
                ],
                p=0.5,
            ),

            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                    A.GaussNoise(var_limit=(3.0, 9.0)),
                ],
                p=0.5,
            ),

            A.Downscale(always_apply=False, p=0.1, scale_min=0.90, scale_max=0.99),
        ],

        p=0.5,
    )
    return transforms