
from typing import Dict, List, Optional, Union, Tuple, Iterable 
import numpy as np 
from PIL import Image 

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


class PaliGemmaProcessor: 

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_len = num_image_tokens
        self.image_size = image_size

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        # tokens for object detection (bounding boxes)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]

        # tokens for object segmentation 
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for in range(128)
        ]

        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        tokenizer.add_bos_token = False 
        tokenizer.add_eos_token = False 

        self.tokenizer = tokenizer


    def __call__(
        self,
        prompts: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(prompts)} prompts."

        pixel_values = process_images(
            images, 
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # stack list of numpy arrays into a single array of shape: [batch_size, channels, height, width]
        pixel_values = np.stack(pixel_values, axis=0)
        # convert to torch tensor 
        pixel_values = torch.tensor(pixel_values)

        # prepend `self.image_seq_len` number of image tokens to the prompt 
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_len,
                image_token=self.IMAGE_TOKEN
            ) 
            for prompt in prompts 
        ]

        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",    # pt: pytorch tensors 
            padding=padding,
            truncation=truncation
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data

