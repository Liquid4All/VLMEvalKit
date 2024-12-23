from pathlib import Path

import devkit
import re
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from liquid_vlm.data.collate import collate
from liquid_vlm.data.datasets.image_dataset import (
    ImageDataInstance,
    ImageDataset,
    ImageDatasetConfig,
)
from liquid_vlm.models.vlm import VLM
from liquid_vlm.utils.defaults import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TOKENIZER,
)
from liquid_vlm.utils.misc import remap_hf_to_native_keys

devkit.runtime.wrappers.set_activation_checkpointing(False)
devkit.runtime.wrappers.set_sharding(False)

from .base import BaseModel
from ..smp import splitlen, listinstr
from PIL import Image

def convert_format(input_data):
    """
    Converts from the MMBench format to the conversation format.
    
    Args:
        input_data (list): List of dictionaries in MMBench format
        
    Returns:
        dict: Converted data in conversation format
    """
    # Initialize output structure
    output = {
        "conversation": [],
        "images": []
    }
    
    # Extract question text and images
    question_text = ""
    
    for item in input_data:
        if item['type'] == 'image':
            output['images'].append(item['value'])
        elif item['type'] == 'text':
            question_text = item['value']
    
    # Create conversation message
    message = {
        "role": "user",
        "content": question_text
    }
    
    # If there are images, prepend "<image>" tag to the content
    if output['images']:
        message['content'] = "<image> " + message['content']
    
    output['conversation'].append(message)
    output['id'] = "0"
    return output

class LIQUID_3Bv(BaseModel):
    def __init__(self, model_path, **kwargs):
        super().__init__()
        # Assuming the rest of your initialization code here...
        vlm = VLM()

        if isinstance(model_path, dict):
            if "state_dict_path" in model_path and "lfm_path" in model_path:
                # New initialization method
                state_dict = torch.load(model_path["state_dict_path"])
                
                # Load projector
                proj_state_dict = {
                    k.replace("mm_projector._model", "model"): v
                    for k, v in state_dict["module"].items()
                    if "mm_projector._model" in k
                }
                vlm.proj.load_state_dict(proj_state_dict, strict=True)
                print("Projector loaded")

                # Load LFM
                files = Path(model_path["lfm_path"]).glob("*.safetensors")
                lfm_state_dict = {}
                for file in files:
                    with safe_open(file, framework="pt") as state:
                        for key in state.keys():
                            lfm_state_dict[key] = state.get_tensor(key)
                
                lfm_state_dict = remap_hf_to_native_keys(lfm_state_dict, vlm.lfm.state_dict())
                vlm.lfm.load_state_dict(lfm_state_dict, strict=True)
                print("LFM loaded")

            elif "vlm_path" in model_path:
                # Old initialization method
                files = Path(model_path["vlm_path"]).glob("*.safetensors")
                lfm_state_dict = {}
                for file in files:
                    with safe_open(file, framework="pt") as state:
                        for key in state.keys():
                            lfm_state_dict[key] = state.get_tensor(key)

                # Rename base_lm.* to lfm.* in state dict keys
                lfm_state_dict = {k
                    .replace("mm_projector.", "proj.model.")
                    .replace("base_lm.model.", "lfm.")
                    .replace("base_lm.", "lfm.")
                    .replace("mm_encoder.", "enc.model.")
                    .replace(".ssm.", ".operator.ssm.")
                    .replace("lm_head", "embedding.to_logits")
                    .replace("lfm.embedding_norm", "lfm.embedding.embedding_norm")
                    .replace("lfm.embed_tokens", "lfm.embedding.embedding")
                    : v for k, v in lfm_state_dict.items()}
                
                layers_with_attn = list(set([int(re.search(r'lfm\.layers\.(\d+)\.self_attn', key).group(1)) 
                            for key in lfm_state_dict.keys() 
                            if re.search(r'lfm\.layers\.(\d+)\.self_attn', key)]))
                lfm_state_dict = {
                    re.sub(r'lfm\.layers\.(\d+)\.self_attn\.', r'lfm.layers.\1.operator.', k): v
                    for k, v in lfm_state_dict.items()
                }
                lfm_state_dict = {
                    re.sub(r'lfm\.layers\.(\d+)\.operator.q_layernorm\.', r'lfm.layers.\1.operator.bounded_attention.q_layernorm.', k): v
                    for k, v in lfm_state_dict.items()
                }
                lfm_state_dict = {
                    re.sub(r'lfm\.layers\.(\d+)\.operator.k_layernorm\.', r'lfm.layers.\1.operator.bounded_attention.k_layernorm.', k): v
                    for k, v in lfm_state_dict.items()
                }
                lfm_state_dict = {
                    re.sub(r'lfm\.layers\.(\d+)\.operator_norm\.', r'lfm.layers.\1.attention_norm.', k): v
                    for k, v in lfm_state_dict.items()
                }

                vlm.load_state_dict(lfm_state_dict, strict=True)
                print("LFM loaded")
            else:
                raise ValueError("model_path dict must contain either 'vlm_path' or both 'state_dict_path' and 'lfm_path'")
        else:
            raise ValueError("model_path must be a dictionary")

        # Common initialization steps
        self.device = torch.device("cuda")
        vlm.to(self.device)
        vlm.eval()
        self.vlm = vlm  # Store the initialized model
        
        kwargs_default = {'max_new_tokens': 512}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

    def generate_inner(self, message, dataset=None):
        # DATA
        try:
            image_dataset_config = ImageDatasetConfig(
                metadata_paths=[],
                add_system_prompt=True,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
            )
            dataset = ImageDataset(image_dataset_config)

            data = convert_format(message)
            sample_instance = ImageDataInstance(**data)
            dataset.metadata = [sample_instance]
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)

            # GENERATE
            batch = next(iter(dataloader))
            output = self.vlm.cached_generate(
                    batch["batch"],
                    tokenizer=tokenizer,
                    device=self.device,
                    max_tokens=self.kwargs["max_new_tokens"],
                    temperature=None,
                )
            output = output.strip()
        except Exception as e:
            print("Error in generation: ", e)
            return "A. An error occurred while generating the response. Please try again later."

        print(repr(output))
        return output

