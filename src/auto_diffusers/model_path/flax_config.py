import os
import re
import torch
import requests
import gc
import jax
import json
from requests import HTTPError
from tqdm.auto import tqdm


from diffusers import FlaxDiffusionPipeline

from ..setup.base_config import Basic_config


class With_Flax(Basic_config):
    def __init__(self):
        super().__init__()

    def sd_to_flax(self,url_or_path):
        hf_path,file_name,model_file_path="","",""
        #from_config or from_single_file
        if os.path.isfile(url_or_path):
            model_file_path=url_or_path

        #from_pretrain
        elif os.path.isdir(url_or_path):
            if os.path.exists(os.path.join(url_or_path, self.Config_file)):
                return url_or_path

            else:
                raise FileNotFoundError(f"model_index.json not found in '{url_or_path}'")
        else:
            raise FileNotFoundError("Invalid dir_path.")

        model_saved_dir = os.path.join(os.path.dirname(model_file_path),"converted")
        os.makedirs(model_saved_dir,exist_ok=True)
        if not os.path.isfile(os.path.join(model_saved_dir,"model_index.json")):
            print("Converting the model...")
            #from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
            is_from_safetensors = self.check_for_safetensors(url_or_path)
            from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
            save_pipeline = download_from_original_stable_diffusion_ckpt(url_or_path, from_safetensors = is_from_safetensors)
            #can not use output format :safetensors
            save_pipeline.save_pretrained(model_saved_dir, safe_serialization = False)
            print("End of model conversion")
            del save_pipeline
        return model_saved_dir


    def Flax_pipe_create(self,url_or_path):
        model_dir_path=self.sd_to_flax(url_or_path)
        model_index_path=os.path.join(model_dir_path,self.Config_file)
        with open(model_index_path, "r") as f:
            pipeline_class_name = json.load(f)["_class_name"]
        #pipeline_class = getattr(diffusers, pipeline_class_name)
        self.logger.info(f"Pipeline class imported: {pipeline_class_name}.")
        try:
            base_pipe,base_params = FlaxDiffusionPipeline.from_pretrained(
                model_dir_path,
                dtype=jax.numpy.bfloat16,
                use_safetensors=True
                )
        except ValueError:
            raise ValueError("Insufficient memory.")
        #from flax.jax_utils import replicate
        #params = replicate(base_params)
        params=base_params
        return base_pipe,params