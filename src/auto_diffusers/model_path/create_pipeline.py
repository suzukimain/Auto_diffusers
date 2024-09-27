import os
import torch

from diffusers import StableDiffusionPipeline

from .mix_class import Config_Mix


class pipeline_setup(Config_Mix):
    def __init__(self):
        self.use_TPU = self.is_TPU()
        super().__init__()
        self.model_path = ""


    def File_search(self):
        """
        only single file
        """

        search_path=""
        paths = []
        for root, dirs, files in os.walk("/"):
            for file in files:
                if any(file.endswith(ext) for ext in self.exts):
                    path = os.path.join(root, file)
                    if path not in self.exclude:
                        if not path.startswith("/root/.cache"):
                            paths.append(path)
        num_path=len(paths)
        if not num_path:
            raise FileNotFoundError("\033[33mModel File not found\033[0m")
        else:
            print(f"{num_path} candidate model files found.")
        for s, path in enumerate(paths, 1):
            print(f"{s}: {path}")
        num = int(input(f"Please enter a number(1〜{num_path}): "))
        if 1 <= num <= len(paths):
            search_path=(paths[num-1])
            print(f"Selected model file: {search_path}\n")
        else:
            raise TypeError(f"\033[33mOnly natural numbers in the following range are valid : (1〜{len(paths)})\033[0m")
        return search_path


    def model_set(self,
                  model_select,
                  auto = True,
                  model_type = "Checkpoint",
                  branch = "main",
                  download: bool = False,
                  path_only: bool=True):
        """
        return:
        if path_only is false
        [model_path:str, {base_model_path: str,from_single_file: bool}]
        """

        if not model_type  in ["Checkpoint", "TextualInversion", "LORA", "Hypernetwork", "AestheticGradient", "Controlnet", "Poses"]:
            raise TypeError(f'Wrong argument. Valid values are "Checkpoint", "TextualInversion", "LORA", "Hypernetwork", "AestheticGradient", "Controlnet", "Poses". What was passed on {model_type}')
        local = True if download else False
        return_dict = {"base_model_path":model_select,
                       "from_dingle_file":False,
                       "local":local,
                       "url_or_path":"",
                       }
        model_path = ""
        file_path = ""
        if model_select in self.model_dict:
            model_path_to_check = self.model_dict[model_select]
            if self.is_url_valid(f"https://huggingface.co/{model_path_to_check}"):
                model_select = model_path_to_check

        if model_select == "search":
            #only file
            model_path = self.File_search()
            return_dict["from_single_file"] = False
            return_dict["url_or_path"] = model_path

        elif model_select.startswith("https://huggingface.co/"):
            if not self.is_url_valid(model_select):
                raise ValueError(self.Error_M1)
            else:
                if download:
                    model_path = self.run_hf_download(model_select)
                    return_dict["from_single_file"] = False
                    return_dict["url_or_path"] = model_path
                else:
                    model_path = model_select
                    return_dict["from_single_file"] = True
                    return_dict["url_or_path"] = model_path

        elif model_select.startswith("https://civitai.com/"):
            #local file
            model_path = self.public_civiai(model_select,
                                            auto,
                                            model_type)
            return_dict["from_single_file"] = True

        elif os.path.isfile(model_select):
            model_path = model_select
            return_dict["from_single_file"] = True
            return_dict["local"] = True

        elif os.path.isdir(model_select):
            if os.path.exists(os.path.join(model_select,self.Config_file)):
                return_dict["model_path"] = model_select
                return_dict["from_single_file"] = False
                return_dict["local"] = True
            else:
                raise FileNotFoundError(f"model_index.json not found in {model_select}")

        elif model_select.count("/") == 1:
            if auto and self.diffusers_model_check(model_select):
                if download:
                    model_path = self.run_hf_download(model_select)
                    return_dict["from_single_file"] = False
                else:
                    model_path = model_select
                    return_dict["from_single_file"] = False
            elif auto and (not self.hf_model_check(model_select)):
                raise ValueError(f'The specified repository could not be found, please try turning off "auto" (model_select:{model_select})')
            else:
                file_path=self.file_name_set(model_select,auto,model_type)
                if file_path == "_hf_no_model":
                    raise ValueError("Model not found")
                elif file_path == "_DFmodel":
                    if download:
                        model_path = self.run_hf_download(model_select)
                        return_dict["from_single_file"] = False
                    else:
                        model_path = model_select
                        return_dict["from_single_file"] = False
                else:
                    hf_model_path=f"https://huggingface.co/{model_select}/blob/{branch}/{file_path}"
                    if download:
                        model_path = self.run_hf_download(hf_model_path)
                        return_dict["from_single_file"] = True

                    else:
                        model_path = hf_model_path
                        return_dict["from_single_file"] = True

        else:
            model_name = self.model_name_search(model_select,auto)
            #self.hf_repo_id = model_name
            #hf->civit
            if not model_name == "_hf_no_model":
                file_path = self.file_name_set(model_name,auto,model_type)
                if model_path == "_DFmodel":
                    if download:
                        model_path = self.run_hf_download(file_path)
                        return_dict["from_single_file"] = False
                    else:
                        model_path = model_name #f"https://huggingface.co/{model_name}"
                        return_dict["from_single_file"] = False

                else:
                    hf_model_path = f"https://huggingface.co/{model_name}/blob/{branch}/{file_path}"
                    if download:
                        model_path = self.run_hf_download(hf_model_path)
                        return_dict["from_single_file"] = True
                    else:
                        model_path = hf_model_path
                        return_dict["from_single_file"] = True


            else:
                model_url, model_path = self.civitai_download(
                    model_select,
                    auto,
                    model_type)

                return_dict["from_single_file"] = True
                return_dict["url_or_path"] = model_url

        if not return_dict["url_or_path"]:
            return_dict["url_or_path"] = model_path
        
        if path_only:
            return model_path
        else:
            return [model_path,return_dict]



    def pipe_status_check(self,pipeline):
        from diffusers.pipelines.stable_diffusion import (
            StableDiffusionSafetyChecker,
            FlaxStableDiffusionSafetyChecker
            )
        from transformers import CLIPImageProcessor
        pipe_class_name_ = pipeline.__class__.__name__
        pipe_type = self.pipeline_metod_type(self.import_on_str(pipe_class_name_,"diffusers"))
        if hasattr(pipeline,"safety_checker"):
            if getattr(pipeline,"safety_checker") is None:
                if pipe_type == "flax":
                    pipeline.safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
                        "CompVis/stable-diffusion-safety-checker", from_pt=True
                    )
                elif pipe_type in ["torch", "onnx"]:
                    pipeline.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                        "CompVis/stable-diffusion-safety-checker"
                    )
        if hasattr(pipeline,"feature_extractor"):
            if getattr(pipeline,"feature_extractor") is None:
                pipeline.feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return pipeline


    def pipe_create(self,
                    model_path,
                    from_single_file):
        if from_single_file:
            #not from_confg
            base_pipe = StableDiffusionPipeline.from_single_file(
                model_path
                ).to(self.device)
        else:
            base_pipe = StableDiffusionPipeline.from_pretrained(
                model_path
                ).to(self.device)

        if self.device == "cuda":
            base_pipe.to(torch_dtype = torch.float16)

        return base_pipe


    def pipeline_task(self,model_select,auto):
        params = None
        model_path,model_dict = self.model_set(model_select,
                                               auto = auto,
                                               download = False)
        #model_path = model_dict["base_model_path"]
        from_single_file = model_dict["from_single_file"]

        update_model_path = self.check_func_hist(key="model_path",value=model_path)

        if self.use_TPU:
            try:
                base_pipe,params = self.Flax_pipe_create(model_path)
            except OSError as a:
                self.logger.debug(a)
                raise OSError("Check your internet connection")

        else:
            try:
                base_pipe = self.pipe_create(model_path, from_single_file)
            except OSError as a:
                self.logger.debug(a)
                raise OSError("Check your internet connection")
        base_pipe = self.pipe_status_check(base_pipe)
        if self.device_type == "cpu" and base_pipe.dtype == torch.float16:
            self.logger.warning("CPU cannot use base_pipe with half precision (torch.float16)")
        return base_pipe, params, model_dict["url_or_path"]