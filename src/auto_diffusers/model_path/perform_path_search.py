import os


from .mix_class import Config_Mix


class Search_cls(Config_Mix):
    def __init__(self):
        super().__init__()


    def __call__(
            self,
            seach_word,
            auto=True,
            download=False,
            model_type="Checkpoint",
            branch = "main", 
            single_file_only = False,
            local_file_only = False,
            return_path = True
            ):
        self.seach_word = seach_word
        self.auto = auto
        self.download = download
        self.model_type = model_type
        self.branch = branch
        self.single_file_only = single_file_only
        self.local_file_only = local_file_only
        self.return_path = return_path
        result = self.model_set(
                  model_select = self.seach_word,
                  auto = self.auto,
                  download = self.download,
                  model_type = self.model_type,
                  branch = self.branch,
                  local_file_only = self.local_file_only,
                  return_path = self.return_path
                  )
        return result
        

    def File_search(
            self,
            search_word,
            auto = True
            ):
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
        elif not auto:
            print(f"{num_path} candidate model files found.")
            for s, path in enumerate(paths, 1):
                print(f"{s}: {path}")
            num = int(input(f"Please enter a number(1〜{num_path}): "))
            if 1 <= num <= len(paths):
                search_path=(paths[num-1])
                print(f"Selected model file: {search_path}\n")
            else:
                raise TypeError(f"\033[33mOnly natural numbers in the following range are valid : (1〜{len(paths)})\033[0m")
        else:
            search_path = self.max_temper(
                search_word = search_word,
                search_list = paths
                )
        return search_path


    def model_set(self,
                  model_select,
                  auto = True,
                  download = False,
                  model_type = "Checkpoint",
                  branch = "main",
                  local_file_only = False,
                  return_path = True):
        """
        return:
        if path_only is false
        [model_path:str, {base_model_path: str,single_file: bool}]
        """

        if not model_type  in ["Checkpoint", "TextualInversion", "LORA", "Hypernetwork", "AestheticGradient", "Controlnet", "Poses"]:
            raise TypeError(f'Wrong argument. Valid values are "Checkpoint", "TextualInversion", "LORA", "Hypernetwork", "AestheticGradient", "Controlnet", "Poses". What was passed on {model_type}')
        
        return_dict = {
            "url_or_path":"",
            "search_word":model_select,
            "single_file":False,
            "local":True if download or local_file_only else False,
            "civitai_url":"",
            }
        
        model_path = model_select
        file_path = ""
        if model_select in self.model_dict:
            model_path_to_check = self.model_dict[model_select]
            if self.is_url_valid(f"https://huggingface.co/{model_path_to_check}"):
                model_select = model_path_to_check

        if local_file_only:
            model_path = self.File_search(
                search_word = model_select,
                auto = auto
                )
            return_dict["single_file"] = True
            return_dict["url_or_path"] = model_path

        elif model_select.startswith("https://huggingface.co/"):
            if not self.is_url_valid(model_select):
                raise ValueError(self.Error_M1)
            else:
                if download:
                    model_path, single_file = self.run_hf_download(model_select)
                    return_dict["single_file"] = single_file
                    return_dict["url_or_path"] = model_path
                else:
                    model_path = model_select
                    return_dict["single_file"] = True
                    return_dict["url_or_path"] = model_path

        elif model_select.startswith("https://civitai.com/"):
            #local file
            model_path = self.public_civiai(model_select,
                                            auto,
                                            model_type)
            return_dict["single_file"] = True

        elif os.path.isfile(model_select):
            model_path = model_select
            return_dict["url_or_path"] = model_select
            return_dict["single_file"] = True
            return_dict["local"] = True

        elif os.path.isdir(model_select):
            if os.path.exists(os.path.join(model_select,self.Config_file)):
                model_path = model_select
                return_dict["url_or_path"] = model_select
                return_dict["single_file"] = False
                return_dict["local"] = True
            else:
                raise FileNotFoundError(f"model_index.json not found in {model_select}")

        elif model_select.count("/") == 1:
            if auto and self.diffusers_model_check(model_select):
                if download:
                    model_path,single_file = self.run_hf_download(model_select)
                    return_dict["single_file"] = False
                else:
                    model_path = model_select
                    return_dict["single_file"] = False
            elif auto and (not self.hf_model_check(model_select)):
                raise ValueError(f'The specified repository could not be found, please try turning off "auto" (model_select:{model_select})')
            else:
                file_path=self.file_name_set(model_select,auto,model_type)
                if file_path == "_hf_no_model":
                    raise ValueError("Model not found")
                elif file_path == "_DFmodel":
                    if download:
                        model_path,single_file = self.run_hf_download(model_select)
                        return_dict["single_file"] = False
                    else:
                        model_path = model_select
                        return_dict["single_file"] = False
                else:
                    hf_model_path=f"https://huggingface.co/{model_select}/blob/{branch}/{file_path}"
                    if download:
                        model_path,single_file = self.run_hf_download(hf_model_path)
                        return_dict["single_file"] = single_file

                    else:
                        model_path = hf_model_path
                        return_dict["single_file"] = True

        else:
            model_name = self.model_name_search(model_select,auto)
            #self.hf_repo_id = model_name
            #hf->civit
            if not model_name == "_hf_no_model":
                file_path = self.file_name_set(model_name,auto,model_type)
                if file_path == "_DFmodel":
                    if download:
                        model_path,single_file = self.run_hf_download(model_name)
                        return_dict["single_file"] = False
                    else:
                        model_path = model_name
                        return_dict["single_file"] = False

                else:
                    hf_model_path = f"https://huggingface.co/{model_name}/blob/{branch}/{file_path}"
                    if download:
                        model_path,single_file = self.run_hf_download(hf_model_path)
                        return_dict["single_file"] = single_file
                    else:
                        model_path = hf_model_path
                        return_dict["single_file"] = True

            else:
                model_url, model_path = self.civitai_download(
                    model_select,
                    auto,
                    model_type,
                    download=download)

                return_dict["single_file"] = True
                return_dict["civitai_url"] = model_url
                return_dict["local"] = True if download else False

        #It is not called, but should not be deleted because it is updating the dictionary.
        update_model_path = self.check_func_hist(key="model_path",value=model_path)
        return_dict["url_or_path"] = model_path
        if return_path:
            return model_path
        else:
            return [model_path, return_dict]
        
