import re
import requests
import gc
from requests import HTTPError
from tqdm.auto import tqdm

from diffusers import (DiffusionPipeline,
                       StableDiffusionPipeline,
                       FlaxDiffusionPipeline)

from huggingface_hub import hf_hub_download

from ..setup.base_config import Basic_config


class Huggingface(Basic_config):
    def __init__(self):
        super().__init__()
        self.num_prints=20
        self.model_id=""
        self.model_name=""
        self.vae_name=""
        self.model_file=""
        self.input_url=False
        self.diffuser_model=False
        self.check_choice_key = ""
        self.choice_number = -1
        self.file_path_dict={}
        self.special_file=""
        self.hf_repo_id = ""
        #self.model_select = ""


    def repo_name_or_path(self,model_name_or_path):
        pattern = r"([^/]+)/([^/]+)/(?:blob/main/)?(.+)"
        weights_name = None
        repo_id = None
        for prefix in self.VALID_URL_PREFIXES:
            model_name_or_path = model_name_or_path.replace(prefix, "")
        match = re.match(pattern, model_name_or_path)
        if not match:
            return repo_id, weights_name
        repo_id = f"{match.group(1)}/{match.group(2)}"
        weights_name = match.group(3)
        return repo_id, weights_name


    def run_hf_download(self,url_or_path):
        """
        retrun:
        [os.path(str), Single_file_or_not(bool)]
        """
        def _hf_repo_download(path):
            model_path = DiffusionPipeline.download(path)
            return model_path
        
        single_file = True

        if any(url_or_path.startswith(checked) for checked in self.VALID_URL_PREFIXES):
            if not self.is_url_valid(url_or_path):
                raise HTTPError("Invalid URL")
            hf_path, file_name =self.repo_name_or_path(url_or_path)
            self.logger.debug(f"url_or_path:{url_or_path}")
            self.logger.debug(f"hf_path: {hf_path} \nfile_name: {file_name}")
            if hf_path and file_name:
                single_file = True
                model_file_path = hf_hub_download(hf_path, file_name)
            elif hf_path and (not file_name):
                if self.diffusers_model_check(hf_path):
                    single_file = False
                    model_file_path = _hf_repo_download(url_or_path)
                else:
                    raise HTTPError("Invalid hf_path")
            else:
                raise TypeError("Invalid path_or_url")

        #from hf_repo
        elif self.diffusers_model_check(url_or_path):
            self.logger.debug(f"url_or_path: {url_or_path}")
            single_file = False
            model_file_path = _hf_repo_download(url_or_path)
        else:
            self.logger.debug(f"url_or_path:{url_or_path}")
            raise TypeError("Invalid path_or_url")
        return [model_file_path, single_file]


    def model_safe_check(self,model_list) ->str:
        if len(model_list)>1:
           for check_model in model_list:
                match = bool(re.search(r"(?i)[-＿]sfw", check_model))
                if match:
                    return check_model
        return model_list[0]


    def list_safe_check(self,model_list) -> list:
        for check_model in model_list:
            if bool(re.search(r"(?i)[-ー_＿]sfw", check_model)):
                model_list.remove(check_model)
                model_list.insert(0, check_model)
                break
        return model_list


    def diffusers_model_check(self,checked_model: str) -> bool:
        index_url=f"https://huggingface.co/{checked_model}/blob/main/model_index.json"
        return self.is_url_valid(index_url)


    def hf_model_check(self,path) -> bool:
        return self.is_url_valid(f"https://huggingface.co/{path}")


    def data_get(self,path) -> list:
        url = f"https://huggingface.co/api/models/{path}"
        data = requests.get(url).json()
        file_value_list = []
        df_model_bool=False
        #fix error': 'Repo model <repo_id>/<model> is gated. You must be authenticated to access it.
        try:
            siblings=data["siblings"]
        except KeyError:
            return []

        for item in siblings:
            data["siblings"]
            file_path=item["rfilename"]
            #model_index.json outside the root directory is not recognized
            if file_path=="model_index.json" and (not self.single_file_only):
                df_model_bool=True
            elif (
                any(file_path.endswith(ext) for ext in self.exts) and
                not any(file_path.endswith(ex) for ex in self.exclude)
                ):
                file_value_list.append(file_path)
        #↓{df_model,file_value_list}
        self.file_path_dict.update({path:(df_model_bool,file_value_list)})
        return file_value_list


    def hf_model_search(self,
                        model_path,
                        limit_num):
        url = f"https://huggingface.co/api/models"#?search={model_name}"
        params={"search":model_path,"sort":"likes","direction":-1,"limit":limit_num}#"downloads",}
        return requests.get(url,params=params).json()


    def hf_models(self,
                  model_name,
                  limit):
        """
        return:
        repo_model_list,with_like : list
        """
        #self.logger.debug(f"model_name: {model_name}")
        data=self.hf_model_search(model_name,limit)
        final_list = []
        if data:
            for item in data:
                model_id,like,private_value,tag_value = item["modelId"],item["likes"],item["private"],item["tags"]
                if  ("audio-to-audio" not in tag_value and
                    (not private_value)):
                    if self.data_get(model_id):
                        model_dict = {"model_id":model_id,
                                      "like":like,}
                        final_list.append(model_dict)
        else:
            print("No models matching your criteria were found on huggingface.")
            return []
        return final_list


    def model_name_search(self,
                          model_name: str,
                          auto_set: bool,
                          Recursive_execution:bool = False):

        def find_max_like(model_dict_list:list):
            """
            Finds the dictionary with the highest "like" value in a list of dictionaries.

            Args:
                model_dict_list: A list of dictionaries.

            Returns:
                The dictionary with the highest "like" value, or the first dictionary if none have "like".
            """
            max_like = 0
            max_like_dict = None
            for model_dict in model_dict_list:
                if model_dict["like"] > max_like:
                    max_like = model_dict["like"]
                    max_like_dict = model_dict
            return max_like_dict["model_id"] or model_dict_list[0]["model_id"]

        """
        auto_set: bool
        loads the model with the most likes in hugface
        """
        if Recursive_execution:
            limit = 1000
        else:
            limit = 15

        repo_model_list = self.hf_models(model_name,limit)
        model_history = self.check_func_hist(key="hf_model_name",
                                             return_value=True)
        if not auto_set:
            print("\033[34mThe following model paths were found")
            if model_history is not None:
                print(f"Previous Choice: {model_history}")
            print("0.Search civitai")
            for (i,(model_dict)) in enumerate(repo_model_list,1):
                model_name = model_dict["model_id"]
                like = model_dict["like"]
                print(f"{i}.model path: {model_name}, evaluation: {like}")

            if Recursive_execution:
                print("16.Other than above")

            while True:
                try:
                    choice = int(input("Select the model path to use: "))
                except ValueError:
                    print("\033[33mOnly natural numbers are valid.\033[34m")
                    continue
                if choice == 0:
                    return "_hf_no_model"
                elif (not Recursive_execution) and choice>=16 and choice == len(repo_model_list)+1:
                    return self.model_name_search(model_name = model_name,
                                                  auto_set = auto_set,
                                                  Recursive_execution = True)
                elif 1 <= choice <= len(repo_model_list):
                    choice_path_dict = repo_model_list[choice-1]
                    choice_path = choice_path_dict["model_id"]
                    break
                else:
                    print(f"Please enter the numbers 1~{len(repo_model_list)}")

        else:
            if repo_model_list:
                choice_path = find_max_like(repo_model_list)
            else:
                choice_path = "_hf_no_model"

        return choice_path


    def file_name_set_sub(self,model_select,file_value,model_type):
        check_key = f"{model_select}_select"
        if not file_value and (not self.diffuser_model):
            print("\033[31mNo candidates found at huggingface\033[0m")
            res = input("Searching for civitai?: ")
            if res.lower() in ["y","yes"]:
                return "_hf_no_model"
            else:
                raise ValueError("No available files were found in the specified repository")
        elif not file_value:
            print("\033[34mOnly models in Diffusers format found")
            while True:
                result=input("Do you want to use it?[y/n]: ")
                if result.lower() in ["y","yes"]:
                    return "_DFmodel"
                elif result.lower() in ["n","no"]:
                    sec_result=input("Searching for civitai?[y/n]: ")
                    if sec_result.lower() in ["y","yes"]:
                        return "_hf_no_model"
                    elif sec_result.lower() in ["n","no"]:
                        raise ValueError("Processing was stopped because no corresponding model was found.")
                else:
                    print("Please enter only [y,n]")

        file_value=self.list_safe_check(file_value)
        if len(file_value)>=self.num_prints: #15
            start_number="1"
            choice_history = self.check_func_hist(key = check_key,return_value=True)
            if choice_history:
                if choice_history>self.num_prints+1:
                    choice_history = self.num_prints+1
                print(f"\033[33m＊Previous number: {choice_history}\033[0m")

            if self.diffuser_model:
                start_number="0"
                print("\033[34m0.Use Diffusers format model")
            for i in range(self.num_prints):
                print(f"\033[34m{i+1}.File name: {file_value[i]}\033[0m")
            print(f"\033[34m{self.num_prints+1}.Other than the files listed above (all candidates will be displayed)\n")
            while True:
                choice = input(f"select the file you want to use({start_number}~21): ")
                try:
                    choice=int(choice)
                except ValueError:
                    print("\033[33mOnly natural numbers are valid\033[34m")
                    continue
                if self.diffuser_model and choice==0:
                    old_num=None
                    self.input_url=False
                    self.choice_number = -1
                    print("\033[0m",end="")
                    choice_history_update = self.check_func_hist(key=check_key,value=choice,update=True)
                    return "_DFmodel"

                elif choice==(self.num_prints+1): #other_file
                    break
                elif 1<=choice<=self.num_prints:
                    self.input_url=True
                    old_num=choice
                    choice_path=file_value[choice-1]
                    self.choice_number = choice
                    print("\033[0m",end="")
                    choice_history_update = self.check_func_hist(key=check_key,value=choice,update=True)
                    return choice_path
                else:
                    print(f"\033[33mPlease enter numbers from 1~{self.num_prints}\033[34m")
            print("\033[0m",end="")
            print("\n\n")

        choice_history = self.check_func_hist(key = check_key,return_value=True)
        if choice_history:
            print(f"\033[33m＊Previous number: {choice_history}\033[0m")

        start_number="1"
        if self.diffuser_model:
            start_number="0"
            print("\033[34m0.Use Diffusers format model\033[0m")
        for i, file_name in enumerate(file_value, 1):
            print(f"\033[34m{i}.File name: {file_name}")
        while True:
            choice = input(f"Select the file you want to use({start_number}~{len(file_value)}): ")
            try:
                choice=int(choice)
            except ValueError:
                print("\033[33mOnly natural numbers are valid\033[34m")
            else:
                if self.diffuser_model and choice==0:
                    self.input_url=False
                    print("\033[0m",end="")
                    self.choice_number = -1
                    choice_history_update = self.check_func_hist(key=check_key,value=choice,update=True)
                    return "_DFmodel"
                if 1<=choice<=len(file_value):
                    self.input_url=True
                    old_num=choice
                    choice_path=file_value[choice-1]
                    self.choice_number = choice
                    print("\033[0m",end="")
                    choice_history_update = self.check_func_hist(key=check_key,value=choice,update=True)
                    return choice_path
                else:
                    print(f"\033[33mPlease enter numbers from 1~{len(file_value)}\033[34m")


    def file_name_set(
            self,
            model_select,
            auto,
            model_type="Checkpoint",
            download=False,
            single_file_only=False
            ):
        
        if self.diffusers_model_check(model_select) and model_type=="Checkpoint":
            self.diffuser_model=True
        #check_choice_key = f"model_select_{model_type}"
        url = f"https://huggingface.co/api/models/{model_select}"
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            raise HTTPError("A hugface login or token is required")
        data = response.json()
        choice_path=""
        file_value = []
        siblings = data["siblings"]
        if data:
            for item in siblings:
                fi_path=item["rfilename"]
                if (any(fi_path.endswith(ext) for ext in self.exts) and
                    (not any(fi_path.endswith(ex) for ex in self.exclude)) and
                    (not any(fi_path.endswith(st) for st in self.config_file_list))):
                    file_value.append(fi_path)
        else:
            raise ValueError("No available file was found.\nPlease check the name.")
        if file_value:
            file_value=self.sort_by_version(file_value)
            if not auto:
                print("\033[34mThe following model files were found\033[0m")
                choice_path=self.file_name_set_sub(model_select,file_value,model_type)
                #if not self.choice_number == -1:
                #    choice_key_update = self.check_func_hist(key=check_key,value=self.choice_number)
            else:
                if self.diffuser_model and (not self.single_file_only):
                    self.input_url=False
                    choice_path="_DFmodel"
                    
                else:
                    self.input_url=True
                    choice_path=self.model_safe_check(file_value)


        elif self.diffuser_model:
            if not auto:
                self.logger.warning("\033[32mOnly models in Diffusers format found\033[0m")
                choice_path = "_DFmodel"
        else:
            raise FileNotFoundError("No available files found in the specified repository")
        #if model_type!="Checkpoint" and model_type!="_DFmodel":
            #self.input_url=False
        if download and not choice_path=="_DFmodel":
            choice_path=hf_hub_download(repo_id=model_select, filename=choice_path)
        #if not self.choice_number== -1:
        #    choice_key_update = self.check_func_hist(key=check_choice_key,value=self.choice_number)
        return choice_path
