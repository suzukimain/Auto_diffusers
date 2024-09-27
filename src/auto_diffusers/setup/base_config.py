import os
import jax
import inspect
import difflib

import diffusers
from natsort import natsorted

from .device_config import device_set
from .data_class import data_config
from ..checker.memorize_config import config_check
from .runtime_config import Runtime_func
from ..utils.get_custom_logger import custom_logger

class Basic_config(  
    data_config,
    config_check,
    Runtime_func,
    device_set
    ):

    def __init__(self):
        super().__init__()
        self.device_count = self.count_device()
        self.device_type = self.device_type_check()
        self.device = self.extra_device_set()
        self.use_TPU = self.is_TPU()
        self.logger = custom_logger()


    @classmethod
    def get_inherited_class(cls,class_name) -> list:
        inherited_class = inspect.getmro(class_name)
        return [cls_method.__name__ for cls_method in inherited_class]


    def get_item(self,dict_obj):
        """
        Returns the first element of the dictionary
        """
        return next(iter(dict_obj.items()))[1]


    def pipe_class_type(
            self,
            class_name
            ):
        """
        Args:
        class_name : class

        Returns:
        Literal['txt2img','img2img','txt2video']
        """
        _txt2img_method_list = [] #else
        _img2img_method_list = ["image"]
        _img2video_method_list = ["video_length","fps"]

        call_method = self.get_call_method(class_name,method_name = '__call__')

        if any(method in call_method for method in _img2video_method_list):
            pipeline_type = "txt2video"
        elif any(method in call_method for method in _img2img_method_list):
            pipeline_type = "img2img"
        else:
            pipeline_type = "txt2img"
        return pipeline_type


    def pipeline_metod_type(self,Target_class) -> str:
        """
        Args:
        Target_class : class

        Returns:
        Literal['torch','flax','onnx']
        """
        torch_list=["DiffusionPipeline",
                    "AutoPipelineForText2Image",
                    "AutoPipelineForImage2Image",
                    "AutoPipelineForInpainting",]

        flax_list = ["FlaxDiffusionPipeline",]

        if isinstance(Target_class,str):
            Target_class = getattr(diffusers, Target_class)

        cls_method= self.get_inherited_class(Target_class)

        if any(method in torch_list for method in cls_method):
            class_type= "torch"
        elif any(method in flax_list for method in cls_method):
            class_type= "flax"
        else:
            class_type= "onnx"
        return class_type


    def key_check(self,keyword) -> bool:
        global key_dict
        if "key_dict" not in globals():
            key_dict = {}
        key = str(keyword)
        key_in = False
        if key in key_dict:
            if keyword == key_dict[key]:
                key_in = True
        key_dict[key] = keyword
        return key_in


    def get_call_method(
            self,
            class_name,
            method_name : str = '__call__'
            ) ->list:
        """
        Acquire the arguments of the function specified by 'method_name'
        for the class specified by 'class_name'
        """
        if isinstance(class_name,str):
            class_name = getattr(getattr(diffusers, class_name),method_name)
        parameters = inspect.signature(class_name).parameters
        arg_names = []
        for param in parameters.values():
            arg_names.append(param.name)
        return arg_names


    def get_class_elements(
            self,
            search
            ):
        return list(search.__class__.__annotations__.keys())


    def check_for_safetensors(
            self,
            path
            ):
        _ext = os.path.basename(path).split(".")[-1]
        if _ext == "safetensors":
            return True
        else:
            return False


    def pipe_class_type(
            self,
            class_name
            ):
        """
        Args:
        class_name : class

        Returns:
        Literal['txt2img','img2img','txt2video']
        """
        _txt2img_method_list = [] #else
        _img2img_method_list = ["image"]
        _img2video_method_list = ["video_length","fps"]

        call_method = self.get_call_method(class_name,method_name = '__call__')

        if any(method in call_method for method in _img2video_method_list):
            pipeline_type = "txt2video"
        elif any(method in call_method for method in _img2img_method_list):
            pipeline_type = "img2img"
        else:
            pipeline_type = "txt2img"
        return pipeline_type


    def import_on_str(
            self,
            desired_function_or_class,
            module_name = ""
            ):
        if not module_name:
            import_object = __import__(desired_function_or_class)
        else:
            import_object = getattr(__import__(module_name), desired_function_or_class)
        return import_object


    def max_temper(
            self,
            search_word,
            search_list
            ):
        return difflib.get_close_matches(search_word, search_list,cutoff=0, n=1)


    def sort_list_obj(
            self,
            list_obj,
            need_txt
            ):
        sorted_list=[]
        for module_obj in list_obj:
            if need_txt.lower() in module_obj.lower():
                sorted_list.append(module_obj)
        return sorted_list


    def sort_by_version(self,sorted_list) -> list:
        """
        Returns:
        Sorted by version in order of newest to oldest
        """
        return natsorted(sorted_list,reverse = True)


