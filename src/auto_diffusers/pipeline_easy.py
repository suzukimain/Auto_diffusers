# Copyright 2026 suzukimain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

# Standard library imports
import os
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from types import MethodType
from typing import Optional, Union
from urllib.parse import urlparse

# Third-party imports
import requests
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionXLInpaintPipeline,
)
from diffusers import __version__ as diffusers_version

# Diffusers core imports
from diffusers.loaders.single_file_utils import (
    VALID_URL_PREFIXES,
    _extract_repo_id_and_weights_name,
    infer_diffusers_model_type,
    load_single_file_checkpoint,
)
from diffusers.pipelines.auto_pipeline import (
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_INPAINT_PIPELINES_MAPPING,
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import logging
from packaging.version import Version

# Flux pipelines (import separately to avoid Flax deprecation warnings)
try:
    from diffusers.pipelines.flux import (
        FluxControlImg2ImgPipeline,
        FluxControlInpaintPipeline,
        FluxControlNetImg2ImgPipeline,
        FluxControlNetInpaintPipeline,
        FluxControlNetPipeline,
        FluxControlPipeline,
        FluxImg2ImgPipeline,
        FluxInpaintPipeline,
        FluxKontextPipeline,
        FluxPipeline,
    )
except ImportError:
    FluxImg2ImgPipeline = None
    FluxInpaintPipeline = None
    FluxPipeline = None
    FluxControlImg2ImgPipeline = None
    FluxControlInpaintPipeline = None
    FluxControlNetImg2ImgPipeline = None
    FluxControlNetInpaintPipeline = None
    FluxControlNetPipeline = None
    FluxControlPipeline = None
    FluxKontextPipeline = None

# AnimateDiff pipelines
from diffusers import (
    AnimateDiffPipeline,
    AnimateDiffSDXLPipeline,
)

# Optional pipelines (newer versions only)
try:
    from diffusers import (
        AuraFlowPipeline,
        ChromaPipeline,
        CogView3PlusPipeline,
        CogView4ControlPipeline,
        CogView4Pipeline,
        HunyuanDiTPipeline,
        Kandinsky3Img2ImgPipeline,
        Kandinsky3Pipeline,
        KandinskyCombinedPipeline,
        KandinskyImg2ImgCombinedPipeline,
        KandinskyInpaintCombinedPipeline,
        KandinskyV22CombinedPipeline,
        KandinskyV22Img2ImgCombinedPipeline,
        KandinskyV22InpaintCombinedPipeline,
        LatentConsistencyModelImg2ImgPipeline,
        LatentConsistencyModelPipeline,
        Lumina2Pipeline,
        LuminaPipeline,
        OvisImagePipeline,
        PixArtAlphaPipeline,
        PixArtSigmaPipeline,
        QwenImageControlNetPipeline,
        QwenImageEditInpaintPipeline,
        QwenImageEditPipeline,
        QwenImageEditPlusPipeline,
        QwenImageImg2ImgPipeline,
        QwenImageInpaintPipeline,
        QwenImageLayeredPipeline,
        QwenImagePipeline,
        SanaPipeline,
        StableDiffusion3ControlNetInpaintingPipeline,
        StableDiffusion3ControlNetPipeline,
        StableDiffusionXLControlNetUnionImg2ImgPipeline,
        StableDiffusionXLControlNetUnionInpaintPipeline,
        StableDiffusionXLControlNetUnionPipeline,
        WanImageToVideoPipeline,
        WanPipeline,
        WanVideoToVideoPipeline,
        ZImageControlNetInpaintPipeline,
        ZImageControlNetPipeline,
        ZImageImg2ImgPipeline,
        ZImageOmniPipeline,
        ZImagePipeline,
    )
except ImportError:
    # Fallback to None for pipelines not available in older versions
    AuraFlowPipeline = None
    ChromaPipeline = None
    CogView3PlusPipeline = None
    CogView4ControlPipeline = None
    CogView4Pipeline = None
    HunyuanDiTPipeline = None
    KandinskyCombinedPipeline = None
    KandinskyImg2ImgCombinedPipeline = None
    KandinskyInpaintCombinedPipeline = None
    KandinskyV22CombinedPipeline = None
    KandinskyV22Img2ImgCombinedPipeline = None
    KandinskyV22InpaintCombinedPipeline = None
    Kandinsky3Img2ImgPipeline = None
    Kandinsky3Pipeline = None
    LatentConsistencyModelImg2ImgPipeline = None
    LatentConsistencyModelPipeline = None
    Lumina2Pipeline = None
    LuminaPipeline = None
    OvisImagePipeline = None
    PixArtAlphaPipeline = None
    PixArtSigmaPipeline = None
    QwenImageControlNetPipeline = None
    QwenImageEditInpaintPipeline = None
    QwenImageEditPipeline = None
    QwenImageEditPlusPipeline = None
    QwenImageImg2ImgPipeline = None
    QwenImageInpaintPipeline = None
    QwenImageLayeredPipeline = None
    QwenImagePipeline = None
    SanaPipeline = None
    StableDiffusion3ControlNetInpaintingPipeline = None
    StableDiffusion3ControlNetPipeline = None
    StableDiffusionXLControlNetUnionImg2ImgPipeline = None
    StableDiffusionXLControlNetUnionInpaintPipeline = None
    StableDiffusionXLControlNetUnionPipeline = None
    WanImageToVideoPipeline = None
    WanPipeline = None
    WanVideoToVideoPipeline = None
    ZImageControlNetInpaintPipeline = None
    ZImageControlNetPipeline = None
    ZImageImg2ImgPipeline = None
    ZImageOmniPipeline = None
    ZImagePipeline = None

from huggingface_hub import hf_api, hf_hub_download, hf_hub_url
from huggingface_hub.file_download import http_get
from huggingface_hub.utils import validate_hf_hub_args

logger = logging.get_logger(__name__)


SINGLE_FILE_MODEL_TYPE_TO_AUTO_PIPELINE_KEY = OrderedDict(
    [
        ("auraflow", "auraflow"),
        ("chroma", "chroma"),
        ("cogview3", "cogview3"),
        ("cogview4", "cogview4"),
        ("cogview4-control", "cogview4-control"),
        ("controlnet", "stable-diffusion-controlnet"),
        ("controlnet_xl", "stable-diffusion-xl-controlnet"),
        ("controlnet_xl_large", "stable-diffusion-xl-controlnet"),
        ("controlnet_xl_mid", "stable-diffusion-xl-controlnet"),
        ("controlnet_xl_small", "stable-diffusion-xl-controlnet"),
        ("controlnet_xl_union", "stable-diffusion-xl-controlnet-union"),
        ("flux-control", "flux-control"),
        ("flux-controlnet", "flux-controlnet"),
        ("flux-depth", "flux"),
        ("flux-dev", "flux"),
        ("flux-fill", "flux"),
        ("flux-kontext", "flux-kontext"),
        ("flux-schnell", "flux"),
        ("hunyuan", "hunyuan"),
        ("kandinsky", "kandinsky"),
        ("kandinsky22", "kandinsky22"),
        ("kandinsky3", "kandinsky3"),
        ("lcm", "lcm"),
        ("lumina", "lumina"),
        ("lumina2", "lumina2"),
        ("pixart-alpha", "pixart-alpha"),
        ("pixart-sigma", "pixart-sigma"),
        ("playground-v2-5", "stable-diffusion-xl"),
        ("qwenimage", "qwenimage"),
        ("qwenimage-controlnet", "qwenimage-controlnet"),
        ("qwenimage-edit", "qwenimage-edit"),
        ("qwenimage-edit-plus", "qwenimage-edit-plus"),
        ("sana", "sana"),
        ("sd3", "stable-diffusion-3"),
        ("sd35_large", "stable-diffusion-3"),
        ("sd35_medium", "stable-diffusion-3"),
        ("sd3-controlnet", "stable-diffusion-3-controlnet"),
        ("v1", "stable-diffusion"),
        ("v2", "stable-diffusion"),
        ("xl_base", "stable-diffusion-xl"),
        ("xl_refiner", "stable-diffusion-xl"),
        ("z-image", "z-image"),
        ("z-image-controlnet", "z-image-controlnet"),
    ]
)

UNSUPPORTED_SINGLE_FILE_MODEL_TYPES = (
    "autoencoder-dc-f128c512",
    "autoencoder-dc-f32c32",
    "autoencoder-dc-f32c32-sana",
    "autoencoder-dc-f64c128",
    "cosmos-1.0-t2w-7B",
    "cosmos-1.0-t2w-14B",
    "cosmos-1.0-v2w-7B",
    "cosmos-1.0-v2w-14B",
    "cosmos-2.0-t2i-2B",
    "cosmos-2.0-t2i-14B",
    "cosmos-2.0-v2w-2B",
    "cosmos-2.0-v2w-14B",
    "hidream",
    "hunyuan-video",
    "instruct-pix2pix",
    "ltx-video",
    "ltx-video-0.9.1",
    "ltx-video-0.9.5",
    "ltx-video-0.9.7",
    "mochi-1-preview",
    "stable_cascade_stage_b",
    "stable_cascade_stage_b_lite",
    "stable_cascade_stage_c",
    "stable_cascade_stage_c_lite",
    "z-image-turbo",
    "z-image-turbo-controlnet",
    "z-image-turbo-controlnet-2.0",
    "z-image-turbo-controlnet-2.1",
)


def _build_single_file_pipeline_mapping(auto_mapping, overrides):
    mapping = OrderedDict()
    for model_type, auto_key in SINGLE_FILE_MODEL_TYPE_TO_AUTO_PIPELINE_KEY.items():
        mapping[model_type] = auto_mapping.get(auto_key)

    for model_type in UNSUPPORTED_SINGLE_FILE_MODEL_TYPES:
        mapping[model_type] = None

    mapping.update(overrides)
    return mapping


_ANIMATEDIFF_OVERRIDES = {
    "animatediff_rgb": AnimateDiffPipeline,
    "animatediff_scribble": AnimateDiffPipeline,
    "animatediff_sdxl_beta": AnimateDiffSDXLPipeline,
    "animatediff_v1": AnimateDiffPipeline,
    "animatediff_v2": AnimateDiffPipeline,
    "animatediff_v3": AnimateDiffPipeline,
}

_ANIMATEDIFF_INPAINT_OVERRIDES = {}
for model_type in _ANIMATEDIFF_OVERRIDES:
    _ANIMATEDIFF_INPAINT_OVERRIDES[model_type] = None

SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING = _build_single_file_pipeline_mapping(
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
    {
        **_ANIMATEDIFF_OVERRIDES,
        "flux-2-dev": FluxPipeline,
        "inpainting": None,
        "inpainting_v2": None,
        "ovis": OvisImagePipeline,
        "upscale": StableDiffusionUpscalePipeline,
        "wan-i2v-14B": WanImageToVideoPipeline,
        "wan-t2v-1.3B": WanPipeline,
        "wan-t2v-14B": WanPipeline,
        "wan-vace-1.3B": WanVideoToVideoPipeline,
        "wan-vace-14B": WanVideoToVideoPipeline,
        "xl_inpaint": None,
        "z-image-omni": ZImageOmniPipeline,
    },
)

SINGLE_FILE_CHECKPOINT_IMAGE2IMAGE_PIPELINE_MAPPING = _build_single_file_pipeline_mapping(
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    {
        **_ANIMATEDIFF_OVERRIDES,
        "flux-2-dev": FluxImg2ImgPipeline,
        "inpainting": None,
        "inpainting_v2": None,
        "qwenimage-layered": QwenImageLayeredPipeline,
        "upscale": StableDiffusionUpscalePipeline,
        "wan-i2v-14B": WanImageToVideoPipeline,
        "wan-t2v-1.3B": None,
        "wan-t2v-14B": None,
        "wan-vace-1.3B": WanVideoToVideoPipeline,
        "wan-vace-14B": WanVideoToVideoPipeline,
        "xl_inpaint": None,
    },
)

SINGLE_FILE_CHECKPOINT_INPAINT_PIPELINE_MAPPING = _build_single_file_pipeline_mapping(
    AUTO_INPAINT_PIPELINES_MAPPING,
    {
        **_ANIMATEDIFF_INPAINT_OVERRIDES,
        "flux-2-dev": FluxInpaintPipeline,
        "inpainting": StableDiffusionInpaintPipeline,
        "inpainting_v2": StableDiffusionInpaintPipeline,
        "upscale": StableDiffusionUpscalePipeline,
        "wan-i2v-14B": None,
        "wan-t2v-1.3B": None,
        "wan-t2v-14B": None,
        "wan-vace-1.3B": None,
        "wan-vace-14B": None,
        "xl_inpaint": StableDiffusionXLInpaintPipeline,
        "z-image-controlnet-inpaint": ZImageControlNetInpaintPipeline,
    },
)

CONFIG_FILE_LIST = [
    "pytorch_model.bin",
    "pytorch_model.fp16.bin",
    "diffusion_pytorch_model.bin",
    "diffusion_pytorch_model.fp16.bin",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.fp16.safetensors",
    "diffusion_pytorch_model.ckpt",
    "diffusion_pytorch_model.fp16.ckpt",
    "diffusion_pytorch_model.non_ema.bin",
    "diffusion_pytorch_model.non_ema.safetensors",
]

DIFFUSERS_CONFIG_DIR = [
    # common pipeline component folders
    "safety_checker",
    "unet",
    "vae",
    "vae_1_0",
    "vae_decoder",
    "vae_encoder",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "scheduler",
    "transformer",
    # legacy / other possible component folders
    "controlnet",
    "clip",
    "encoder",
]

TOKENIZER_SHAPE_MAP = {
    768: [
        "SD 1.4",
        "SD 1.5",
        "SD 1.5 LCM",
        "SDXL 0.9",
        "SDXL 1.0",
        "SDXL 1.0 LCM",
        "SDXL Distilled",
        "SDXL Turbo",
        "SDXL Lightning",
        "PixArt a",
        "Playground v2",
        "Pony",
    ],
    1024: ["SD 2.0", "SD 2.0 768", "SD 2.1", "SD 2.1 768", "SD 2.1 Unclip"],
}


def get_allowed_extensions(allow_unsafe_formats: bool = False) -> list[str]:
    r"""
    Get the list of allowed file extensions based on safety preferences.
    
    Parameters:
        allow_unsafe_formats (`bool`, *optional*, defaults to `False`):
            If True, allows all formats (.safetensors, .ckpt, .bin).
            If False (default), only allows .safetensors format.
    
    Returns:
        `list[str]`: List of allowed file extensions.
    """
    if allow_unsafe_formats:
        return [".safetensors", ".ckpt", ".bin"]
    else:
        return [".safetensors"]


# Default extension list (for backward compatibility)
EXTENSION = [".safetensors", ".ckpt", ".bin"]

CACHE_HOME = os.path.expanduser("~/.cache")
REQUEST_TIMEOUT = 30
ALLOWED_DOWNLOAD_HOSTS = (
    "civitai.com",
    "civitai.green",
    "huggingface.co",
    "hf.co",
)


def _normalize_dtype_kwargs(kwargs):
    kwargs = dict(kwargs)
    if "dtype" in kwargs and "torch_dtype" in kwargs:
        raise ValueError("Pass only one of `dtype` or `torch_dtype`, not both.")

    if "dtype" in kwargs and Version(diffusers_version) < Version("0.39.0"):
        kwargs["torch_dtype"] = kwargs.pop("dtype")
    elif "torch_dtype" in kwargs and Version(diffusers_version) >= Version("0.39.0"):
        kwargs["dtype"] = kwargs.pop("torch_dtype")
    return kwargs


def _pipeline_loading_kwargs(kwargs):
    load_kwargs = dict(kwargs)
    for key in (
        "_max_retries",
        "_pipeline_tag",
        "allow_unsafe_formats",
        "base_model",
        "candidate_index",
        "checkpoint_format",
        "download",
        "gated",
        "include_params",
        "model_type",
        "pipeline_tag",
        "request_timeout",
        "resume",
        "skip_error",
        "sort",
    ):
        load_kwargs.pop(key, None)
    return _normalize_dtype_kwargs(load_kwargs)


def _validate_remote_url(url, allowed_hosts=ALLOWED_DOWNLOAD_HOSTS):
    parsed = urlparse(str(url))
    host = (parsed.hostname or "").lower()

    is_allowed_host = any(
        host == allowed_host or host.endswith(f".{allowed_host}")
        for allowed_host in allowed_hosts
    )

    if parsed.scheme != "https" or not is_allowed_host:
        raise ValueError("Only HTTPS URLs from Civitai or Hugging Face are supported.")


def _safe_filename(file_name):
    if not isinstance(file_name, str) or not file_name or "\x00" in file_name:
        raise ValueError("The remote API returned an invalid filename.")
    if file_name != os.path.basename(file_name) or "/" in file_name or "\\" in file_name:
        raise ValueError("The remote API returned an unsafe filename.")
    return file_name


def _safe_cache_path(cache_dir, *parts):
    cache_root = os.path.abspath(os.fspath(cache_dir))
    path_parts = [os.fspath(part) for part in parts]
    candidate = os.path.abspath(os.path.join(cache_root, *path_parts))
    try:
        contained = os.path.commonpath([cache_root, candidate]) == cache_root
    except ValueError:
        contained = False
    if not contained:
        raise ValueError("The requested cache path is outside the cache directory.")
    return candidate


@dataclass
class RepoStatus:
    r"""
    Data class for storing repository status information.

    Attributes:
        repo_id (`str`):
            The name of the repository.
        repo_hash (`str`):
            The hash of the repository.
        version (`str`):
            The version ID of the repository.
    """

    repo_id: str = ""
    repo_hash: str = ""
    version: str = ""


@dataclass
class ModelStatus:
    r"""
    Data class for storing model status information.

    Attributes:
        search_word (`str`):
            The search word used to find the model.
        download_url (`str`):
            The URL to download the model.
        file_name (`str`):
            The name of the model file.
        local (`bool`):
            Whether the model exists locally
        site_url (`str`):
            The URL of the site where the model is hosted.
    """

    search_word: str = ""
    download_url: str = ""
    file_name: str = ""
    local: bool = False
    site_url: str = ""


@dataclass
class ExtraStatus:
    r"""
    Data class for storing extra status information.

    Attributes:
        trained_words (`str`):
            The words used to trigger the model
    """

    trained_words: Union[list[str], None] = None


@dataclass
class SearchResult:
    r"""
    Data class for storing model data.

    Attributes:
        model_path (`str`):
            The path to the model.
        loading_method (`str`):
            The type of loading method used for the model ( None or 'from_single_file' or 'from_pretrained')
        checkpoint_format (`str`):
            The format of the model checkpoint (`single_file` or `diffusers`).
        repo_status (`RepoStatus`):
            The status of the repository.
        model_status (`ModelStatus`):
            The status of the model.
    """

    model_path: str = ""
    loading_method: Union[str, None] = None
    checkpoint_format: Union[str, None] = None
    repo_status: RepoStatus = field(default_factory=RepoStatus)
    model_status: ModelStatus = field(default_factory=ModelStatus)
    extra_status: ExtraStatus = field(default_factory=ExtraStatus)


@validate_hf_hub_args
def load_pipeline_from_single_file(
    pretrained_model_or_path, pipeline_mapping, **kwargs
):
    r"""
    Instantiate a [`DiffusionPipeline`] from pretrained pipeline weights saved in the `.ckpt` or `.safetensors`
    format. The pipeline is set in evaluation mode (`model.eval()`) by default.

    Parameters:
        pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
            Can be either:
                - A link to the `.ckpt` file (for example
                  `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                - A path to a *file* containing all pipeline weights.
        pipeline_mapping (`dict`):
            A mapping of model types to their corresponding pipeline classes. This is used to determine
            which pipeline class to instantiate based on the model type inferred from the checkpoint.
        dtype (`torch.dtype` or `dict`, *optional*):
            Override the default dtype. `torch_dtype` remains accepted for compatibility.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
            is not used.
        proxies (`dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only load local model weights and configuration files or not. If set to `True`, the model
            won't be downloaded from the Hub.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
            `diffusers-cli login` (stored in `~/.huggingface`) is used.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
            allowed by Git.
        original_config_file (`str`, *optional*):
            The path to the original config file that was used to train the model. If not provided, the config file
            will be inferred from the checkpoint file.
        config (`str`, *optional*):
            Can be either:
                - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                  hosted on the Hub.
                - A path to a *directory* (for example `./my_pipeline_directory/`) containing the pipeline
                  component configs in Diffusers format.
        checkpoint (`dict`, *optional*):
            The loaded state dictionary of the model.
        kwargs (remaining dictionary of keyword arguments, *optional*):
            Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
            class). The overwritten components are passed directly to the pipelines `__init__` method. See example
            below for more information.
    """

    # Load the checkpoint from the provided link or path
    checkpoint = load_single_file_checkpoint(pretrained_model_or_path)

    # Infer the model type from the loaded checkpoint
    model_type = infer_diffusers_model_type(checkpoint)

    # Get the corresponding pipeline class from the explicit overrides or the
    # installed Diffusers AutoPipeline mapping.
    pipeline_class = pipeline_mapping.get(model_type)

    # For tasks not supported by this pipeline
    if pipeline_class is None:
        raise ValueError(
            f"{model_type} is not supported in this pipeline."
            "For `Text2Image`, please use `EasyPipelineForText2Image.from_pretrained`, "
            "for `Image2Image` , please use `EasyPipelineForImage2Image.from_pretrained`, "
            "and `inpaint` is only supported in `EasyPipelineForInpainting.from_pretrained`"
        )

    else:
        # Load the pipeline from the checkpoint
        return pipeline_class.from_single_file(
            pretrained_model_or_path, **_pipeline_loading_kwargs(kwargs)
        )


def get_keyword_types(keyword):
    r"""
    Determine the type and loading method for a given keyword.

    Parameters:
        keyword (`str`):
            The input keyword to classify.

    Returns:
        `dict`: A dictionary containing the model format, loading method,
                and various types and extra types flags.
    """

    # Initialize the status dictionary with default values
    status = {
        "checkpoint_format": None,
        "loading_method": None,
        "type": {
            "other": False,
            "hf_url": False,
            "hf_repo": False,
            "civitai_url": False,
            "local": False,
        },
        "extra_type": {
            "url": False,
            "missing_model_index": None,
        },
    }

    # Check if the keyword is an HTTP or HTTPS URL
    status["extra_type"]["url"] = bool(re.search(r"^(https?)://", keyword))

    is_supported_url = any(
        keyword.startswith(prefix)
        for prefix in VALID_URL_PREFIXES
    )

    # Check if the keyword is a file
    if os.path.isfile(keyword):
        status["type"]["local"] = True
        status["checkpoint_format"] = "single_file"
        status["loading_method"] = "from_single_file"

    # Check if the keyword is a directory
    elif os.path.isdir(keyword):
        status["type"]["local"] = True
        status["checkpoint_format"] = "diffusers"
        status["loading_method"] = "from_pretrained"
        if not os.path.exists(os.path.join(keyword, "model_index.json")):
            status["extra_type"]["missing_model_index"] = True

    # Check if the keyword is a Civitai API download URL
    elif keyword.startswith("https://civitai.com/api/download/"):
        status["type"]["civitai_url"] = True
        status["checkpoint_format"] = "single_file"
        status["loading_method"] = "from_single_file"

    # Check if the keyword is a Civitai URL
    elif keyword.startswith("https://civitai.com/"):
        status["type"]["civitai_url"] = True
        status["checkpoint_format"] = "single_file"
        status["loading_method"] = None

    # Check if the keyword starts with any valid URL prefixes
    elif is_supported_url:
        repo_id, weights_name = _extract_repo_id_and_weights_name(keyword)
        if weights_name:
            status["type"]["hf_url"] = True
            status["checkpoint_format"] = "single_file"
            status["loading_method"] = "from_single_file"
        else:
            status["type"]["hf_repo"] = True
            status["checkpoint_format"] = "diffusers"
            status["loading_method"] = "from_pretrained"

    # Check if the keyword matches a Hugging Face repository format
    elif re.match(r"^[^/]+/[^/]+$", keyword):
        status["type"]["hf_repo"] = True
        status["checkpoint_format"] = "diffusers"
        status["loading_method"] = "from_pretrained"

    # If none of the above apply
    else:
        status["type"]["other"] = True
        status["checkpoint_format"] = None
        status["loading_method"] = None

    return status


def validate_url_with_head(
    url: str, 
    token: Optional[str] = None, 
    headers: Optional[dict] = None,
    timeout: int = 10,
) -> None:
    """
    Validates a URL is accessible by performing a HEAD request.
    Raises HTTPError for any 4xx/5xx status codes (including 401 Unauthorized).
    Logs and continues for other errors.
    
    Parameters:
        url (`str`):
            The URL to validate.
        token (`str`, *optional*):
            Authentication token for the request.
        headers (`dict`, *optional*):
            Dictionary of HTTP Headers to send with the request.
            If provided, takes precedence over token.
        timeout (`int`, *optional*, defaults to `2`):
            Timeout in seconds for the HEAD request.
    
    Raises:
        requests.HTTPError: If the URL returns any 4xx or 5xx status code.
    """
    _validate_remote_url(url)

    if headers is None:
        headers = {}
    else:
        # Create a copy to avoid modifying the original
        headers = dict(headers)
    
    # If token is provided and authorization not already in headers, add it
    # Check case-insensitively since HTTP headers are case-insensitive
    has_authorization_header = any(
        header_name.lower() == "authorization"
        for header_name in headers
    )

    if token and not has_authorization_header:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        response = requests.head(url, headers=headers, allow_redirects=True, timeout=timeout)
        _validate_remote_url(response.url)
        # Raise HTTPError for any 4xx/5xx status codes (including 401)
        response.raise_for_status()
    except requests.HTTPError:
        # Re-raise HTTPError (primarily for 401, but also other 4xx/5xx from raise_for_status)
        raise
    except requests.exceptions.RequestException as e:
        # Log other request errors but don't fail - the actual download may still succeed
        logger.info(f"HEAD check failed (continuing to download): {url} -> {e}")


def file_downloader(
    url,
    save_path,
    **kwargs,
) -> None:
    """
    Downloads a file from a given URL and saves it to the specified path.

    parameters:
        url (`str`):
            The URL of the file to download.
        save_path (`str`):
            The local path where the file will be saved.
        resume (`bool`, *optional*, defaults to `False`):
            Whether to resume an incomplete download.
        headers (`dict`, *optional*, defaults to `None`):
            Dictionary of HTTP Headers to send with the request.
        proxies (`dict`, *optional*, defaults to `None`):
            Dictionary mapping protocol to the URL of the proxy passed to `requests.request`.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether to force the download even if the file already exists.
        displayed_filename (`str`, *optional*):
            The filename of the file that is being downloaded. Value is used only to display a nice progress bar. If
            not set, the filename is guessed from the URL or the `Content-Disposition` header.

    returns:
        None
    """

    _validate_remote_url(url)

    # Get optional parameters from kwargs, with their default values
    resume = kwargs.pop("resume", False)
    headers = kwargs.pop("headers", None)
    proxies = kwargs.pop("proxies", None)
    force_download = kwargs.pop("force_download", False)
    displayed_filename = kwargs.pop("displayed_filename", None)

    # Default mode for file writing and initial file size
    mode = "wb"
    file_size = 0

    # Create directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Check if the file already exists at the save path
    if os.path.exists(save_path):
        existing_size = os.path.getsize(save_path)
        # If file exists and not forcing download
        if not force_download:
            # If the file is non-empty, assume it's already downloaded
            if existing_size > 0:
                logger.info(f"File already exists and is non-empty: {save_path}, skipping download.")
                return None
            # If the file is zero bytes, remove it and proceed to download
            logger.info(f"Found zero-byte file, removing and retrying download: {save_path}")
            try:
                os.remove(save_path)
            except OSError:
                # If removal fails, try to proceed by truncating the file
                with open(save_path, "wb"):
                    pass
        # If resuming is requested and file exists, append
        if resume and os.path.exists(save_path):
            mode = "ab"
            file_size = os.path.getsize(save_path)

    # Try HEAD validation but don't fail if it returns errors (Cloudflare R2 blocks HEAD requests)
    # The actual download attempt will reveal if the URL is truly inaccessible
    try:
        validate_url_with_head(url, headers=headers)
    except requests.HTTPError as e:
        # Log but continue - HEAD requests may fail even when GET succeeds
        status_code = e.response.status_code if hasattr(e, 'response') else 'unknown'
        logger.info(f"HEAD request returned {status_code} for {url}, attempting download anyway...")

    # Open the file in the appropriate mode (write or append)
    with open(save_path, mode) as model_file:
        # Call the http_get function to perform the file download
        result = http_get(
            url=url,
            temp_file=model_file,
            resume_size=file_size,
            displayed_filename=displayed_filename,
            headers=headers,
            proxies=proxies,
        )
        logger.info(f"Successfully downloaded: {url} -> {save_path}")
        return result


def search_huggingface(search_word: str, **kwargs) -> Union[str, SearchResult, None]:
    r"""
    Downloads a model from Hugging Face.

    Parameters:
        search_word (`str`):
            The search query string.
        revision (`str`, *optional*):
            The specific version of the model to download.
        checkpoint_format (`str`, *optional*, defaults to `"single_file"`):
            The format of the model checkpoint.
        download (`bool`, *optional*, defaults to `False`):
            Whether to download the model.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether to force the download if the model already exists.
        include_params (`bool`, *optional*, defaults to `False`):
            Whether to include parameters in the returned data.
        pipeline_tag (`str`, *optional*):
            Tag to filter models by pipeline.
        token (`str`, *optional*):
            API token for Hugging Face authentication.
        gated (`bool`, *optional*, defaults to `False` ):
            A boolean to filter models on the Hub that are gated or not.
        skip_error (`bool`, *optional*, defaults to `False`):
            Whether to skip errors and return None.
        allow_unsafe_formats (`bool`, *optional*, defaults to `False`):
            If True, allows potentially unsafe formats (.ckpt, .bin) in addition to .safetensors.
            If False (default), only allows .safetensors format for maximum security.

    Returns:
        `Union[str,  SearchResult, None]`: The model path or  SearchResult or None.
    """
    # Extract additional parameters from kwargs
    revision = kwargs.pop("revision", None)
    model_type = kwargs.pop("model_type", "checkpoint")
    # normalize model_type for simple comparisons
    try:
        model_type = str(model_type).lower()
    except Exception:
        model_type = ""
    checkpoint_format = kwargs.pop("checkpoint_format", "single_file")
    download = kwargs.pop("download", False)
    force_download = kwargs.pop("force_download", False)
    include_params = kwargs.pop("include_params", False)
    pipeline_tag = kwargs.pop("pipeline_tag", None)
    token = kwargs.pop("token", None)
    gated = kwargs.pop("gated", False)
    cache_dir = kwargs.pop("cache_dir", None)
    skip_error = kwargs.pop("skip_error", False)
    allow_unsafe_formats = kwargs.pop("allow_unsafe_formats", False)
    
    # Get allowed extensions based on safety preference
    allowed_extensions = get_allowed_extensions(allow_unsafe_formats)

    file_list = []
    hf_repo_info = {}
    hf_security_info = {}
    model_path = ""
    repo_id, file_name = "", ""
    candidate_index = kwargs.pop("candidate_index", 0)

    # Get the type and loading method for the keyword
    search_word_status = get_keyword_types(search_word)

    if search_word_status["type"]["hf_repo"]:
        repo_id = search_word
        hf_repo_info = asdict(
            hf_api.model_info(repo_id=search_word, securityStatus=True)
        )
        if download:
            model_path = DiffusionPipeline.download(
                search_word,
                revision=revision,
                token=token,
                force_download=force_download,
                cache_dir=cache_dir,
                **kwargs,
            )
        else:
            model_path = search_word
    elif search_word_status["type"]["hf_url"]:
        repo_id, weights_name = _extract_repo_id_and_weights_name(search_word)
        if download:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=weights_name,
                revision=revision,
                token=token,
                force_download=force_download,
                cache_dir=cache_dir,
            )
        else:
            model_path = search_word
    elif search_word_status["type"]["local"]:
        model_path = search_word
    elif search_word_status["type"]["civitai_url"]:
        if skip_error:
            return None
        else:
            raise ValueError(
                "The URL for Civitai is invalid with `from_huggingface`. Please use `from_civitai` instead."
            )
    else:
        # Get model data from HF API
        hf_models = hf_api.list_models(
            search=search_word,
            sort="downloads",
            limit=100,
            fetch_config=True,
            pipeline_tag=pipeline_tag,
            full=True,
            gated=gated,
            token=token,
        )
        model_dicts = []
        for model in hf_models:
            model_dicts.append(asdict(model))

        # Collect all valid candidates first
        candidates = []
        for repo_info in model_dicts:
            repo_id = repo_info["id"]
            file_list = []
            try:
                hf_repo_info = asdict(
                    hf_api.model_info(repo_id=repo_id, securityStatus=True)
                )
            except Exception as e:
                logger.info(f"Failed to get repo info for {repo_id}: {e}")
                continue
                
            # Lists files with security issues.
            hf_security_info = hf_repo_info.get("security_repo_status") or {}
            if not hf_security_info.get("scansDone"):
                logger.info(f"Skipping {repo_id} because its security scan is incomplete.")
                continue
            exclusion = []
            for issue in hf_security_info.get("filesWithIssues", []):
                if "path" in issue:
                    exclusion.append(issue["path"])

            # Checks for multi-folder diffusers model or valid files (models with security issues are excluded).
            if hf_security_info["scansDone"]:
                for info in repo_info.get("siblings", []):
                    file_path = info["rfilename"]
                    if "model_index.json" == file_path and checkpoint_format in [
                        "diffusers",
                        "all",
                    ]:
                        # Found diffusers model
                        candidates.append({
                            "type": "diffusers",
                            "repo_id": repo_id,
                            "repo_info": hf_repo_info,
                        })
                        break

                    has_allowed_extension = any(
                        file_path.endswith(extension)
                        for extension in allowed_extensions
                    )
                    is_config_file = any(
                        config_name in file_path
                        for config_name in CONFIG_FILE_LIST
                    )
                    has_security_issue = any(
                        excluded_path in file_path
                        for excluded_path in exclusion
                    )

                    parent_directory = os.path.basename(os.path.dirname(file_path))
                    is_diffusers_component = parent_directory in DIFFUSERS_CONFIG_DIR

                    is_checkpoint_auxiliary = False
                    if model_type == "checkpoint":
                        lower_file_path = file_path.lower()
                        auxiliary_markers = ["lora", "textual", "inversion"]
                        is_checkpoint_auxiliary = any(
                            marker in lower_file_path
                            for marker in auxiliary_markers
                        )

                    if (
                        has_allowed_extension
                        and not is_config_file
                        and not has_security_issue
                        and not is_diffusers_component
                        and not is_checkpoint_auxiliary
                    ):
                        file_list.append(file_path)
                
                # Add single file candidates
                if file_list:
                    # Sort and find the safest model
                    file_name = file_list[0]
                    for model_file in sorted(file_list, reverse=True):
                        if re.search(r"(?i)[-_](safe|sfw)", model_file):
                            file_name = model_file
                            break
                    candidates.append({
                        "type": "single_file",
                        "repo_id": repo_id,
                        "file_name": file_name,
                        "repo_info": hf_repo_info,
                    })

        if not candidates:
            # Handle case where no models match the criteria
            if skip_error:
                return None
            else:
                raise ValueError(
                    "No models matching your criteria were found on huggingface."
                )

        # Check if candidate_index is valid
        if candidate_index >= len(candidates):
            if skip_error:
                return None
            else:
                raise ValueError(f"No more candidates available (requested index {candidate_index}, total {len(candidates)})")
        
        # Get the candidate at the specified index
        candidate = candidates[candidate_index]
        repo_id = candidate["repo_id"]
        hf_repo_info = candidate["repo_info"]
        
        try:
            if candidate["type"] == "diffusers":
                # Validate URL early to detect 401 errors
                if download:
                    # Validate URL accessibility before downloading
                    # Check if model_index.json is accessible
                    model_index_url = hf_hub_url(
                        repo_id=repo_id,
                        filename="model_index.json",
                        revision=revision
                    )
                    validate_url_with_head(model_index_url, token=token)
                    
                    model_path = DiffusionPipeline.download(
                        repo_id,
                        token=token,
                        cache_dir=cache_dir,
                        **kwargs,
                    )
                else:
                    model_path = repo_id
                
                file_name = ""
                
            elif candidate["type"] == "single_file":
                file_name = candidate["file_name"]
                
                # Validate URL early to detect 401 errors
                if download:
                    # Validate URL accessibility before downloading
                    # Construct the download URL for validation using hf_hub_url
                    file_url = hf_hub_url(
                        repo_id=repo_id,
                        filename=file_name,
                        revision=revision
                    )
                    validate_url_with_head(file_url, token=token)
                    
                    model_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=file_name,
                        revision=revision,
                        token=token,
                        force_download=force_download,
                        cache_dir=cache_dir,
                    )
                else:
                    model_path = f"https://huggingface.co/{repo_id}/blob/main/{file_name}"
                    
        except Exception as e:
            # Log the error and let the caller decide whether to retry
            logger.info(f"Candidate {candidate_index+1}/{len(candidates)} ({repo_id}) failed: {e}")
            if skip_error:
                return None
            else:
                raise

    # `pathlib.PosixPath` may be returned
    if model_path:
        model_path = str(model_path)

    if file_name:
        download_url = f"https://huggingface.co/{repo_id}/blob/main/{file_name}"
    else:
        download_url = f"https://huggingface.co/{repo_id}"

    output_info = get_keyword_types(model_path)

    if include_params:
        return SearchResult(
            model_path=model_path or download_url,
            loading_method=output_info["loading_method"],
            checkpoint_format=output_info["checkpoint_format"],
            repo_status=RepoStatus(
                repo_id=repo_id, repo_hash=hf_repo_info["sha"], version=revision
            ),
            model_status=ModelStatus(
                search_word=search_word,
                site_url=download_url,
                download_url=download_url,
                file_name=file_name,
                local=download,
            ),
            extra_status=ExtraStatus(trained_words=None),
        )

    else:
        return model_path or download_url


def search_civitai(search_word: str, **kwargs) -> Union[str, SearchResult, None]:
    r"""
    Downloads a model from Civitai.

    Parameters:
        search_word (`str`):
            The search query string.
        model_type (`str`, *optional*, defaults to `Checkpoint`):
            The type of model to search for.
        sort (`str`, *optional*):
            The order in which you wish to sort the results(for example, `Highest Rated`, `Most Downloaded`, `Newest`).
        base_model (`str`, *optional*):
            The base model to filter by.
        download (`bool`, *optional*, defaults to `False`):
            Whether to download the model.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether to force the download if the model already exists.
        token (`str`, *optional*):
            API token for Civitai authentication.
        include_params (`bool`, *optional*, defaults to `False`):
            Whether to include parameters in the returned data.
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored.
        resume (`bool`, *optional*, defaults to `False`):
            Whether to resume an incomplete download.
        skip_error (`bool`, *optional*, defaults to `False`):
            Whether to skip errors and return None.
        allow_unsafe_formats (`bool`, *optional*, defaults to `False`):
            If True, allows potentially unsafe formats (.ckpt, .bin) in addition to .safetensors.
            If False (default), only allows .safetensors format for maximum security.

    Returns:
        `Union[str,  SearchResult, None]`: The model path or ` SearchResult` or None.
    """

    # Extract additional parameters from kwargs
    model_type = kwargs.pop("model_type", "Checkpoint")
    sort = kwargs.pop("sort", None)
    download = kwargs.pop("download", False)
    base_model = kwargs.pop("base_model", None)
    force_download = kwargs.pop("force_download", False)
    token = kwargs.pop("token", None)
    include_params = kwargs.pop("include_params", False)
    resume = kwargs.pop("resume", False)
    cache_dir = kwargs.pop("cache_dir", None)
    skip_error = kwargs.pop("skip_error", False)
    allow_unsafe_formats = kwargs.pop("allow_unsafe_formats", False)
    request_timeout = kwargs.pop("request_timeout", REQUEST_TIMEOUT)
    
    # Get allowed extensions based on safety preference
    allowed_extensions = get_allowed_extensions(allow_unsafe_formats)

    # Initialize additional variables with default values
    model_path = ""
    repo_name = ""
    repo_id = ""
    version_id = ""
    trainedWords = ""
    models_list = []
    selected_repo = {}
    selected_model = {}
    selected_version = {}
    civitai_cache_dir = cache_dir or os.path.join(CACHE_HOME, "Civitai")

    # Handle direct Civitai API download URLs
    if search_word.startswith("https://civitai.com/api/download/"):
        _validate_remote_url(search_word, allowed_hosts=("civitai.com",))
        # Extract model version ID from URL
        match = re.search(r'/models/(\d+)', search_word)
        if match:
            version_id = match.group(1)
            # Try to get model info from API
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            
            try:
                # Get version info to extract model name and ID
                response = requests.get(
                    f"https://civitai.com/api/v1/model-versions/{version_id}",
                    headers=headers,
                    timeout=request_timeout,
                )
                response.raise_for_status()
                version_data = response.json()
                
                repo_name = version_data.get("model", {}).get("name", f"model_{version_id}")
                repo_id = str(version_data.get("modelId", ""))
                trainedWords = version_data.get("trainedWords", [])
                
                # Keep only files that passed both scans and use an allowed extension.
                files = []
                for file_info in version_data.get("files", []):
                    pickle_scan_passed = (
                        file_info.get("pickleScanResult") == "Success"
                    )
                    virus_scan_passed = (
                        file_info.get("virusScanResult") == "Success"
                    )

                    file_name = str(file_info.get("name", ""))
                    has_allowed_extension = any(
                        file_name.endswith(extension)
                        for extension in allowed_extensions
                    )

                    if (
                        pickle_scan_passed
                        and virus_scan_passed
                        and has_allowed_extension
                    ):
                        files.append(file_info)

                if not files:
                    raise ValueError(
                        "The Civitai model version has no file that passed its "
                        "security scans."
                    )

                selected_file = files[0]
                for file_info in files:
                    if file_info.get("primary"):
                        selected_file = file_info
                        break

                file_name = _safe_filename(selected_file["name"])
                    
            except Exception as e:
                if skip_error:
                    return None
                raise ValueError("Could not verify the Civitai model version metadata.") from e
            
            download_url = search_word
            
            if download:
                # Download from the direct URL
                local_path = _safe_cache_path(
                    civitai_cache_dir, str(repo_id), str(version_id), file_name
                )
                
                try:
                    file_downloader(
                        url=download_url,
                        save_path=local_path,
                        resume=resume,
                        force_download=force_download,
                        displayed_filename=file_name,
                        headers=headers,
                    )
                    model_path = local_path
                except Exception as e:
                    if skip_error:
                        return None
                    else:
                        raise ValueError(f"Failed to download from direct URL: {e}")
            else:
                model_path = download_url
            
            output_info = get_keyword_types(model_path)
            
            if not include_params:
                return model_path
            else:
                return SearchResult(
                    model_path=model_path,
                    loading_method=output_info["loading_method"],
                    checkpoint_format=output_info["checkpoint_format"],
                    repo_status=RepoStatus(
                        repo_id=repo_name, repo_hash=repo_id, version=version_id
                    ),
                    model_status=ModelStatus(
                        search_word=search_word,
                        site_url=f"https://civitai.com/models/{repo_id}?modelVersionId={version_id}" if repo_id else download_url,
                        download_url=download_url,
                        file_name=file_name,
                        local=output_info["type"]["local"],
                    ),
                    extra_status=ExtraStatus(trained_words=trainedWords or None),
                )
        else:
            if skip_error:
                return None
            else:
                raise ValueError("Could not extract model version ID from Civitai API URL")

    # Set up parameters and headers for the CivitAI API request
    params = {
        "query": search_word,
        "types": model_type,
        "limit": 100,
    }
    if base_model is not None:
        if not isinstance(base_model, list):
            base_model = [base_model]
        params["baseModel"] = base_model

    if sort is not None:
        params["sort"] = sort

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        # Make the request to the CivitAI API
        response = requests.get(
            "https://civitai.com/api/v1/models",
            params=params,
            headers=headers,
            timeout=request_timeout,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        if skip_error:
            return None
        else:
            raise requests.RequestException("Could not query the Civitai API.") from err
    else:
        try:
            data = response.json()
        except (TypeError, ValueError):
            if skip_error:
                return None
            else:
                raise ValueError("Invalid JSON response")

    repositories = data.get("items", [])

    # Collect all valid candidates instead of selecting one
    all_candidates = []

    for selected_repo in repositories:
        repo_name = selected_repo["name"]
        repo_id = selected_repo["id"]

        # Sort versions within the selected repo by download count
        sorted_versions = sorted(
            selected_repo["modelVersions"],
            key=lambda x: x["stats"]["downloadCount"],
            reverse=True,
        )
        for selected_version in sorted_versions:
            version_id = selected_version["id"]
            trainedWords = selected_version["trainedWords"]
            models_list = []
            # When searching for textual inversion, results other than the values entered for the base model may come up, so check again.
            if base_model is None or selected_version["baseModel"] in base_model:
                for model_data in selected_version["files"]:
                    # Check if the file passes security scans and has a valid extension
                    file_name = model_data["name"]

                    has_allowed_extension = any(
                        file_name.endswith(extension)
                        for extension in allowed_extensions
                    )

                    if (
                        model_data.get("pickleScanResult") == "Success"
                        and model_data.get("virusScanResult") == "Success"
                        and has_allowed_extension
                    ):
                        file_name = _safe_filename(file_name)
                        _validate_remote_url(model_data["downloadUrl"])
                        file_status = {
                            "filename": file_name,
                            "download_url": model_data["downloadUrl"],
                        }
                        models_list.append(file_status)

            if models_list:
                # Sort the models list by filename and find the safest model
                sorted_models = sorted(
                    models_list, key=lambda x: x["filename"], reverse=True
                )
                # Prefer safe/sfw models
                candidate_model = sorted_models[0]
                for model_data in sorted_models:
                    is_safe_variant = re.search(
                        r"(?i)[-_](safe|sfw)", model_data["filename"]
                    )
                    if is_safe_variant:
                        candidate_model = model_data
                        break
                
                # Add to candidates list (skip URL validation which causes 403 errors)
                # Actual download will be attempted and retried recursively
                all_candidates.append({
                    'model': candidate_model,
                    'repo_name': repo_name,
                    'repo_id': repo_id,
                    'version_id': version_id,
                    'trainedWords': trainedWords,
                })
                logger.info(f"Added candidate: {candidate_model['filename']} from {repo_name}")


    # Exception handling when no candidates are found
    if not all_candidates:
        if skip_error:
            return None
        else:
            # Provide helpful error message based on token availability
            if token:
                error_msg = (
                    "No model found. Please try changing the search word or filter criteria "
                    "(e.g., base_model, sort). You can also check if your CivitAI token has "
                    "access to the models you're searching for."
                )
            else:
                error_msg = (
                    "No model found. Please try changing the search word or filter criteria. "
                    "If you're looking for private or exclusive models, provide a CivitAI token: "
                    "search_civitai(search_word, token='your_civitai_token')"
                )
            raise ValueError(error_msg)

    # Select first candidate for initial metadata
    selected = all_candidates[0]
    selected_model = selected['model']
    repo_name = selected['repo_name']
    repo_id = selected['repo_id']
    version_id = selected['version_id']
    trainedWords = selected['trainedWords']
    
    file_name = selected_model["filename"]
    download_url = selected_model["download_url"]

    # Handle file download and setting model information
    if download:
        # Try downloading from all candidates recursively
        download_success = False
        
        logger.info(f"Attempting to download from {len(all_candidates)} candidate(s)...")
        
        for idx, candidate in enumerate(all_candidates):
            try:
                candidate_model = candidate['model']
                file_name = candidate_model["filename"]
                download_url = candidate_model["download_url"]
                candidate_repo_id = candidate['repo_id']
                candidate_version_id = candidate['version_id']
                
                logger.info(f"Attempt {idx + 1}/{len(all_candidates)}: {file_name} from {candidate['repo_name']}")
                
                # The path where the model is to be saved
                model_path = _safe_cache_path(
                    civitai_cache_dir,
                    str(candidate_repo_id),
                    str(candidate_version_id),
                    file_name,
                )
                
                # Download Model File
                file_downloader(
                    url=download_url,
                    save_path=model_path,
                    resume=resume,
                    force_download=force_download,
                    displayed_filename=file_name,
                    headers=headers,
                    **kwargs,
                )
                
                # Update selected metadata on success
                download_success = True
                selected_model = candidate_model
                repo_name = candidate['repo_name']
                repo_id = candidate_repo_id
                version_id = candidate_version_id
                trainedWords = candidate['trainedWords']
                file_name = candidate_model["filename"]
                download_url = candidate_model["download_url"]
                
                logger.info(f"Successfully downloaded: {file_name} from {repo_name}")
                break
                
            except requests.HTTPError as e:
                status_code = e.response.status_code if hasattr(e, 'response') else 'unknown'
                logger.info(f"HTTP error {status_code} for {download_url}, trying next candidate...")
                continue
            except Exception as e:
                logger.info(f"Download failed for {download_url}: {e}, trying next candidate...")
                continue
        
        if not download_success:
            if skip_error:
                return None
            else:
                # Show helpful error message based on token availability
                if token:
                    error_msg = (
                        f"Failed to download any model file after trying {len(all_candidates)} candidate(s). "
                        "All downloads returned errors. Please check your CivitAI token permissions or try different search criteria."
                    )
                else:
                    error_msg = (
                        f"Failed to download any model file after trying {len(all_candidates)} candidate(s). "
                        "If models require authentication, provide a CivitAI token: search_civitai(search_word, token='your_token')"
                    )
                raise ValueError(error_msg)

    else:
        model_path = download_url

    output_info = get_keyword_types(model_path)

    if not include_params:
        return model_path
    else:
        return SearchResult(
            model_path=model_path,
            loading_method=output_info["loading_method"],
            checkpoint_format=output_info["checkpoint_format"],
            repo_status=RepoStatus(
                repo_id=repo_name, repo_hash=repo_id, version=version_id
            ),
            model_status=ModelStatus(
                search_word=search_word,
                site_url=f"https://civitai.com/models/{repo_id}?modelVersionId={version_id}",
                download_url=download_url,
                file_name=file_name,
                local=output_info["type"]["local"],
            ),
            extra_status=ExtraStatus(trained_words=trainedWords or None),
        )


def _load_pipeline_with_retries(cls, pretrained_model_link_or_path, pipeline_mapping, **kwargs):
    """
    Helper function to load a pipeline with retry logic using an iterative loop.
    Handles candidate selection, retries, and error tracking.
    
    Parameters:
        cls: The pipeline class
        pretrained_model_link_or_path: Model identifier or path
        pipeline_mapping: Pipeline mapping for single file checkpoints
        **kwargs: Additional arguments
        
    Returns:
        Loaded pipeline with methods added
    """
    # Extract retry parameters from kwargs
    max_retries = kwargs.pop('_max_retries', 9)
    failed_count = 0
    
    search_kwargs = {
        **kwargs,
        "download": True,
        "include_params": True,
        "skip_error": False,
        "pipeline_tag": kwargs.get("_pipeline_tag"),
    }
    
    # Try each candidate up to max_retries times
    for attempt in range(max_retries + 1):
        # Search for the model on Hugging Face
        try:
            hf_checkpoint_status = search_huggingface(
                pretrained_model_link_or_path,
                candidate_index=attempt,
                **search_kwargs,
            )
        except Exception as e:
            logger.info(f"Candidate {attempt + 1}/{max_retries + 1} search failed: {e}")
            failed_count += 1
            if attempt < max_retries:
                logger.info(f"Trying next candidate (attempt {attempt + 2}/{max_retries + 1})...")
                continue
            else:
                logger.warning(
                    f"Note: {failed_count} model(s) skipped due to errors. "
                    f"To access gated models, provide a token via the 'token' parameter."
                )
                raise
        
        checkpoint_path = hf_checkpoint_status.model_path  # type: ignore
        
        # Check the format of the model checkpoint and load pipeline
        try:
            if hf_checkpoint_status.loading_method == "from_single_file":  # type: ignore
                # Load the pipeline from a single file checkpoint
                pipeline = load_pipeline_from_single_file(
                    pretrained_model_or_path=checkpoint_path,
                    pipeline_mapping=pipeline_mapping,
                    **kwargs,
                )
            else:
                pipeline = cls.from_pretrained(
                    checkpoint_path, **_pipeline_loading_kwargs(kwargs)
                )
            
            logger.info(
                f"checkpoint_path: {hf_checkpoint_status.model_status.download_url}"  # type: ignore
            )
            
            # Success - show warning if some candidates were skipped
            if failed_count > 0:
                logger.warning(
                    f"Note: {failed_count} model(s) skipped due to errors. "
                    f"To access gated models, provide a token via the 'token' parameter."
                )
            
            return add_methods(pipeline)
            
        except Exception as e:
            logger.info(f"Failed to load pipeline: {e}")
            failed_count += 1
            if attempt < max_retries:
                logger.info(f"Trying next candidate (attempt {attempt + 2}/{max_retries + 1})...")
                continue
            else:
                logger.warning(
                    f"Note: {failed_count} model(s) skipped due to errors. "
                    f"To access gated models, provide a token via the 'token' parameter."
                )
                raise


def add_methods(pipeline):
    r"""
    Add methods from `AutoConfig` to the pipeline.

    Parameters:
        pipeline (`Pipeline`):
            The pipeline to which the methods will be added.
    """
    for attr_name in dir(AutoConfig):
        attr_value = getattr(AutoConfig, attr_name)
        if callable(attr_value) and not attr_name.startswith("__"):
            setattr(pipeline, attr_name, MethodType(attr_value, pipeline))
    return pipeline


class AutoConfig:
    def auto_load_textual_inversion(
        self,
        pretrained_model_name_or_path: Union[str, list[str]],
        token: Optional[Union[str, list[str]]] = None,
        base_model: Optional[Union[str, list[str]]] = None,
        tokenizer=None,
        text_encoder=None,
        **kwargs,
    ):
        r"""
        Load Textual Inversion embeddings into the text encoder of [`StableDiffusionPipeline`] (both 🤗 Diffusers and
        Automatic1111 formats are supported).

        Parameters:
            pretrained_model_name_or_path (`str`, `os.PathLike`, `list`, or `dict`):
                Can be either one of the following or a list of them:

                    - Search keywords for pretrained model (for example `EasyNegative`).
                    - A string, the *model id* (for example `sd-concepts-library/low-poly-hd-logos-icons`) of a
                      pretrained model hosted on the Hub.
                    - A path to a *directory* (for example `./my_text_inversion_directory/`) containing the textual
                      inversion weights.
                    - A path to a *file* (for example `./my_text_inversions.pt`) containing textual inversion weights.
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            token (`str` or `list[str]`, *optional*):
                Override the token to use for the textual inversion weights. If `pretrained_model_name_or_path` is a
                list, then `token` must also be a list of equal length.
            text_encoder ([`~transformers.CLIPTextModel`], *optional*):
                Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
                If not specified, function will take self.tokenizer.
            tokenizer ([`~transformers.CLIPTokenizer`], *optional*):
                A `CLIPTokenizer` to tokenize text. If not specified, function will take self.tokenizer.
            weight_name (`str`, *optional*):
                Name of a custom weight file. This should be used when:

                    - The saved textual inversion file is in 🤗 Diffusers format, but was saved under a specific weight
                      name such as `text_inv.bin`.
                    - The saved textual inversion file is in the Automatic1111 format.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        Examples:

        ```py
        >>> from auto_diffusers import EasyPipelineForText2Image

        >>> pipeline = EasyPipelineForText2Image.from_huggingface("stable-diffusion-v1-5")

        >>> pipeline.auto_load_textual_inversion("EasyNegative", token="EasyNegative")

        >>> image = pipeline(prompt).images[0]
        ```

        """
        # 1. Set tokenizer and text encoder
        tokenizer = tokenizer or getattr(self, "tokenizer", None)
        text_encoder = text_encoder or getattr(self, "text_encoder", None)

        # Check if tokenizer and text encoder are provided
        if tokenizer is None or text_encoder is None:
            raise ValueError("Tokenizer and text encoder must be provided.")

        # 2. Normalize inputs
        pretrained_model_name_or_paths = (
            [pretrained_model_name_or_path]
            if not isinstance(pretrained_model_name_or_path, list)
            else pretrained_model_name_or_path
        )

        # 2.1 Normalize tokens
        tokens = [token] if not isinstance(token, list) else token
        if tokens[0] is None:
            tokens = tokens * len(pretrained_model_name_or_paths)

        for check_token in tokens:
            # Check if token is already in tokenizer vocabulary
            if check_token in tokenizer.get_vocab():
                raise ValueError(
                    f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
                )

        expected_shape = text_encoder.get_input_embeddings().weight.shape[
            -1
        ]  # Expected shape of tokenizer

        for search_word in pretrained_model_name_or_paths:
            if isinstance(search_word, str):
                # Update kwargs to ensure the model is downloaded and parameters are included
                _status = {
                    "download": True,
                    "include_params": True,
                    "skip_error": False,
                    "model_type": "TextualInversion",
                }
                # Get tags for the base model of textual inversion compatible with tokenizer.
                # If the tokenizer is 768-dimensional, set tags for SD 1.x and SDXL.
                # If the tokenizer is 1024-dimensional, set tags for SD 2.x.
                if expected_shape in TOKENIZER_SHAPE_MAP:
                    # Retrieve the appropriate tags from the TOKENIZER_SHAPE_MAP based on the expected shape
                    tags = TOKENIZER_SHAPE_MAP[expected_shape]
                    if base_model is not None:
                        if isinstance(base_model, list):
                            tags.extend(base_model)
                        else:
                            tags.append(base_model)
                    _status["base_model"] = tags

                kwargs.update(_status)
                # Search for the model on Civitai and get the model status
                textual_inversion_path = search_civitai(search_word, **kwargs)
                logger.warning(
                    f"textual_inversion_path: {search_word} -> {textual_inversion_path.model_status.site_url}"  # type: ignore
                )

                pretrained_model_name_or_paths[
                    pretrained_model_name_or_paths.index(search_word)
                ] = textual_inversion_path.model_path  # type: ignore

        self.load_textual_inversion(
            pretrained_model_name_or_paths,
            token=tokens,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            **kwargs,
        )

    def auto_load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, dict[str, torch.Tensor]],
        adapter_name=None,
        base_model: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ):
        r"""
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.unet` and
        `self.text_encoder`.

        All kwargs are forwarded to `self.lora_state_dict`.

        See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`] for more details on how the state dict is
        loaded.

        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_unet`] for more details on how the state dict is
        loaded into `self.unet`.

        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_text_encoder`] for more details on how the state
        dict is loaded into `self.text_encoder`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            base_model (`str` or `list[str]`, *optional*):
                Base model tag(s) to filter LoRA models (e.g., "SD 1.5", "SDXL 1.0"). Used when searching Civitai
                to ensure LoRA compatibility with the current pipeline's base model.
            low_cpu_mem_usage (`bool`, *optional*):
                Speed up model loading by only loading the pretrained LoRA weights and not initializing the random
                weights.
            kwargs (`dict`, *optional*):
                See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
        """
        if isinstance(pretrained_model_name_or_path_or_dict, str):
            # Update kwargs to ensure the model is downloaded and parameters are included
            _status = {
                "download": True,
                "include_params": True,
                "skip_error": False,
                "model_type": "LORA",
            }
            if base_model is not None:
                _status["base_model"] = base_model if isinstance(base_model, list) else [base_model]
            kwargs.update(_status)
            # Search for the model on Civitai and get the model status
            lora_path = search_civitai(pretrained_model_name_or_path_or_dict, **kwargs)
            # search_civitai may return a SearchResult, a string (download url or path) or None.
            if isinstance(lora_path, SearchResult):
                logger.warning(f"lora_path: {lora_path.model_status.site_url}")
                logger.warning(f"trained_words: {lora_path.extra_status.trained_words}")
                pretrained_model_name_or_path_or_dict = lora_path.model_path
            elif lora_path is None:
                raise ValueError(
                    f"No LORA model found for {pretrained_model_name_or_path_or_dict}"
                )
            else:
                # fallback when search_civitai returns a plain path/URL string
                logger.warning(f"lora_path: {lora_path}")
                pretrained_model_name_or_path_or_dict = lora_path

        try:
            self.load_lora_weights(
                pretrained_model_name_or_path_or_dict,
                adapter_name=adapter_name,
                **kwargs,
            )
        except ImportError as error:
            if "incompatible version of torchao" in str(error):
                raise ImportError(
                    "The installed torchao version is incompatible with PEFT. "
                    "If torchao quantization is not needed, uninstall torchao and restart the runtime. "
                    "Otherwise install mutually compatible PyTorch, PEFT, and torchao versions."
                ) from error
            raise


class EasyPipelineForText2Image(AutoPipelineForText2Image):
    r"""
    [`EasyPipelineForText2Image`] is a generic pipeline class that instantiates a text-to-image pipeline class. The
    specific underlying pipeline class is automatically selected from either the
    [`~EasyPipelineForText2Image.from_pretrained`], [`~EasyPipelineForText2Image.from_pipe`], [`~EasyPipelineForText2Image.from_huggingface`] or [`~EasyPipelineForText2Image.from_civitai`] methods.

    This class cannot be instantiated using `__init__()` (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """

    config_name = "model_index.json"

    def __init__(self, *args, **kwargs):
        # EnvironmentError is returned
        super().__init__()

    @classmethod
    @validate_hf_hub_args
    def from_huggingface(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A keyword to search for Hugging Face (for example `Stable Diffusion`)
                    - Link to `.ckpt` or `.safetensors` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.safetensors"`) on the Hub.
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            checkpoint_format (`str`, *optional*, defaults to `"single_file"`):
                The format of the model checkpoint.
            pipeline_tag (`str`, *optional*):
                Tag to filter models by pipeline.
            dtype (`torch.dtype` or `dict`, *optional*):
                Override the default dtype. `torch_dtype` remains accepted for compatibility.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            gated (`bool`, *optional*, defaults to `False` ):
                A boolean to filter models on the Hub that are gated or not.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from auto_diffusers import EasyPipelineForText2Image

        >>> pipeline = EasyPipelineForText2Image.from_huggingface("stable-diffusion-v1-5")
        >>> image = pipeline(prompt).images[0]
        ```
        """
        # Update kwargs to ensure the model is downloaded and parameters are included
        _status = {
            "download": True,
            "include_params": True,
            "skip_error": False,
            "pipeline_tag": "text-to-image",
        }
        kwargs.update(_status)

        # Use the helper function to load the pipeline with retries
        kwargs['_max_retries'] = 9
        kwargs['_pipeline_tag'] = 'text-to-image'
        return _load_pipeline_with_retries(
            cls, 
            pretrained_model_link_or_path, 
            SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING,
            **kwargs
        )

    @classmethod
    def from_civitai(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A keyword to search for Hugging Face (for example `Stable Diffusion`)
                    - Link to `.ckpt` or `.safetensors` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.safetensors"`) on the Hub.
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            model_type (`str`, *optional*, defaults to `Checkpoint`):
                The type of model to search for. (for example `Checkpoint`, `TextualInversion`, `LORA`, `Controlnet`)
            base_model (`str`, *optional*):
                The base model to filter by.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            resume (`bool`, *optional*, defaults to `False`):
                Whether to resume an incomplete download.
            dtype (`torch.dtype` or `dict`, *optional*):
                Override the default dtype. `torch_dtype` remains accepted for compatibility.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files.
            device_map (`str` or `dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from auto_diffusers import EasyPipelineForText2Image

        >>> pipeline = EasyPipelineForText2Image.from_huggingface("stable-diffusion-v1-5")
        >>> image = pipeline(prompt).images[0]
        ```
        """
        # Update kwargs to ensure the model is downloaded and parameters are included
        _status = {
            "download": True,
            "include_params": True,
            "skip_error": False,
            "model_type": "Checkpoint",
        }
        kwargs.update(_status)

        # Search for the model on Civitai and get the model status
        checkpoint_status = search_civitai(pretrained_model_link_or_path, **kwargs)
        logger.warning(f"checkpoint_path: {checkpoint_status.model_status.site_url}")  # type: ignore
        checkpoint_path = checkpoint_status.model_path  # type: ignore

        # Load the pipeline from a single file checkpoint
        pipeline = load_pipeline_from_single_file(
            pretrained_model_or_path=checkpoint_path,
            pipeline_mapping=SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING,
            **kwargs,
        )
        return add_methods(pipeline)


class EasyPipelineForImage2Image(AutoPipelineForImage2Image):
    r"""

    [`EasyPipelineForImage2Image`] is a generic pipeline class that instantiates an image-to-image pipeline class. The
    specific underlying pipeline class is automatically selected from either the
    [`~EasyPipelineForImage2Image.from_pretrained`], [`~EasyPipelineForImage2Image.from_pipe`], [`~EasyPipelineForImage2Image.from_huggingface`] or [`~EasyPipelineForImage2Image.from_civitai`] methods.

    This class cannot be instantiated using `__init__()` (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """

    config_name = "model_index.json"

    def __init__(self, *args, **kwargs):
        # EnvironmentError is returned
        super().__init__()

    @classmethod
    @validate_hf_hub_args
    def from_huggingface(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A keyword to search for Hugging Face (for example `Stable Diffusion`)
                    - Link to `.ckpt` or `.safetensors` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.safetensors"`) on the Hub.
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            checkpoint_format (`str`, *optional*, defaults to `"single_file"`):
                The format of the model checkpoint.
            pipeline_tag (`str`, *optional*):
                Tag to filter models by pipeline.
            dtype (`torch.dtype` or `dict`, *optional*):
                Override the default dtype. `torch_dtype` remains accepted for compatibility.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            gated (`bool`, *optional*, defaults to `False` ):
                A boolean to filter models on the Hub that are gated or not.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from auto_diffusers import EasyPipelineForImage2Image

        >>> pipeline = EasyPipelineForImage2Image.from_huggingface("stable-diffusion-v1-5")
        >>> image = pipeline(prompt, image).images[0]
        ```
        """
        # Use the helper function to load the pipeline with retries
        kwargs['_max_retries'] = 9
        kwargs['_pipeline_tag'] = 'image-to-image'
        return _load_pipeline_with_retries(
            cls, 
            pretrained_model_link_or_path, 
            SINGLE_FILE_CHECKPOINT_IMAGE2IMAGE_PIPELINE_MAPPING,
            **kwargs
        )

    @classmethod
    def from_civitai(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A keyword to search for Hugging Face (for example `Stable Diffusion`)
                    - Link to `.ckpt` or `.safetensors` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.safetensors"`) on the Hub.
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            model_type (`str`, *optional*, defaults to `Checkpoint`):
                The type of model to search for. (for example `Checkpoint`, `TextualInversion`, `LORA`, `Controlnet`)
            base_model (`str`, *optional*):
                The base model to filter by.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            resume (`bool`, *optional*, defaults to `False`):
                Whether to resume an incomplete download.
            dtype (`torch.dtype` or `dict`, *optional*):
                Override the default dtype. `torch_dtype` remains accepted for compatibility.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files.
            device_map (`str` or `dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from auto_diffusers import EasyPipelineForImage2Image

        >>> pipeline = EasyPipelineForImage2Image.from_huggingface("stable-diffusion-v1-5")
        >>> image = pipeline(prompt, image).images[0]
        ```
        """
        # Update kwargs to ensure the model is downloaded and parameters are included
        _status = {
            "download": True,
            "include_params": True,
            "skip_error": False,
            "model_type": "Checkpoint",
        }
        kwargs.update(_status)

        # Search for the model on Civitai and get the model status
        checkpoint_status = search_civitai(pretrained_model_link_or_path, **kwargs)
        logger.warning(f"checkpoint_path: {checkpoint_status.model_status.site_url}")  # type: ignore
        checkpoint_path = checkpoint_status.model_path  # type: ignore

        # Load the pipeline from a single file checkpoint
        pipeline = load_pipeline_from_single_file(
            pretrained_model_or_path=checkpoint_path,
            pipeline_mapping=SINGLE_FILE_CHECKPOINT_IMAGE2IMAGE_PIPELINE_MAPPING,
            **kwargs,
        )
        return add_methods(pipeline)


class EasyPipelineForInpainting(AutoPipelineForInpainting):
    r"""

    [`EasyPipelineForInpainting`] is a generic pipeline class that instantiates an inpainting pipeline class. The
    specific underlying pipeline class is automatically selected from either the
    [`~EasyPipelineForInpainting.from_pretrained`], [`~EasyPipelineForInpainting.from_pipe`], [`~EasyPipelineForInpainting.from_huggingface`] or [`~EasyPipelineForInpainting.from_civitai`] methods.

    This class cannot be instantiated using `__init__()` (throws an error).

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.

    """

    config_name = "model_index.json"

    def __init__(self, *args, **kwargs):
        # EnvironmentError is returned
        super().__init__()

    @classmethod
    @validate_hf_hub_args
    def from_huggingface(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A keyword to search for Hugging Face (for example `Stable Diffusion`)
                    - Link to `.ckpt` or `.safetensors` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.safetensors"`) on the Hub.
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            checkpoint_format (`str`, *optional*, defaults to `"single_file"`):
                The format of the model checkpoint.
            pipeline_tag (`str`, *optional*):
                Tag to filter models by pipeline.
            dtype (`torch.dtype` or `dict`, *optional*):
                Override the default dtype. `torch_dtype` remains accepted for compatibility.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.

            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            gated (`bool`, *optional*, defaults to `False` ):
                A boolean to filter models on the Hub that are gated or not.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from auto_diffusers import EasyPipelineForInpainting

        >>> pipeline = EasyPipelineForInpainting.from_huggingface("stable-diffusion-2-inpainting")
        >>> image = pipeline(prompt, image=init_image, mask_image=mask_image).images[0]
        ```
        """
        # Use the helper function to load the pipeline with retries
        kwargs['_max_retries'] = 9
        kwargs['_pipeline_tag'] = 'inpainting'
        return _load_pipeline_with_retries(
            cls, 
            pretrained_model_link_or_path, 
            SINGLE_FILE_CHECKPOINT_INPAINT_PIPELINE_MAPPING,
            **kwargs
        )

    @classmethod
    def from_civitai(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Parameters:
            pretrained_model_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A keyword to search for Hugging Face (for example `Stable Diffusion`)
                    - Link to `.ckpt` or `.safetensors` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.safetensors"`) on the Hub.
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            model_type (`str`, *optional*, defaults to `Checkpoint`):
                The type of model to search for. (for example `Checkpoint`, `TextualInversion`, `LORA`, `Controlnet`)
            base_model (`str`, *optional*):
                The base model to filter by.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            resume (`bool`, *optional*, defaults to `False`):
                Whether to resume an incomplete download.
            dtype (`torch.dtype` or `dict`, *optional*):
                Override the default dtype. `torch_dtype` remains accepted for compatibility.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files.
            device_map (`str` or `dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from auto_diffusers import EasyPipelineForInpainting

        >>> pipeline = EasyPipelineForInpainting.from_huggingface("stable-diffusion-2-inpainting")
        >>> image = pipeline(prompt, image=init_image, mask_image=mask_image).images[0]
        ```
        """
        # Update kwargs to ensure the model is downloaded and parameters are included
        _status = {
            "download": True,
            "include_params": True,
            "skip_error": False,
            "model_type": "Checkpoint",
        }
        kwargs.update(_status)

        # Search for the model on Civitai and get the model status
        checkpoint_status = search_civitai(pretrained_model_link_or_path, **kwargs)
        logger.warning(f"checkpoint_path: {checkpoint_status.model_status.site_url}")  # type: ignore
        checkpoint_path = checkpoint_status.model_path  # type: ignore

        # Load the pipeline from a single file checkpoint
        pipeline = load_pipeline_from_single_file(
            pretrained_model_or_path=checkpoint_path,
            pipeline_mapping=SINGLE_FILE_CHECKPOINT_INPAINT_PIPELINE_MAPPING,
            **kwargs,
        )
        return add_methods(pipeline)
