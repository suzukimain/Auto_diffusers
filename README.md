<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/suzukimain/auto_diffusers/master/images/auto_diffusers_icon.png"/>
    <br>
<p>

<p>
    <a href="https://github.com/suzukimain/auto_diffusers/blob/main/LICENSE"><img alt="GitHub release" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=social"></a>
</p>
<p>
    <a href="https://pepy.tech/project/auto_diffusers"><img alt="GitHub release" src="https://static.pepy.tech/badge/auto_diffusers"></a>
    <a href="https://github.com/suzukimain/auto_diffusers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/suzukimain/auto_diffusers.svg"></a>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=suzukimain.auto_diffusers" alt="Visitor Badge">
</p>


>CONTENTS
+ [About The Project](#About_The_Project)
+ [How to use](#How_to_use)
+ [Search Hubs](#Search_Civitai_and_Huggingface)
  - [Search Civitai](#Search_Civitai)
  - [Search Huggingface](#Search_Huggingface)
+ [License](#License)
+ [Acknowledgement](#Acknowledgement)


## About The Project<a name = "About_The_Project"></a>

Enhance the functionality of diffusers.  
* Search models from Hugging Face and Civitai.  
* Officially listed in the [Hugging Face Diffusers Community Projects](https://huggingface.co/docs/diffusers/community_projects) documentation.

<details>
<summary>Image of the listing location on Hugging Face</summary>

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/suzukimain/auto_diffusers/master/images/huggingface_highlight.png"/>
    <br>
<p>


</details>

## Regarding the Civitai API bug

The previously reported Civitai API issue has largely been resolved.

<details>
<summary>Previous issue description</summary>

Due to recent specification changes in the Civitai API, we were aware of the following issues:
- Reduced search accuracy
- Some search results not being returned

A temporary workaround has been implemented for the above issues, but since certain search results are still not returned, there is a possibility that some models may not be found.

link： [civitai/civitai#1757](https://github.com/civitai/civitai/issues/1757)

</details>

## How to use<a name = "How_to_use"></a>

```python
pip install --quiet auto_diffusers
```

```python
import torch
from auto_diffusers import EasyPipelineForText2Image

# Load an official Hugging Face model
pipe = EasyPipelineForText2Image.from_huggingface(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16,
).to("cuda")
img = pipe("cat").images[0]
img.save("cat.png")


# Search for Civitai
pipe = EasyPipelineForText2Image.from_civitai(
    "search_word",
    dtype=torch.float16,
).to("cuda")
image = pipe("cat").images[0]
image.save("cat.png")

```

Use `dtype` for new code. `torch_dtype` remains supported for compatibility, but do not pass both arguments at the same time.

## Search Civitai and Huggingface<a name = "Search_Civitai_and_Huggingface"></a>

```python
# Load Lora into the pipeline.
pipe.auto_load_lora_weights("Detail Tweaker")

# Load TextualInversion into the pipeline.
pipe.auto_load_textual_inversion("EasyNegative", token="EasyNegative")
```

### Search Civitai<a name = "Search_Civitai"></a>

> [!TIP]
> **If an error occurs, insert the `token` and run again.**

#### `EasyPipeline.from_civitai` parameters

| Name            | Type                   | Default       | Description                                                                         |
|:---------------:|:----------------------:|:-------------:|:-----------------------------------------------------------------------------------:|
| search_word     | string, Path           | required      | The search query string. Can be a keyword, Civitai URL, local directory or file path. |
| model_type      | string                 | `Checkpoint`  | The type of model to search for. <br>(for example `Checkpoint`, `TextualInversion`, `Controlnet`, `LORA`, `Hypernetwork`, `AestheticGradient`, `Poses`) |
| base_model      | string                 | None          | Trained model tag (for example `SD 1.5`, `SD 3.5`, `SDXL 1.0`)                     |
| dtype           | string, torch.dtype    | None          | Override the default `torch.dtype` and load the model with another dtype.            |
| force_download  | bool                   | False         | Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist. |
| cache_dir       | string, Path           | None          | Path to the folder where cached files are stored.                                    |
| resume          | bool                   | False         | Whether to resume an incomplete download.                                            |
| token           | string                 | None          | API token for Civitai authentication.                                                |


#### `search_civitai` parameters

| Name            | Type           | Default       | Description                                                                         |
|:---------------:|:--------------:|:-------------:|:-----------------------------------------------------------------------------------:|
| search_word     | string, Path   | required      | The search query string. Can be a keyword, Civitai URL, local directory or file path. |
| model_type      | string         | `Checkpoint`  | The type of model to search for. <br>(for example `Checkpoint`, `TextualInversion`, `Controlnet`, `LORA`, `Hypernetwork`, `AestheticGradient`, `Poses`) |
| base_model      | string         | None          | Trained model tag (for example `SD 1.5`, `SD 3.5`, `SDXL 1.0`)                      |
| download        | bool           | False         | Whether to download the model.                                                       |
| force_download  | bool           | False         | Whether to force the download if the model already exists.                           |
| cache_dir       | string, Path   | None          | Path to the folder where cached files are stored.                                    |
| resume          | bool           | False         | Whether to resume an incomplete download.                                            |
| token           | string         | None          | API token for Civitai authentication.                                                |
| include_params  | bool           | False         | Whether to include parameters in the returned data.                                  |
| skip_error      | bool           | False         | Whether to skip errors and return None.                                              |

### Search Huggingface<a name = "Search_Huggingface"></a>

> [!TIP]
> **If an error occurs, insert the `token` and run again.**

#### `EasyPipeline.from_huggingface` parameters

| Name                  | Type                | Default        | Description                                                                      |
|:---------------------:|:-------------------:|:--------------:|:--------------------------------------------------------------------------------:|
| search_word           | string, Path        | required       | The search query string. Can be a keyword, Hugging Face URL, local directory or file path, or a Hugging Face path (`<creator>/<repo>`). |
| checkpoint_format     | string              | `single_file`  | The format of the model checkpoint.<br>- `single_file` to search for `single file checkpoint` <br>- `diffusers` to search for `multifolder diffusers format checkpoint` |
| dtype                 | string, torch.dtype | None           | Override the default `torch.dtype` and load the model with another dtype.         |
| force_download        | bool                | False          | Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist. |
| cache_dir             | string, Path        | None           | Path to a directory where a downloaded pretrained model configuration is cached if the standard cache is not used. |
| token                 | string, bool        | None           | The token to use as HTTP bearer authorization for remote files.                   |


#### `search_huggingface` parameters

| Name                  | Type                | Default        | Description                                                                      |
|:---------------------:|:-------------------:|:--------------:|:--------------------------------------------------------------------------------:|
| search_word           | string, Path        | required       | The search query string. Can be a keyword, Hugging Face URL, local directory or file path, or a Hugging Face path (`<creator>/<repo>`). |
| checkpoint_format     | string              | `single_file`  | The format of the model checkpoint. <br>- `single_file` to search for `single file checkpoint` <br>- `diffusers` to search for `multifolder diffusers format checkpoint` |
| pipeline_tag          | string              | None           | Tag to filter models by pipeline.                                                 |
| download              | bool                | False          | Whether to download the model.                                                    |
| force_download        | bool                | False          | Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist. |
| cache_dir             | string, Path        | None           | Path to a directory where a downloaded pretrained model configuration is cached if the standard cache is not used. |
| token                 | string, bool        | None           | The token to use as HTTP bearer authorization for remote files.                   |
| include_params        | bool                | False          | Whether to include parameters in the returned data.                               |
| skip_error            | bool                | False          | Whether to skip errors and return None.                                           |


## License<a name = "License"></a>
In accordance with [Apache-2.0 license](https://github.com/suzukimain/auto_diffusers/blob/master/LICENSE)


## Acknowledgement<a name = "Acknowledgement"></a>

I have used open source resources and free tools in the creation of this project.

I would like to take this opportunity to thank the open source community and those who provided free tools.
