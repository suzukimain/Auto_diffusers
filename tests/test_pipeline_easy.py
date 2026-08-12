from dataclasses import dataclass, field

import pytest
import requests
import torch

from auto_diffusers import pipeline_easy


def test_keyword_types(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert pipeline_easy.get_keyword_types(str(checkpoint))["loading_method"] == "from_single_file"
    assert pipeline_easy.get_keyword_types(str(model_dir))["loading_method"] == "from_pretrained"
    official_repo = "stabilityai/stable-diffusion-xl-base-1.0"
    assert pipeline_easy.get_keyword_types(official_repo)["type"]["hf_repo"]
    assert pipeline_easy.get_keyword_types("https://civitai.com/models/1")["type"]["civitai_url"]


def test_dtype_normalization(monkeypatch):
    monkeypatch.setattr(pipeline_easy, "diffusers_version", "0.36.0")
    assert pipeline_easy._normalize_dtype_kwargs({"dtype": torch.float16}) == {
        "torch_dtype": torch.float16
    }

    monkeypatch.setattr(pipeline_easy, "diffusers_version", "0.39.0")
    assert pipeline_easy._normalize_dtype_kwargs({"torch_dtype": torch.float16}) == {
        "dtype": torch.float16
    }

    with pytest.raises(ValueError, match="only one"):
        pipeline_easy._normalize_dtype_kwargs(
            {"dtype": torch.float16, "torch_dtype": torch.float16}
        )


def test_pipeline_loading_kwargs_removes_search_options(monkeypatch):
    monkeypatch.setattr(pipeline_easy, "diffusers_version", "0.39.0")
    result = pipeline_easy._pipeline_loading_kwargs(
        {
            "dtype": torch.float16,
            "download": True,
            "include_params": True,
            "pipeline_tag": "text-to-image",
            "token": "token",
        }
    )
    assert result == {"dtype": torch.float16, "token": "token"}


def test_safe_filename_accepts_a_plain_filename():
    assert pipeline_easy._safe_filename("model.safetensors") == "model.safetensors"


def test_safe_filename_rejects_parent_and_subdirectory_paths():
    with pytest.raises(ValueError, match="unsafe filename"):
        pipeline_easy._safe_filename("../model.safetensors")

    with pytest.raises(ValueError, match="unsafe filename"):
        pipeline_easy._safe_filename("folder/model.safetensors")


def test_cache_path_and_remote_url_validation(tmp_path):
    path = pipeline_easy._safe_cache_path(tmp_path, "1", "2", "model.safetensors")
    assert path.startswith(str(tmp_path))

    with pytest.raises(ValueError, match="outside"):
        pipeline_easy._safe_cache_path(tmp_path, "..", "model.safetensors")

    pipeline_easy._validate_remote_url("https://civitai.com/api/download/models/1")
    pipeline_easy._validate_remote_url(
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
    )

    with pytest.raises(ValueError, match="HTTPS"):
        pipeline_easy._validate_remote_url("http://civitai.com/api/download/models/1")

    with pytest.raises(ValueError, match="HTTPS"):
        pipeline_easy._validate_remote_url("https://example.com/model.safetensors")


def test_head_validation_rejects_redirect_to_untrusted_host(monkeypatch):
    class Response:
        url = "https://example.com/model.safetensors"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(pipeline_easy.requests, "head", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="HTTPS URLs from Civitai or Hugging Face"):
        pipeline_easy.validate_url_with_head(
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
        )


def test_auto_pipeline_mappings_follow_diffusers():
    assert (
        pipeline_easy.SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING["v1"]
        is pipeline_easy.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["stable-diffusion"]
    )
    assert (
        pipeline_easy.SINGLE_FILE_CHECKPOINT_IMAGE2IMAGE_PIPELINE_MAPPING["v1"]
        is pipeline_easy.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["stable-diffusion"]
    )
    assert (
        pipeline_easy.SINGLE_FILE_CHECKPOINT_INPAINT_PIPELINE_MAPPING["v1"]
        is pipeline_easy.AUTO_INPAINT_PIPELINES_MAPPING["stable-diffusion"]
    )
    assert pipeline_easy.SINGLE_FILE_CHECKPOINT_TEXT2IMAGE_PIPELINE_MAPPING["inpainting"] is None


def test_single_file_loader_dispatch_and_dtype(monkeypatch):
    captured = {}

    class FakePipeline:
        @classmethod
        def from_single_file(cls, path, **kwargs):
            captured.update(path=path, kwargs=kwargs)
            return "pipeline"

    monkeypatch.setattr(pipeline_easy, "load_single_file_checkpoint", lambda _: {})
    monkeypatch.setattr(pipeline_easy, "infer_diffusers_model_type", lambda _: "v1")
    monkeypatch.setattr(pipeline_easy, "diffusers_version", "0.39.0")

    result = pipeline_easy.load_pipeline_from_single_file(
        "model.safetensors",
        {"v1": FakePipeline},
        dtype=torch.float16,
        download=True,
    )
    assert result == "pipeline"
    assert captured == {
        "path": "model.safetensors",
        "kwargs": {"dtype": torch.float16},
    }

    monkeypatch.setattr(pipeline_easy, "infer_diffusers_model_type", lambda _: "unknown")
    with pytest.raises(ValueError, match="not supported"):
        pipeline_easy.load_pipeline_from_single_file("model.safetensors", {})


@dataclass
class HubModel:
    id: str
    siblings: list = field(default_factory=list)


def default_security_repo_status():
    return {
        "scansDone": True,
        "filesWithIssues": [],
    }


@dataclass
class HubRepo:
    sha: str = "abc"
    security_repo_status: dict = field(default_factory=default_security_repo_status)


def test_huggingface_search_uses_supported_sort(monkeypatch):
    calls = {}

    class FakeHubApi:
        def list_models(self, **kwargs):
            calls["list_models"] = kwargs
            return [
                HubModel(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    [{"rfilename": "model.safetensors"}],
                )
            ]

        def model_info(self, **kwargs):
            return HubRepo()

    monkeypatch.setattr(pipeline_easy, "hf_api", FakeHubApi())
    result = pipeline_easy.search_huggingface(
        "stabilityai stable diffusion xl",
        include_params=True,
    )

    assert calls["list_models"]["sort"] == "downloads"
    assert "direction" not in calls["list_models"]
    assert result.repo_status.repo_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert result.model_status.file_name == "model.safetensors"


def test_huggingface_search_fails_closed_when_scan_is_incomplete(monkeypatch):
    class FakeHubApi:
        def list_models(self, **kwargs):
            return [
                HubModel(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    [{"rfilename": "model.safetensors"}],
                )
            ]

        def model_info(self, **kwargs):
            return HubRepo(security_repo_status={"scansDone": False})

    monkeypatch.setattr(pipeline_easy, "hf_api", FakeHubApi())
    with pytest.raises(ValueError, match="No models matching"):
        pipeline_easy.search_huggingface("stabilityai stable diffusion xl")


def _civitai_response(file_name="model.safetensors", scan="Success"):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "items": [
                    {
                        "id": 1,
                        "name": "Stable Diffusion 1.5",
                        "modelVersions": [
                            {
                                "id": 2,
                                "baseModel": "SD 1.5",
                                "trainedWords": [],
                                "stats": {"downloadCount": 1},
                                "files": [
                                    {
                                        "name": file_name,
                                        "downloadUrl": "https://civitai.com/api/download/models/2",
                                        "pickleScanResult": scan,
                                        "virusScanResult": scan,
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }

    return Response()


def test_civitai_search_timeout_and_scan(monkeypatch):
    calls = {}

    def fake_get(url, **kwargs):
        calls.update(url=url, kwargs=kwargs)
        return _civitai_response()

    monkeypatch.setattr(pipeline_easy.requests, "get", fake_get)
    result = pipeline_easy.search_civitai("Stable Diffusion 1.5", request_timeout=7)
    assert result == "https://civitai.com/api/download/models/2"
    assert calls["kwargs"]["timeout"] == 7

    monkeypatch.setattr(
        pipeline_easy.requests,
        "get",
        lambda *args, **kwargs: _civitai_response(scan="Pending"),
    )
    with pytest.raises(ValueError, match="No model found"):
        pipeline_easy.search_civitai("Stable Diffusion 1.5")


def test_civitai_rejects_unsafe_filename(monkeypatch):
    monkeypatch.setattr(
        pipeline_easy.requests,
        "get",
        lambda *args, **kwargs: _civitai_response("../model.safetensors"),
    )
    with pytest.raises(ValueError, match="unsafe filename"):
        pipeline_easy.search_civitai("Stable Diffusion 1.5")


def test_civitai_timeout_is_explicit_and_can_be_skipped(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(
        pipeline_easy.requests,
        "get",
        raise_timeout,
    )

    with pytest.raises(requests.RequestException, match="Could not query"):
        pipeline_easy.search_civitai("Stable Diffusion 1.5", request_timeout=1)

    result = pipeline_easy.search_civitai(
        "Stable Diffusion 1.5",
        request_timeout=1,
        skip_error=True,
    )
    assert result is None
