import os
import mimetypes
import json
import shutil
import re
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import seed as seed_helper
from replicate_weights import download_replicate_weights
from dataclasses import dataclass
import requests
from huggingface_hub import HfApi
import tempfile

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
COMFYUI_LORAS_DIR = "ComfyUI/models/loras"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")
api_json_file = "workflow.json"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


@dataclass
class Inputs:
    prompt = Input(description="Text prompt for video generation")
    negative_prompt = Input(
        description="Things you do not want to see in your video", default=""
    )
    image = Input(
        description="Image to use as a starting frame for image to video generation.",
        default=None,
    )
    aspect_ratio = Input(
        description="The aspect ratio of the video. 16:9, 9:16, 1:1, etc.",
        choices=["16:9", "9:16", "1:1"],
        default="16:9",
    )
    frames = Input(
        description="The number of frames to generate (1 to 5 seconds)",
        choices=[17, 33, 49, 65, 81],
        default=81,
    )
    model = Input(
        description="The model to use. 1.3b is faster, but 14b is better quality. A LORA either works with 1.3b or 14b, depending on the version it was trained on.",
        choices=["1.3b", "14b"],
        default="14b",
    )
    lora_url = Input(
        description="Optional: The URL or HuggingFace repo ID (username/repo) of a LORA to use", 
        default=None
    )
    lora_strength_model = Input(
        description="Strength of the LORA applied to the model. 0.0 is no LORA.",
        default=1.0,
    )
    lora_strength_clip = Input(
        description="Strength of the LORA applied to the CLIP model. 0.0 is no LORA.",
        default=1.0,
    )
    sample_shift = Input(
        description="Sample shift factor", default=8.0, ge=0.0, le=10.0
    )
    sample_guide_scale = Input(
        description="Higher guide scale makes prompt adherence better, but can reduce variation",
        default=5.0,
        ge=0.0,
        le=10.0,
    )
    sample_steps = Input(
        description="Number of generation steps. Fewer steps means faster generation, at the expensive of output quality. 30 steps is sufficient for most prompts",
        default=30,
        ge=1,
        le=60,
    )
    seed = Input(default=seed_helper.predict_seed())
    replicate_weights = Input(
        description="Replicate LoRA weights to use. Leave blank to use the default weights.",
        default=None,
    )
    fast_mode = Input(
        description="Speed up generation with different levels of acceleration. Faster modes may degrade quality somewhat. The speedup is dependent on the content, so different videos may see different speedups.",
        choices=["Off", "Balanced", "Fast"],
        default="Balanced",
    )
    resolution = Input(
        description="The resolution of the video. 720p is not supported for 1.3b.",
        choices=["480p", "720p"],
        default="480p",
    )


def is_huggingface_repo_id(text: str) -> bool:
    """Check if the text looks like a HuggingFace repo ID (username/repo)"""
    if not text or "//" in text:  # Skip empty strings and URLs
        return False
    return bool(re.match(r"^[^/]+/[^/]+$", text))

def download_from_huggingface(repo_id: str, lora_dir: str):
    """Downloads a .safetensors file from a HuggingFace repository"""
    # Initialize HuggingFace API
    api = HfApi()
    try:
        # List files in the repository
        files = api.list_repo_files(repo_id)
    except Exception as e:
        raise ValueError(f"Failed to access HuggingFace repo '{repo_id}': {e}")
    
    # Find safetensors files
    safetensors_files = [f for f in files if f.endswith(".safetensors")]
    if not safetensors_files:
        raise ValueError(f"No .safetensors files found in HuggingFace repo: {repo_id}")
    
    # Log all available .safetensors files for transparency
    if len(safetensors_files) > 1:
        print(f"Found multiple .safetensors files in {repo_id}:")
        for i, file in enumerate(safetensors_files):
            print(f"  {i+1}. {file}")
        print(f"Using the first file: {safetensors_files[0]}")
    
    # Use the first safetensors file
    hf_filename = safetensors_files[0]
    
    # Determine model type from filename - use cautious approach
    model_type = None
    if "1.3b" in hf_filename.lower():
        model_type = "1.3b"
    elif "14b" in hf_filename.lower():
        model_type = "14b"
    else:
        print(f"Warning: Could not determine model type from filename '{hf_filename}'. Using default model.")
    
    # Download URL
    download_url = f"https://huggingface.co/{repo_id}/resolve/main/{hf_filename}"
    
    # Local path
    local_filename = os.path.basename(hf_filename)
    target_path = os.path.join(lora_dir, local_filename)
    
    # Download file
    print(f"Downloading {download_url}")
    response = requests.get(download_url)
    response.raise_for_status()
    
    with open(target_path, 'wb') as f:
        f.write(response.content)
    
    return local_filename, model_type


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        os.makedirs("ComfyUI/models/loras", exist_ok=True)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[
                "wan_2.1_vae.safetensors",
                "umt5_xxl_fp16.safetensors",
                "clip_vision_h.safetensors",
            ],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    def get_width_and_height(self, resolution: str, aspect_ratio: str):
        sizes = {
            "480p": {
                "16:9": (832, 480),
                "9:16": (480, 832),
                "1:1": (644, 644),
            },
            "720p": {
                "16:9": (1280, 720),
                "9:16": (720, 1280),
                "1:1": (980, 980),
            },
        }
        return sizes[resolution][aspect_ratio]

    def set_model_loader(self, model_loader, model: str):
        if model == "14b-i2v-480p":
            model_loader["unet_name"] = "wan2.1_i2v_480p_14B_bf16.safetensors"
        elif model == "14b-i2v-720p":
            model_loader["unet_name"] = "wan2.1_i2v_720p_14B_bf16.safetensors"
        elif model == "14b":
            model_loader["unet_name"] = "wan2.1_t2v_14B_bf16.safetensors"
        elif model == "1.3b":
            model_loader["unet_name"] = "wan2.1_t2v_1.3B_bf16.safetensors"

    def update_workflow(self, workflow, **kwargs):
        is_image_to_video = kwargs["image_filename"] is not None
        model = f"{kwargs['model']}-i2v-{kwargs['resolution']}" if is_image_to_video else kwargs["model"]

        self.set_model_loader(workflow["37"]["inputs"], model)

        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        negative_prompt = workflow["7"]["inputs"]
        negative_prompt["text"] = f"nsfw, {kwargs['negative_prompt']}"

        sampler = workflow["3"]["inputs"]
        sampler["seed"] = kwargs["seed"]
        sampler["cfg"] = kwargs["sample_guide_scale"]
        sampler["steps"] = kwargs["sample_steps"]

        shift = workflow["48"]["inputs"]
        shift["shift"] = kwargs["sample_shift"]

        if is_image_to_video:
            del workflow["40"]
            wan_i2v_latent = workflow["58"]["inputs"]
            wan_i2v_latent["length"] = kwargs["frames"]

            image_loader = workflow["55"]["inputs"]
            image_loader["image"] = kwargs["image_filename"]

            image_resizer = workflow["56"]["inputs"]
            if kwargs["resolution"] == "720p":
                image_resizer["target_size"] = 1008
            else:
                image_resizer["target_size"] = 644

        else:
            del workflow["55"]
            del workflow["56"]
            del workflow["57"]
            del workflow["58"]
            del workflow["59"]
            del workflow["60"]
            width, height = self.get_width_and_height(
                kwargs["resolution"], kwargs["aspect_ratio"]
            )
            empty_latent_video = workflow["40"]["inputs"]
            empty_latent_video["length"] = kwargs["frames"]
            empty_latent_video["width"] = width
            empty_latent_video["height"] = height

            sampler["model"] = ["48", 0]
            sampler["positive"] = ["6", 0]
            sampler["negative"] = ["7", 0]
            sampler["latent_image"] = ["40", 0]

        thresholds = {
            "14b": {
                "Balanced": 0.15,
                "Fast": 0.2,
                "coefficients": "14B",
            },
            "14b-i2v-480p": {
                "Balanced": 0.19,
                "Fast": 0.26,
                "coefficients": "i2v_480",
            },
            "14b-i2v-720p": {
                "Balanced": 0.2,
                "Fast": 0.3,
                "coefficients": "i2v_720",
            },
            "1.3b": {
                "Balanced": 0.07,
                "Fast": 0.08,
                "coefficients": "1.3B",
            },
        }

        fast_mode = kwargs["fast_mode"]
        if fast_mode == "Off":
            # Turn off tea cache
            del workflow["54"]
            workflow["49"]["inputs"]["model"] = ["37", 0]
        else:
            tea_cache = workflow["54"]["inputs"]
            tea_cache["coefficients"] = thresholds[model]["coefficients"]
            tea_cache["rel_l1_thresh"] = thresholds[model][fast_mode]

        if kwargs["lora_url"] or kwargs["lora_filename"]:
            lora_loader = workflow["49"]["inputs"]
            if kwargs["lora_filename"]:
                lora_loader["lora_name"] = kwargs["lora_filename"]
            elif kwargs["lora_url"]:
                lora_loader["lora_name"] = kwargs["lora_url"]

            lora_loader["strength_model"] = kwargs["lora_strength_model"]
            lora_loader["strength_clip"] = kwargs["lora_strength_clip"]
        else:
            del workflow["49"]  # delete lora loader node
            positive_prompt["clip"] = ["38", 0]
            shift["model"] = ["37", 0] if fast_mode == "Off" else ["54", 0]

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        image: Path | None = None,
        aspect_ratio: str = "16:9",
        frames: int = 81,
        model: str | None = None,
        lora_url: str | None = None,
        lora_strength_model: float = 1.0,
        lora_strength_clip: float = 1.0,
        fast_mode: str = "Balanced",
        sample_shift: float = 8.0,
        sample_guide_scale: float = 5.0,
        sample_steps: int = 30,
        seed: int | None = None,
        resolution: str = "480p",
        replicate_weights: str | None = None,
    ) -> List[Path]:
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        if image and model == "1.3b":
            raise ValueError("Image to video generation is not supported for 1.3b")

        image_filename = None
        if image:
            image_filename = self.filename_with_extension(image, "image")
            self.handle_input_file(image, image_filename)

        lora_filename = None
        inferred_model_type = None
        if replicate_weights:
            lora_filename, inferred_model_type = download_replicate_weights(
                replicate_weights, COMFYUI_LORAS_DIR
            )
            model = inferred_model_type
        elif lora_url:
            # Handle existing Replicate URL cases first
            if m := re.match(
                r"^(?:https?://replicate.com/)?([^/]+)/([^/]+)/?$", lora_url
            ):
                owner, model_name = m.groups()
                try:
                    lora_filename, inferred_model_type = download_replicate_weights(
                        f"https://replicate.com/{owner}/{model_name}/_weights",
                        COMFYUI_LORAS_DIR,
                    )
                except Exception as e:
                    print(f"Not a valid Replicate model, trying as HuggingFace repo: {str(e)}")
                    if is_huggingface_repo_id(lora_url):
                        lora_filename, inferred_model_type = download_from_huggingface(
                            lora_url, COMFYUI_LORAS_DIR
                        )
            elif lora_url.startswith("https://replicate.delivery"):
                lora_filename, inferred_model_type = download_replicate_weights(
                    lora_url, COMFYUI_LORAS_DIR
                )
            # Try HuggingFace repo ID as a last resort
            elif is_huggingface_repo_id(lora_url):
                print(f"Processing HuggingFace repo ID: {lora_url}")
                lora_filename, inferred_model_type = download_from_huggingface(
                    lora_url, COMFYUI_LORAS_DIR
                )
                if inferred_model_type:
                    model = model or inferred_model_type

        if inferred_model_type and inferred_model_type != model:
            print(
                f"Warning: Model type mismatch between requested model ({model}) and inferred model type ({inferred_model_type}). Using {inferred_model_type}."
            )
            model = inferred_model_type

        if resolution == "720p" and model == "1.3b":
            print("Warning: 720p is not supported for 1.3b, using 480p instead")
            resolution = "480p"

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            fast_mode=fast_mode,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            sample_steps=sample_steps,
            model=model,
            frames=frames,
            aspect_ratio=aspect_ratio,
            lora_filename=lora_filename,
            lora_url=lora_url,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
            resolution=resolution,
            image_filename=image_filename,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return self.comfyUI.get_files(OUTPUT_DIR, file_extensions=["mp4"])


class StandaloneLoraPredictor(Predictor):
    def predict(
        self,
        prompt: str = Inputs.prompt,
        negative_prompt: str = Inputs.negative_prompt,
        image: Path = Inputs.image,
        aspect_ratio: str = Inputs.aspect_ratio,
        frames: int = Inputs.frames,
        model: str = Inputs.model,
        resolution: str = Inputs.resolution,
        lora_url: str = Inputs.lora_url,
        lora_strength_model: float = Inputs.lora_strength_model,
        lora_strength_clip: float = Inputs.lora_strength_clip,
        fast_mode: str = Inputs.fast_mode,
        sample_steps: int = Inputs.sample_steps,
        sample_guide_scale: float = Inputs.sample_guide_scale,
        sample_shift: float = Inputs.sample_shift,
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        return self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            aspect_ratio=aspect_ratio,
            frames=frames,
            model=model,
            resolution=resolution,
            lora_url=lora_url,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
            fast_mode=fast_mode,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            sample_steps=sample_steps,
            seed=seed,
            replicate_weights=None,
        )


class TrainedLoraPredictor(Predictor):
    def predict(
        self,
        prompt: str = Inputs.prompt,
        negative_prompt: str = Inputs.negative_prompt,
        aspect_ratio: str = Inputs.aspect_ratio,
        frames: int = Inputs.frames,
        resolution: str = Inputs.resolution,
        lora_strength_model: float = Inputs.lora_strength_model,
        lora_strength_clip: float = Inputs.lora_strength_clip,
        fast_mode: str = Inputs.fast_mode,
        sample_steps: int = Inputs.sample_steps,
        sample_guide_scale: float = Inputs.sample_guide_scale,
        sample_shift: float = Inputs.sample_shift,
        seed: int = seed_helper.predict_seed(),
        replicate_weights: str = Inputs.replicate_weights,
    ) -> List[Path]:
        return self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=None,
            aspect_ratio=aspect_ratio,
            frames=frames,
            model=None,
            resolution=resolution,
            lora_url=None,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
            fast_mode=fast_mode,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            sample_steps=sample_steps,
            seed=seed,
            replicate_weights=replicate_weights,
        )
