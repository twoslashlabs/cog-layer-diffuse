from cog import BasePredictor, Input, Path # type: ignore
from typing import List
import os
import random
from typing import Sequence, Mapping, Any, Union
import torch
import utils.samplers
import warnings
warnings.simplefilter("ignore", UserWarning)


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    if path is None:
        path = os.getcwd()

    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    parent_directory = os.path.dirname(path)

    if parent_directory == path:
        return None

    return find_path(name, parent_directory)

'''
def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


add_comfyui_directory_to_sys_path()


def import_custom_nodes() -> None:
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    init_custom_nodes()
'''


from utils.nodes import (
    KSampler,
    CheckpointLoaderSimple,
    EmptyLatentImage,
    SaveImage,
    CLIPTextEncode,
    VAEDecode,
)

from utils.layerdiffuse import NODE_CLASS_MAPPINGS

with torch.inference_mode():
    checkpointloadersimple = CheckpointLoaderSimple()
    checkpointloadersimple = checkpointloadersimple.load_checkpoint(
        ckpt_name="./models/model.safetensors"
    )

    def pipe(prompt, negative_prompt, num_outputs, height, width, steps, cfg, sampler_name, scheduler):
        emptylatentimage = EmptyLatentImage().generate(
            width=width, height=height, batch_size=num_outputs
        )

        cliptextencode = CLIPTextEncode()

        cliptextencode_positive = cliptextencode.encode(
            text=prompt,
            clip=get_value_at_index(checkpointloadersimple, 1),
        )

        cliptextencode_negative = cliptextencode.encode(
            text=negative_prompt, clip=get_value_at_index(checkpointloadersimple, 1)
        )

        """
        layereddiffusionapply = NODE_CLASS_MAPPINGS["LayeredDiffusionApply"]().apply_layered_diffusion(
            config="SDXL, Conv Injection",
            weight=1,
            model=get_value_at_index(checkpointloadersimple, 0),
        )
        """
        
        layereddiffusionapply = NODE_CLASS_MAPPINGS["LayeredDiffusionApply"]().apply_layered_diffusion(
            config="SDXL, Attention Injection",
            weight=1,
            model=get_value_at_index(checkpointloadersimple, 0),
        )

        ksampler = KSampler().sample(
            seed=random.randint(1, 2**64),
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=1.0,
            model=get_value_at_index(layereddiffusionapply, 0),
            positive=get_value_at_index(cliptextencode_positive, 0),
            negative=get_value_at_index(cliptextencode_negative, 0),
            latent_image=get_value_at_index(emptylatentimage, 0),
        )

        vaedecode = VAEDecode().decode(
            samples=get_value_at_index(ksampler, 0),
            vae=get_value_at_index(checkpointloadersimple, 2),
        )

        layereddiffusiondecodergba = NODE_CLASS_MAPPINGS["LayeredDiffusionDecodeRGBA"]().decode(
            sd_version="SDXL",
            sub_batch_size=16,
            samples=get_value_at_index(ksampler, 0),
            images=get_value_at_index(vaedecode, 0),
        )

        saveimage = SaveImage().save_images(
            filename_prefix="output",
            images=get_value_at_index(layereddiffusiondecodergba, 0),
        )

        return saveimage

class Predictor(BasePredictor):
    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="abstract beauty, centered, looking at the camera, approaching perfection, dynamic, moonlight, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by Carne Griffiths and Wadim Kashin"
        ),
        negative_prompt: str = Input(
            description="Negative Input prompt",
            default="watermark, text"
        ),
        width: int = Input(
            description="Width of output image",
            default=1024
        ),
        height: int = Input(
            description="Height of output image",
            default=1024
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        cfg: float = Input(
            description="Guidance strength",
            ge=0.00,
            le=100.00,
            default=7.50,
        ),
        sampler_name: str = Input(
            description="sampler",
            choices=utils.samplers.KSampler.SAMPLERS,
            default="dpmpp_sde",
        ),
        scheduler: str = Input(
            description="sampler",
            choices=utils.samplers.KSampler.SCHEDULERS,
            default="ddim_uniform",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=25
        ),
    ) -> List[Path]:
        if height % 64:
            height = (height // 64 + 1) * 64
            print(f"[Info] Output height has been resized to {height}")

        if width % 64:
            width = (width // 64 + 1) * 64
            print(f"[Info] Output witdh has been resized to {width}")

        images = pipe(prompt, negative_prompt, num_outputs, height, width, num_inference_steps, cfg, sampler_name, scheduler)['ui']['images']

        output_paths = []

        for i, image in enumerate(images):
            output_paths.append(Path(f"./utils/output/{image['filename']}"))

        print(output_paths)

        return output_paths