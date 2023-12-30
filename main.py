import json
import random
from pathlib import Path
import torch
from PIL import Image
from pathlib import Path
import torchvision as tv
# from dmvton_pipeline import DMVTONPipeline
from utils.torch_utils import select_device

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import torch.nn.functional as F

from models.generators.mobile_unet import MobileNetV2_unet
from models.warp_modules.mobile_afwm import MobileAFWM as AFWM
from pipelines.base_pipeline import BaseVTONPipeline
from utils.torch_utils import get_ckpt, load_ckpt
import torch.nn as nn

from pathlib import Path


from utils.torch_utils import select_device


class BaseVTONPipeline(nn.Module):
    """
    Base class for pipeline
    """

    def __init__(self, checkpoints=None, **kargs):
        super().__init__()


class DMVTONPipeline(BaseVTONPipeline):
    """
    DM-VTON inference pipeline
    """

    def __init__(self, align_corners=True, checkpoints=None):
        super().__init__()
        self.align_corners = align_corners
        self.warp_model = AFWM(3, align_corners)
        self.gen_model = MobileNetV2_unet(7, 4)

        if checkpoints is not None:
            self._load_pretrained(checkpoints)

    def _load_pretrained(self, checkpoints):
        if checkpoints.get('warp') is not None:
            warp_ckpt = get_ckpt(checkpoints['warp'])
            load_ckpt(self.warp_model, warp_ckpt)
        if checkpoints.get('gen') is not None:
            gen_ckpt = get_ckpt(checkpoints['gen'])
            load_ckpt(self.gen_model, gen_ckpt)

    def forward(self, person, clothes, clothes_edge, phase="test"):
        clothes_edge = (clothes_edge > 0.5).float()
        clothes = clothes * clothes_edge

        # Warp
        flow_out = self.warp_model(person, clothes, phase=phase)
        (
            warped_cloth,
            last_flow,
        ) = flow_out
        warped_edge = F.grid_sample(
            clothes_edge,
            last_flow.permute(0, 2, 3, 1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=self.align_corners,
        )

        # Gen
        gen_inputs = torch.cat([person, warped_cloth, warped_edge], 1)
        gen_outputs = self.gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        return p_tryon, warped_cloth

def preprocess_image(image, size=(256, 192)):
    transform = get_transform()
    image = image.resize(size, Image.BICUBIC)
    image_tensor = transform(image)
    return image_tensor

def get_transform(method=Image.BICUBIC, normalize=True):
    transform_list = [transforms.Resize(size=(256, 192), interpolation=method),
                      transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def run_test_pf(pipeline, person_img, clothing_img, device):
    person_tensor = preprocess_image(person_img).to(device)
    clothing_tensor = preprocess_image(clothing_img).to(device)

    # Run the pipeline
    with torch.no_grad():
        p_tryon, _ = pipeline(person_tensor.unsqueeze(0), clothing_tensor.unsqueeze(0))

    # Save the result
    script_dir = Path(__file__).parent
    save_dir = script_dir / 'save'
    tryon_dir = save_dir / 'tryon'
    tryon_dir.mkdir(parents=True, exist_ok=True)
    tv.utils.save_image(p_tryon, tryon_dir / 'result.jpg', normalize=True, value_range=(-1, 1))

def main():
    device = select_device("cuda" if torch.cuda.is_available() else "cpu")

    # Correctly build checkpoint paths
    script_dir = Path(__file__).parent
    warp_checkpoint = script_dir / 'checkpoints' / 'dmvton_pf_warp.pth'
    gen_checkpoint = script_dir / 'checkpoints' / 'dmvton_pf_gen.pth'

    # Hardcoded image paths from 'images' folder
    person_image_path = script_dir / 'images' / '000038_0.jpg' 
    clothing_image_path = script_dir / 'images' / '015095_1.jpg'

    pipeline = DMVTONPipeline(checkpoints={'warp': warp_checkpoint, 'gen': gen_checkpoint}).to(device)
    pipeline.eval()

    person_img = Image.open(person_image_path).convert('RGB')
    clothing_img = Image.open(clothing_image_path).convert('RGB')

    run_test_pf(pipeline, person_img, clothing_img, device)

if __name__ == "__main__":
    main()
