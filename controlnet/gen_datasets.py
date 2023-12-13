# Author: Yuru Jia
# Last Modified: 2023-10-19

import os
import os.path as osp
import random
import logging
import argparse
import json

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from diffusers import DDIMScheduler
from diffusers import AutoencoderKL
from controlnet.pipeline_refine import StableDiffusionControlNetRefinePipeline
from controlnet.controlnet_model import ControlNetModel
from controlnet.tools.training_classes import (
    get_class_stacks, 
    make_one_hot,
    get_cs_classes,
    map_label2RGB
    )
from controlnet.tools.refine import get_connected_components, encode_latents


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet inference script.")
    parser.add_argument(
        "--basemodel_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default=None,
        required=True,
        help="Path to controlnet model.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output_images",
        help="output image save path",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--num_generated_images",
        type=int,
        default=1,
        help="Number of generated images per prompt.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Crop resolution.",
    )    
    parser.add_argument(
        "--gen_file",
        type=str,
        help="Label files to be generated.",
    )
    parser.add_argument(
        "--resize_ratios",
        type=list,
        # default=[0.5, 0.75, 1.0, 1.25, 1.5],
        default=[],
        help="Resize ratio for controlnet conditioning image."
    )
    parser.add_argument(
        "--multidiffusion_rescale_factor",
        type=int,
        default=2,
        help="Rescale factor for multidiffusion."
    )
    parser.add_argument(
        "--comp_area_thre_la",
        type=int,
        default=20000,
        help="Minimum area for large connected components."
    )
    parser.add_argument(
        "--comp_area_thre_sm",
        type=int,
        default=2000,
        help="Maxmum area for small connected components."
    )
    parser.add_argument(
        "--multi_scale",
        type=int,
        default=1,
        help="1: use connected components, 0: no connected components."
    )
    parser.add_argument(
        "--multi_diff_stride",
        type=int,
        default=8,
        help="Stride for multi-diffusion."
    )
    parser.add_argument(
        "--weather_prompt",
        type=list, 
        default=["snowy", "rainy", "sunny","foggy", "night"],
        help="Diversify prompts from perspective of weather conditions."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def visualize_generated_image_grid(img_gen, img_mask, resolution=512):

    vis_grid = Image.new('RGB', (resolution * 3, resolution), (255, 255, 255))
    vis_grid.paste(img_gen, (0, 0))
    vis_grid.paste(Image.fromarray(map_label2RGB(img_mask)), (resolution, 0))
    vis_grid.paste(Image.blend(img_gen, Image.fromarray(map_label2RGB(img_mask)), 0.5), (resolution* 2, 0, resolution * 3, resolution))
 
    return vis_grid


def get_random_crop_rcs(label_map, ori_gta_file, c, rng, crop_size=512, resize_ratio=1.0):
        if isinstance(label_map, str):
            label_map = Image.open(label_map)
        if isinstance(ori_gta_file, str):
            ori_gta = Image.open(ori_gta_file)

        label_map_arr = np.array(label_map)
        indices = np.where(label_map_arr == c)
        w, h = label_map.size

        # resize image
        if resize_ratio != 1.0:
            label_map = label_map.resize((int(w * resize_ratio), int(h * resize_ratio)), Image.NEAREST)
            ori_gta = ori_gta.resize((int(w * resize_ratio), int(h * resize_ratio)), Image.BILINEAR)
            w, h = label_map.size
            indices = np.where(label_map_arr == c)


        for _ in range(10):
            # idx = np.random.randint(0, len(indices[0]) - 1)
            idx = rng.integers(0, len(indices[0]) - 1)
            y, x = indices[0][idx], indices[1][idx]
            x1 = min(max(0, x - crop_size // 2), w -  crop_size)
            y1 = min(max(0, y - crop_size // 2), h - crop_size)
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            if np.sum(label_map_arr[y1:y2, x1:x2] == c) > 0.01 * crop_size * crop_size:
                break

        new_condition_img = label_map.crop((x1, y1, x2, y2))
        new_origta_img = ori_gta.crop((x1, y1, x2, y2))
        new_texts = get_class_stacks(new_condition_img)

        crop_results = {
            "crop_coords": (x1, y1, x2, y2),
            "crop_condition_img": new_condition_img,
            "crop_origta_img": new_origta_img,        # original GTA image
            "crop_texts": new_texts,
        }
        
          
        return crop_results



def main(args):
   
    # prepare the model and the pipeline
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path)
    vae = AutoencoderKL.from_pretrained(args.basemodel_path, 
                                        subfolder="vae", revision=None)
    vae.requires_grad_(False)
    vae.to("cuda", dtype=torch.float32)

    pipe = StableDiffusionControlNetRefinePipeline.from_pretrained(
                    args.basemodel_path, controlnet=controlnet)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    gen_id = int(os.path.basename(args.gen_file).replace(".json", ""))
    generator = torch.manual_seed(gen_id)
    rng = np.random.default_rng(gen_id)

    # create output folder
    out_dir = os.path.join(args.output_folder, str(gen_id))

    os.makedirs(out_dir, exist_ok=True)
    out_dir_img= osp.join(out_dir, "images")
    os.makedirs(out_dir_img, exist_ok=True)
    out_dir_label= osp.join(out_dir, "labels")
    os.makedirs(out_dir_label, exist_ok=True)
    out_dir_vis= osp.join(out_dir, "vis")
    os.makedirs(out_dir_vis, exist_ok=True)

    # config logging file
    logging.basicConfig(filename=os.path.join(out_dir, f"gen_{gen_id}.log"), level=logging.INFO)

    # Read label file as input
    with open(args.gen_file, 'r') as of:
        gen_label_files = json.load(of)   
        
    for i in range(len(gen_label_files)):

        c = int(gen_label_files[i]["class"])
        label_map = gen_label_files[i]["file_name"] 

        ori_gta_file = label_map.replace("_labelTrainIds", "").replace("labels", "images")

        label_file_id = os.path.basename(label_map).replace("_labelTrainIds.png", "")
        generated_id =  label_file_id + f"_genid{gen_id}"
        print(generated_id)
        
        
        if len(args.resize_ratios) > 0:
            # rescale_ratio = np.random.choice(args.resize_ratios)
            rescale_ratio = rng.choice(args.resize_ratios)
            w, h = Image.open(label_map).size
            label_map = Image.open(label_map).resize((int(w * rescale_ratio), int(h * rescale_ratio)), Image.NEAREST)
        else:
            label_map = Image.open(label_map)

        
        # get the original condition image which is cropped from the original label map with crop size 512
        rcs_crop_results = get_random_crop_rcs(label_map, ori_gta_file, c, rng, crop_size=args.resolution)
        crop_condition_img = np.array(rcs_crop_results["crop_condition_img"])

        prompt = rcs_crop_results["crop_texts"]
        
        if np.random.rand() < 0.5 and args.weather_prompt is not None:
            weather_prompt = np.random.choice(args.weather_prompt)
            prompt = "A city street scene with " + prompt + ", in " + weather_prompt + " weather."
            logging.info(f"rcs_class: {get_cs_classes()[c]}, weather: {weather_prompt}, generated_id: {generated_id}")

        
        rescale_factor = args.multidiffusion_rescale_factor

        # connected_components analysis               
        if args.multi_scale > 0:
            # Large connected components
            condition_comps_la, n_condition_comps = get_connected_components(
                crop_condition_img, 
                comp_area_thre=args.comp_area_thre_la,
                mode="large"
            )
            components_mask_la = condition_comps_la!=0
            components_mask_la = components_mask_la.astype(int)
            components_mask_la = torch.Tensor(components_mask_la)    
              
        # process cropped image label into one-hot encoding
        crop_condition_img_onehot = torch.Tensor(make_one_hot(crop_condition_img))
        crop_condition_img_onehot = torch.unsqueeze(crop_condition_img_onehot.permute(2, 0, 1), 0) #[1, class, H, W]
                
        # initial generation
        images_ini = pipe(
            prompt, 
            num_inference_steps=args.num_inference_steps, 
            generator=generator, 
            num_images_per_prompt=args.num_generated_images,
            cond_image=crop_condition_img_onehot,
            output_type="both",
            strength=1,
            rescale_factor=1,
            multi_diff_stride=64,
        ).images
        
        image_ini = images_ini["image"][0]
        image_ini_upsampl = image_ini.resize((args.resolution*rescale_factor, args.resolution*rescale_factor), Image.LANCZOS)
        image_ini_upsampl_latents = encode_latents(image_ini_upsampl, vae, generator)

        images = pipe(
            prompt, 
            num_inference_steps=args.num_inference_steps, 
            generator=generator, 
            num_images_per_prompt=args.num_generated_images,
            cond_image=crop_condition_img_onehot,
            output_type="pil",
            strength=1,
            rescale_factor=rescale_factor,
            multi_diff_stride=args.multi_diff_stride,
            add_inpaint=True,
            ini_latents=image_ini_upsampl_latents,
            init_img_mask_la=components_mask_la          
        ).images

        if images[0].size != (args.resolution, args.resolution):
            images[0] = images[0].resize((args.resolution, args.resolution), Image.LANCZOS)
        
        # save generated image
        output_file_img = f"{out_dir_img}/{generated_id}.png"
        output_file_label = f"{out_dir_label}/{generated_id}_labelTrainIds.png"
        output_file_grid = f"{out_dir_vis}/{generated_id}.png"
        
        images[0].save(output_file_img)

        if not isinstance(crop_condition_img, Image.Image):
            crop_condition_img = Image.fromarray(crop_condition_img)
        crop_condition_img.save(output_file_label)

        vis_grid = visualize_generated_image_grid(images[0], crop_condition_img, resolution=args.resolution)
        vis_grid.save(output_file_grid)


if __name__ == "__main__":
    args = parse_args()  
    main(args)