import os
import torch
import json
import torch.nn.functional as F

from torchvision.utils import save_image,make_grid
from torchvision.io import read_image
from pytorch_lightning import seed_everything


from diffusers import DDIMScheduler
from OIIctrl.diffuser_utils import OIICtrlPipeline
from OIIctrl.OIIctrl import OIISelfAttentionControlMaskExpand,OIISelfAttentionControlMask
from OIIctrl.config import Config as cfg
from OIIctrl.OIIctrl_utils import load_image,expand_mask,regiter_attention_editor_diffusers,AttentionBase,get_ref_object_token_ids
from PIL import Image


model_path = "botp/stable-diffusion-v1-5"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model = OIICtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(cfg.DEVICE)


root="dataset"

SOURCE_IMAGES_PATH = "./images"
SOURCE_MASKS_PATH="./masks"

target_prompt_dict={
    # "ball":"a tomato",
    # "messi":"messi and a rugby",
    # "pink T-shirt":"blue T-shirt",
    # "trophy":"a small fire hydrant on the table",
    # "watch_2":"he has a bracelet on his arm",
    # "watch":"he has a bracelet on his arm"
    
    # "ball":["a tomato","a rugby","a mango","a tennis ball"],
    # "messi":["messi and a tomato","messi and a rugby","messi and a mango","messi and a tennis ball","a tomato","a rugby","a mango","a tennis ball"],
    # "trophy":["a fire hydrant","a traffic light","a tree","a vase","a flower pot","a fire hydrant on the table","a traffic light on the table","a tree on the table","a vase on the table","a flower pot on the table"],
    # "watch":["a apple watch","a ring","a bracelet","a watch","he has a bracelet on his arm","he has a ring on his arm","he has a apple watch on his arm"],
    # "pink T-shirt":["blue T-shirt","white T-shirt","a jacket","a vest"],
    
    'elephant':["a ice octopus","a Godzilla"],
    'giraffe':["a fire octopus","a Godzilla"],
    'zebra':["a octopus","a Godzilla"],
}

# target_prompt_object_dict={
#    "ball":"a tomato",
#    "messi":"a rugby",
#     "pink T-shirt":"blue T-shirt",
#     "trophy":"a small fire hydrant",
#     "watch_2":"a bracelet",
#     "watch":"a bracelet"
# }

def extract_object_mask(image):
    
    if image.shape[0]>3: image=image[1]
    else: image=image[0]
    object_mask = image > 0.0  # Adjust the threshold if needed
    
    object_mask = object_mask.float()
   
    return object_mask

# cfg.GUIDANCE_SCALE=30
task=f"result"
 
cfg.STEP_QUERY=7
cfg.LAYER_QUERY=17
cfg.SCALE_MASK=0.05
cfg.GUIDANCE_SCALE=15

for img_file in os.listdir(os.path.join(root,SOURCE_IMAGES_PATH)):

    
    source_image = load_image(os.path.join(root,SOURCE_IMAGES_PATH,img_file), cfg.DEVICE)
    source_prompt = img_file.split(".")[0]
    
    source_mask=torch.ones((512,512))
    
    # source_mask[:,:source_mask.shape[1]//2]=0
    source_mask[:source_mask.shape[0]//2]=0
    # source_mask=read_image(os.path.join(root,SOURCE_MASKS_PATH,f"{source_prompt}.png"))[0].to(cfg.DEVICE)/255.0 # shape (H,W)
    
    # target_mask=expand_mask(source_mask,cfg.SCALE_MASK)
    
    target_mask=source_mask.clone()
    
    
    
    target_prompts= target_prompt_dict[source_prompt]
 


    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)

    start_code, intermediates = model.invert(source_image,
                                            prompt="",
                                            guidance_scale=cfg.GUIDANCE_SCALE,
                                            num_inference_steps=cfg.MAX_STEP,
                                            return_intermediates=True)


    start_code = start_code.expand(2, -1, -1, -1)
    ref_original=intermediates[0]
    # prompts = [target_prompt]
    
    
    
    image_fixed1 = model(target_prompts[0],
                            latents=start_code[-1:],
                            num_inference_steps=cfg.MAX_STEP,
                            guidance_scale=cfg.GUIDANCE_SCALE)
    #--------------------------------------------------------
    
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)
    # prompts = [target_prompt]
    
    image_fixed2 = model(target_prompts[1],
                            latents=start_code[-1:],
                            num_inference_steps=cfg.MAX_STEP,
                            guidance_scale=cfg.GUIDANCE_SCALE)
    #---------------------------------------------------------------------------------------------#
    
    
    
    editor = OIISelfAttentionControlMask(start_step=cfg.STEP_QUERY, start_layer=cfg.LAYER_QUERY,
                                        mask_s=source_mask,mask_t=target_mask,
                                        total_steps=cfg.MAX_STEP,
                                        )
    

    regiter_attention_editor_diffusers(model, editor)

    

    
    image_combined_1= model(target_prompts,
                    latents=start_code,
                    ref_intermediates=intermediates,
                    guidance_scale=cfg.GUIDANCE_SCALE,
                    num_inference_steps=cfg.MAX_STEP,
                    )
    
    #--------------------------------------------------------------------------------------------------------
    source_mask=1-source_mask
    target_mask=1-target_mask
    editor = OIISelfAttentionControlMask(start_step=cfg.STEP_QUERY, start_layer=cfg.LAYER_QUERY,
                                        mask_s=source_mask,mask_t=target_mask,
                                        total_steps=cfg.MAX_STEP,
                                        )
    

    regiter_attention_editor_diffusers(model, editor)    
    image_combined_2= model(target_prompts,
                    latents=start_code,
                    ref_intermediates=intermediates,
                    guidance_scale=cfg.GUIDANCE_SCALE,
                    num_inference_steps=cfg.MAX_STEP,
                    )

    #---------------------------------------------------------------------------------------------#
    
    source_mask=torch.ones((512,512))
    
    source_mask[:,:source_mask.shape[1]//2]=0
    target_mask=source_mask.clone()
    editor = OIISelfAttentionControlMask(start_step=cfg.STEP_QUERY, start_layer=cfg.LAYER_QUERY,
                                        mask_s=source_mask,mask_t=target_mask,
                                        total_steps=cfg.MAX_STEP,
                                        )
    

    regiter_attention_editor_diffusers(model, editor)

    

    
    image_combined_3= model(target_prompts,
                    latents=start_code,
                    ref_intermediates=intermediates,
                    guidance_scale=cfg.GUIDANCE_SCALE,
                    num_inference_steps=cfg.MAX_STEP,
                    )
    
    #--------------------------------------------------------------------------------------------------------
    source_mask=1-source_mask
    target_mask=1-target_mask
    editor = OIISelfAttentionControlMask(start_step=cfg.STEP_QUERY, start_layer=cfg.LAYER_QUERY,
                                        mask_s=source_mask,mask_t=target_mask,
                                        total_steps=cfg.MAX_STEP,
                                        )
    

    regiter_attention_editor_diffusers(model, editor)    
    image_combined_4= model(target_prompts,
                    latents=start_code,
                    ref_intermediates=intermediates,
                    guidance_scale=cfg.GUIDANCE_SCALE,
                    num_inference_steps=cfg.MAX_STEP,
                    )
    
    image_compose=[
        # image_fixed[0],
        # tgt_mask[0],   
        image_combined_1[0],
        image_combined_1[0],
        image_fixed1[-1],
        image_fixed2[-1],
        image_combined_1[-1],
        image_combined_2[-1],
         image_combined_3[-1],
          image_combined_4[-1],
        # image_orginal_interpolate_intermediate_expand2[2],
        # image_orginal_interpolate_intermediate_expand3[2]
    ]
    
    out_dir=task+"_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path=os.path.join(out_dir,f"{ source_prompt }_{target_prompts[0]}_{target_prompts[1]}.png")
    out_images=make_grid(image_compose,nrow=2)              
    save_image(out_images, out_path)

    print("Syntheiszed images are saved in", out_path)
    # break
