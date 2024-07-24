from audioop import mul
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler, \
                      EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, ControlNetModel, \
                      DDIMInverseScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path
import os
import random

import torchvision.transforms as T
# suppress partial model loading warning
logging.set_verbosity_error()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator
from .solver import FSolver

from .sd_step import *

def rgb2sat(img, T=None):
    max_ = torch.max(img, dim=1, keepdim=True).values + 1e-5
    min_ = torch.min(img, dim=1, keepdim=True).values
    sat = (max_ - min_) / max_
    if T is not None:
        sat = (1 - T) * sat
    return sat

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, t_range=[0.02, 0.98], max_t_range=0.98, num_train_timesteps=None, 
                 ddim_inv=False, use_control_net=False, textual_inversion_path = None, 
                 LoRA_path = None, guidance_opt=None):
        super().__init__()

        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32

        print(f'[INFO] loading stable diffusion...')

        model_key = guidance_opt.model_key
        assert model_key is not None

        is_safe_tensor = guidance_opt.is_safe_tensor
        assert guidance_opt.base_model_key is not None or not is_safe_tensor
        base_model_key = "stabilityai/stable-diffusion-v2-1" if guidance_opt.base_model_key is None else guidance_opt.base_model_key # for finetuned model only

        if is_safe_tensor:
            pipe = StableDiffusionPipeline.from_single_file(model_key, use_safetensors=True, torch_dtype=self.precision_t, load_safety_checker=False)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        # import pdb; pdb.set_trace()
        # if unet_key:
        #     pipe.unet = UNet2DConditionModel.from_pretrained(unet_key, torch_dtype=self.precision_t)
        self.ism = not guidance_opt.sds
        self.scheduler = DDIMScheduler.from_pretrained(model_key if not is_safe_tensor else base_model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        self.sche_func = ddim_step

        if use_control_net:
            controlnet_model_key = guidance_opt.controlnet_model_key
            self.controlnet_depth = ControlNetModel.from_pretrained(controlnet_model_key,torch_dtype=self.precision_t).to(device)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()

        pipe.enable_xformers_memory_efficient_attention()

        pipe = pipe.to(self.device)
        if textual_inversion_path is not None:
            pipe.load_textual_inversion(textual_inversion_path)
            print("load textual inversion in:.{}".format(textual_inversion_path))
        
        if LoRA_path is not None:
            from lora_diffusion import tune_lora_scale, patch_pipe
            print("load lora in:.{}".format(LoRA_path))
            patch_pipe(
                pipe,
                LoRA_path,
                patch_text=True,
                patch_ti=True,
                patch_unet=True,
            )
            tune_lora_scale(pipe.unet, 1.00)
            tune_lora_scale(pipe.text_encoder, 1.00)

        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        
        self.num_train_timesteps = num_train_timesteps if num_train_timesteps is not None else self.scheduler.config.num_train_timesteps        
        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)

        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, ))
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.warmup_step = int(self.num_train_timesteps*(max_t_range-t_range[1]))

        
        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(guidance_opt.noise_seed)
        self.noise_temp = None

        # self.noise = torch.randn((4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.rgb_latent_factors = torch.tensor([
                    # R       G       B
                    [ 0.298,  0.207,  0.208],
                    [ 0.187,  0.286,  0.173],
                    [-0.158,  0.189,  0.264],
                    [-0.184, -0.271, -0.473]
                ], device=self.device)
        self.alpha_schedule = torch.sqrt(self.scheduler.alphas_cumprod).to(self.device)
        self.sigma_schedule = torch.sqrt(1 - self.scheduler.alphas_cumprod).to(self.device)
        self.f_solver = FSolver(self.alpha_schedule,
                                self.sigma_schedule,
                                self.scheduler.config.num_train_timesteps,
                                self.scheduler.config.prediction_type,
                                'dpmsolver++', 1,
                                precision_t=self.precision_t)

        print(f'[INFO] loaded stable diffusion!')

    def augmentation(self, *tensors):
        augs = T.Compose([
                        T.RandomHorizontalFlip(p=0.5),
                    ])
        
        channels = [ten.shape[1] for ten in tensors]
        tensors_concat = torch.concat(tensors, dim=1)
        tensors_concat = augs(tensors_concat)

        results = []
        cur_c = 0
        for i in range(len(channels)):
            results.append(tensors_concat[:, cur_c:cur_c + channels[i], ...])
            cur_c += channels[i]
        return (ten for ten in results)

    def add_noise_with_cfg(self, latents, noise, 
                           ind_t, ind_prev_t, 
                           text_embeddings=None, cfg=1.0, 
                           delta_t=1, inv_steps=1,
                           is_noisy_latent=False,
                           eta=0.0):

        text_embeddings = text_embeddings.to(self.precision_t)
        if cfg <= 1.0:
            uncond_text_embedding = text_embeddings.reshape(2, -1, text_embeddings.shape[-2], text_embeddings.shape[-1])[1]

        unet = self.unet

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []

        for i in range(inv_steps): # 5
            # pred noise
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
            
            if cfg > 1.0:
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                unet_output = unet(latent_model_input, timestep_model_input, 
                                encoder_hidden_states=text_embeddings).sample
                
                uncond, cond = torch.chunk(unet_output, chunks=2)
                
                unet_output = cond + cfg * (uncond - cond) # reverse cfg to enhance the distillation
            else:
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat_.shape[0], 1).reshape(-1)
                unet_output = unet(cur_noisy_lat_, timestep_model_input, 
                                    encoder_hidden_states=uncond_text_embedding).sample

            pred_scores.append((cur_ind_t, unet_output))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t-cur_t if isinstance(self.scheduler, DDIMScheduler) else next_ind_t-cur_ind_t

            cur_noisy_lat = self.sche_func(self.scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]

    def ode_inverse(self, e, t, latents, cond, B, cfg=1.):
        if cfg == 1.:
            eps_e = self.cond_pred(latents, e, cond)
            x_t   = self.f_solver.dpm_solver_first_order_update(eps_e, e.reshape(1, 1).repeat(B, 1).reshape(-1), 
                                                                t.reshape(1, 1).repeat(B, 1).reshape(-1), latents)
        else:
            latents_input = torch.cat([latents, latents])
            eps_e = self.cond_pred(latents_input, e, cond)
            uncond, cond = torch.chunk(eps_e, chunks=2)
            eps_e = cond + cfg * (uncond - cond)
            x_t   = self.f_solver.dpm_solver_first_order_update(eps_e, e.reshape(1, 1).repeat(B, 1).reshape(-1), 
                                                                t.reshape(1, 1).repeat(B, 1).reshape(-1), latents)
        return (x_t.to(self.precision_t), eps_e.to(self.precision_t))
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, resolution=(512, 512)):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def cfg_perpneg_pred(self, latents_noisy, t, text_embeddings, 
                         weights, guidance_scale, K, B, resolution):
        H, W = resolution[0] // 8, resolution[1] // 8
        latent_model_input = latents_noisy[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 4, H, W)
        tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
        unet_output = self.unet(latent_model_input.to(self.precision_t), 
                                tt.to(self.precision_t), 
                                encoder_hidden_states=text_embeddings.to(self.precision_t)).sample

        unet_output = unet_output.reshape(1 + K, -1, 4,  )
        noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, H, W), \
                                                unet_output[1:].reshape(-1, 4, H, W)
        delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
        delta_DSD = weighted_perpendicular_aggregator(delta_noise_preds,\
                                                      weights,\
                                                      B)     
        pred_noise_neg = noise_pred_uncond + guidance_scale * delta_DSD
        return pred_noise_neg.to(self.precision_t)

    @torch.no_grad()
    def cfg_cond_pred(self, latents_noisy, t, text_embeddings, resolution, guidance_scale):
        latent_model_input = latents_noisy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
        tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
        unet_output = self.unet(latent_model_input.to(self.precision_t), 
                                tt.to(self.precision_t), 
                                encoder_hidden_states=text_embeddings.to(self.precision_t)).sample
        unet_output = unet_output.reshape(2, -1, 4, resolution[0] // 8, resolution[1] // 8, )
        noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
        delta_DSD = noise_pred_text - noise_pred_uncond
        pred_noise = noise_pred_uncond + guidance_scale * delta_DSD
        return pred_noise
    
    @torch.no_grad()
    def cond_pred(self, latent_model_input, t, text_embeddings):
        tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
        pred_noise = self.unet(latent_model_input.to(self.precision_t), 
                                tt.to(self.precision_t), 
                                encoder_hidden_states=text_embeddings.to(self.precision_t)).sample
        return pred_noise
        
    def train_step_perpneg(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                           grad_scale=1,use_control_net=False,
                           save_folder:Path=None, iteration=0, warm_up_rate = 0, weights = 0, 
                           resolution=(512, 512), guidance_opt=None,as_latent=False, embedding_inverse = None):
        # flip aug
        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        if as_latent:      
            latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
        else:
            latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        
        weights = weights.reshape(-1)
        if self.noise_temp is None:
            self.noise_temp = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
        if guidance_opt.fix_noise:
            assert self.noise_temp is not None
            noise = self.noise_temp.to(device=latents.device)
        else:
            noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])
        uncond_embeddings = inverse_text_embeddings.reshape(2, -1, inverse_text_embeddings.shape[-2], inverse_text_embeddings.shape[-1])[1]
        cond_text_embeddings = text_embeddings.reshape(K+1, -1, text_embeddings.shape[-2], text_embeddings.shape[-1])[1]

        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1]) # make it k+1, c * t, ...

        if guidance_opt.annealing_intervals:
            current_delta_t =  int(guidance_opt.delta_t + np.ceil((warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t)))
        else:
            current_delta_t =  guidance_opt.delta_t

        uncond_embeddings = uncond_embeddings.to(self.precision_t)
        text_embeddings   = text_embeddings.to(self.precision_t)

        # time definitions.
        ind_t      = torch.randint(self.min_step + int(self.warmup_step*warm_up_rate), 
                                   self.max_step + int(self.warmup_step*warm_up_rate), 
                                   (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        ind_t      = max(ind_t, torch.ones_like(ind_t) * current_delta_t + 1)
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t))
        # ind_end_t  = max(ind_t - 2 * current_delta_t, torch.ones_like(ind_t))
        end_min_t  = int(max(ind_t - 2 * current_delta_t, torch.zeros_like(ind_t)))
        end_max_t  = int(max(ind_t - 1 * current_delta_t - 10, torch.ones_like(ind_t)))
        ind_end_t  = torch.randint(end_min_t,
                                   end_max_t, 
                                   (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        ind_end_t  = max(ind_end_t, torch.ones_like(ind_t))
        self.ind_t = ind_t

        t_t = self.timesteps[ind_t]
        t_s = self.timesteps[ind_prev_t]
        t_e = self.timesteps[ind_end_t]

        # Reverse ODE.
        x_e_0  = self.scheduler.add_noise(latents, noise, t_e).to(self.precision_t)
        x_s_e, eps_e_null = self.ode_inverse(t_e, t_s, x_e_0, uncond_embeddings, B)
        x_t_s, eps_s_null = self.ode_inverse(t_s, t_t, x_s_e, uncond_embeddings, B)

        eps_t_perp = self.cfg_perpneg_pred(x_t_s, t_t, text_embeddings, 
                                           weights, guidance_opt.guidance_scale, K, B, resolution)
        x_s_t = self.f_solver.dpm_solver_first_order_update(eps_t_perp, t_t.reshape(1, 1).repeat(B, 1).reshape(-1), 
                                                            t_s.reshape(1, 1).repeat(B, 1).reshape(-1), x_t_s).to(self.precision_t)
        # calculate loss.
        loss = 0.

        if guidance_opt.w_cc > 0:
            x_e_s_null, _ = self.ode_inverse(t_s, t_e, x_s_t, uncond_embeddings, B)
            x_e_t_null, _ = self.ode_inverse(t_t, t_e, x_t_s, uncond_embeddings, B)
            L_cc = guidance_opt.w_cc * torch.nn.functional.mse_loss(x_e_t_null, x_e_s_null.detach(), reduction="sum") / B
            loss += L_cc
        
        if guidance_opt.w_gc > 0:
            x_e_t = self.f_solver.dpm_solver_first_order_update(eps_t_perp, t_t.reshape(1, 1).repeat(B, 1).reshape(-1), 
                                                                t_e.reshape(1, 1).repeat(B, 1).reshape(-1), x_t_s).to(self.precision_t)
            eps_e_perp = self.cfg_perpneg_pred(x_e_t, t_e, text_embeddings,
                                               weights, guidance_opt.guidance_scale, K, B, resolution)
            x_0_s_perp = pred_original(self.scheduler, eps_e_perp, t_e, x_e_t)
            x_0_s_null = pred_original(self.scheduler, eps_e_null, t_e, x_e_0)
            L_gc = guidance_opt.w_gc * torch.nn.functional.mse_loss(x_0_s_perp.detach(), x_0_s_null, reduction="sum") / B
            loss += L_gc
        
        if guidance_opt.w_cp > 0:
            pred_x0_s_p = self.decode_latents(x_0_s_perp.detach())
            pred_x0_s_n = self.decode_latents(x_0_s_null.detach())
            pred_div = (pred_rgb + pred_x0_s_p - pred_x0_s_n).detach()
            L_cp = guidance_opt.rgb_scale * guidance_opt.w_cp * (torch.nn.functional.mse_loss(pred_rgb.to(self.precision_t), pred_div.to(self.precision_t), reduction="sum") / B)
            loss += L_cp

        loss = loss * grad_scale

        if iteration % guidance_opt.vis_interval == 0:
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration,t_s.item()))
            with torch.no_grad():
                pred_x0_latent_t = x_0_s_perp.detach()
                pred_x0_latent_e = x_0_s_null.detach()
                pred_x0_s_p = self.decode_latents(x_0_s_perp.detach())
                pred_x0_s_n = self.decode_latents(x_0_s_null.detach())
                pred_x0_pos = pred_x0_s_p
                pred_x0_sp  = pred_x0_s_n
                grad = (pred_x0_s_p - pred_x0_s_n).detach()
                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                latents_t_rgb = F.interpolate(lat2rgb(pred_x0_latent_e), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
                latents_e_rgb = F.interpolate(lat2rgb(pred_x0_latent_t), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)

                viz_images = torch.cat([pred_rgb, 
                                        pred_depth.repeat(1, 3, 1, 1), 
                                        pred_alpha.repeat(1, 3, 1, 1), 
                                        rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                        latents_t_rgb, latents_e_rgb, 
                                        norm_grad,
                                        pred_x0_sp, pred_x0_pos],dim=0) 
                save_image(viz_images, save_path_iter)
        return loss

    def train_step(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                        grad_scale=1,use_control_net=False,
                        save_folder:Path=None, iteration=0, warm_up_rate = 0,
                        resolution=(512, 512), guidance_opt=None,as_latent=False, embedding_inverse = None):

         # flip aug
        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        if as_latent:      
            latents,_ = self.encode_imgs(pred_depth.repeat(1,3,1,1).to(self.precision_t))
        else:
            latents,_ = self.encode_imgs(pred_rgb.to(self.precision_t))
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        
        noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])
        uncond_embeddings = inverse_text_embeddings.reshape(2, -1, inverse_text_embeddings.shape[-2], inverse_text_embeddings.shape[-1])[1]
        cond_text_embeddings   = text_embeddings.reshape(2, -1, inverse_text_embeddings.shape[-2], inverse_text_embeddings.shape[-1])[1]
        text_embeddings   = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1]) # make it k+1, c * t, ...

        if guidance_opt.annealing_intervals:
            current_delta_t =  int(guidance_opt.delta_t + np.ceil((warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t)))
        else:
            current_delta_t =  guidance_opt.delta_t

        uncond_embeddings = uncond_embeddings.to(self.precision_t)
        text_embeddings   = text_embeddings.to(self.precision_t)

        # time definitions.
        ind_t      = torch.randint(self.min_step + int(self.warmup_step*warm_up_rate), 
                                   self.max_step + int(self.warmup_step*warm_up_rate), 
                                   (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        ind_t      = max(ind_t, torch.ones_like(ind_t) * current_delta_t + 1)
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t))
        # ind_end_t  = max(ind_t - 2 * current_delta_t, torch.ones_like(ind_t))
        end_min_t  = int(max(ind_t - 2 * current_delta_t, torch.zeros_like(ind_t)))
        end_max_t  = int(max(ind_t - 1 * current_delta_t - 10, torch.ones_like(ind_t)))
        ind_end_t  = torch.randint(end_min_t,
                                   end_max_t, 
                                   (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        ind_end_t  = max(ind_end_t, torch.ones_like(ind_t))
        self.ind_t = ind_t

        t_t = self.timesteps[ind_t]
        t_s = self.timesteps[ind_prev_t]
        t_e = self.timesteps[ind_end_t]

        x_e_0  = self.scheduler.add_noise(latents, noise, t_e).to(self.precision_t)
        x_s_e, eps_e_null = self.ode_inverse(t_e, t_s, x_e_0, uncond_embeddings, B)
        x_t_s, _ = self.ode_inverse(t_s, t_t, x_s_e, uncond_embeddings, B)

        with torch.no_grad():
            eps_t_perp = self.cfg_cond_pred(x_t_s, t_t, text_embeddings, resolution, guidance_opt.guidance_scale)
        x_s_t = self.f_solver.dpm_solver_first_order_update(eps_t_perp, t_t.reshape(1, 1).repeat(B, 1).reshape(-1), 
                                                            t_s.reshape(1, 1).repeat(B, 1).reshape(-1), x_t_s).to(self.precision_t)
        # calculate loss.
        loss = 0.
        if guidance_opt.w_cc > 0:
            x_e_s_null, _ = self.ode_inverse(t_s, t_e, x_s_t, uncond_embeddings, B)
            x_e_t_null, _ = self.ode_inverse(t_t, t_e, x_t_s, uncond_embeddings, B)
            L_cc = guidance_opt.w_cc * torch.nn.functional.mse_loss(x_e_t_null, x_e_s_null.detach(), reduction="sum") / B
            loss += L_cc
        
        if guidance_opt.w_gc > 0:
            x_e_t = self.f_solver.dpm_solver_first_order_update(eps_t_perp, t_t.reshape(1, 1).repeat(B, 1).reshape(-1), 
                                                                t_e.reshape(1, 1).repeat(B, 1).reshape(-1), x_t_s).to(self.precision_t)
            eps_e_perp = self.cfg_cond_pred(x_e_t, t_e, text_embeddings, resolution, guidance_opt.guidance_scale)
            x_0_s_perp = pred_original(self.scheduler, eps_e_perp, t_e, x_e_t)
            x_0_s_null = pred_original(self.scheduler, eps_e_null, t_e, x_e_0)
            L_gc = guidance_opt.w_gc * torch.nn.functional.mse_loss(x_0_s_perp.detach(), x_0_s_null, reduction="sum") / B
            loss += L_gc
        
        if guidance_opt.w_cp > 0:
            pred_x0_s_p = self.decode_latents(x_0_s_perp.detach())
            pred_x0_s_n = self.decode_latents(x_0_s_null.detach())
            pred_div = (pred_rgb + pred_x0_s_p - pred_x0_s_n).detach()
            L_cp = guidance_opt.rgb_scale * guidance_opt.w_cp * (torch.nn.functional.mse_loss(pred_rgb.to(self.precision_t), pred_div.to(self.precision_t), reduction="sum") / B)
            loss += L_cp

        loss = loss * grad_scale

        if iteration % guidance_opt.vis_interval == 0:
            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration,t_s.item()))
            with torch.no_grad():
                pred_x0_latent_t = x_0_s_perp.detach()
                pred_x0_latent_e = x_0_s_null.detach()
                pred_x0_s_p = self.decode_latents(x_0_s_perp.detach())
                pred_x0_s_n = self.decode_latents(x_0_s_null.detach())
                pred_x0_pos = pred_x0_s_p
                pred_x0_sp  = pred_x0_s_n
                grad = (pred_x0_s_p - pred_x0_s_n).detach()
                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                latents_t_rgb = F.interpolate(lat2rgb(pred_x0_latent_e), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
                latents_e_rgb = F.interpolate(lat2rgb(pred_x0_latent_t), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)

                viz_images = torch.cat([pred_rgb, 
                                        pred_depth.repeat(1, 3, 1, 1), 
                                        pred_alpha.repeat(1, 3, 1, 1), 
                                        rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                        latents_t_rgb, latents_e_rgb, 
                                        norm_grad,
                                        pred_x0_sp, pred_x0_pos],dim=0) 
                save_image(viz_images, save_path_iter)
        return loss

    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor

        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs.to(target_dtype)

    def encode_imgs(self, imgs):
        target_dtype = imgs.dtype
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        kl_divergence = posterior.kl()

        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(target_dtype), kl_divergence