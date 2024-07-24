import torch
import numpy as np
from contextlib import nullcontext


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0

def predicted_prev_given_current(model_output, timesteps_t, timesteps_s, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas_t = extract_into_tensor(sigmas, timesteps_t, sample.shape)
        alphas_t = extract_into_tensor(alphas, timesteps_t, sample.shape)
        pred_x_0 = (sample - sigmas_t * model_output) / alphas_t
        sigmas_s = extract_into_tensor(sigmas, timesteps_s, sample.shape)
        alphas_s = extract_into_tensor(alphas, timesteps_s, sample.shape)
        pred_x_s_t = alphas_s * pred_x_0 + sigmas_s * model_output
    elif prediction_type == "v_prediction":
        return NotImplementedError
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")
    return pred_x_s_t

class FSolver:
    def __init__(
        self,
        alpha_schedule,
        sigma_schedule,
        num_train_timesteps,
        prediction_type,
        solver_type,
        solver_order,
        precision_t
    ):
        self.alphas = alpha_schedule
        self.sigmas = sigma_schedule
        self.num_train_timesteps = num_train_timesteps
        
        self.prediction_type = prediction_type
        self.solver_type = solver_type
        self.solver_order = solver_order
        self.precision_t = precision_t

    def predicted_prev_given_current_ddim(self, model_output, timesteps_t, timesteps_s, sample):
        return self._predicted_prev_given_current_ddim(model_output, timesteps_t, timesteps_s, sample)
        
    def predicted_prev_given_current(
        self, 
        unet, 
        timesteps_t, 
        timesteps_s, 
        sample, 
        prompt_embeds, 
        added_cond_kwargs, 
        require_autocast=False, 
        weight_dtype=None,
    ):
        if self.solver_type == "ddim":
            model_output = self.model_fn(unet, sample, timesteps_t, prompt_embeds, added_cond_kwargs, require_autocast, weight_dtype)
            pred_x_s_t = self._predicted_prev_given_current_ddim(model_output, timesteps_t, timesteps_s, sample)
        elif "dpmsolver" in self.solver_type:
            if self.solver_order == 1:
                with torch.no_grad():
                    model_output = self.model_fn(unet, sample, timesteps_t, prompt_embeds, added_cond_kwargs, require_autocast, weight_dtype)
                pred_x_s_t = self._dpm_solver_first_order_update(model_output, timesteps_t, timesteps_s, sample)
            elif self.solver_order == 2:
                pred_x_s_t = self._singlestep_dpm_solver_second_order_update(
                    unet, 
                    timesteps_t, 
                    timesteps_s, 
                    sample,
                    prompt_embeds,
                    added_cond_kwargs,
                    require_autocast,
                    weight_dtype,
                )
            else:
                raise NotImplementedError
        else:
            raise ValueError(f"Solver type {self.solver_type} currently not supported.")

        return pred_x_s_t

    def predicted_prev_given_current_with_cfg(
        self, 
        unet, 
        timesteps_t, 
        timesteps_s, 
        sample, 
        prompt_embeds, 
        added_cond_kwargs, 
        cfg,
        require_autocast=False, 
        weight_dtype=None,
    ):
        if self.solver_type == "ddim":
            raise NotImplementedError
        elif "dpmsolver" in self.solver_type:
            if self.solver_order == 1:
                with torch.no_grad():
                    model_input  = torch.cat([sample, sample])
                    t_input      = timesteps_t[0].reshape(1, 1).repeat(model_input.shape[0], 1).reshape(-1)
                    model_output = self.model_fn(unet, model_input, t_input, prompt_embeds, added_cond_kwargs, require_autocast, weight_dtype)
                    uncond, cond = torch.chunk(model_output, chunks=2)
                    model_output = cond + cfg * (uncond - cond)
                pred_x_s_t = self._dpm_solver_first_order_update(model_output, timesteps_t, timesteps_s, sample)
            elif self.solver_order == 2:
                pred_x_s_t = self._singlestep_dpm_solver_second_order_update(
                    unet, 
                    timesteps_t, 
                    timesteps_s, 
                    sample,
                    prompt_embeds,
                    added_cond_kwargs,
                    require_autocast,
                    weight_dtype,
                )
            else:
                raise NotImplementedError
        else:
            raise ValueError(f"Solver type {self.solver_type} currently not supported.")

        return pred_x_s_t
    
    def dpm_solver_first_order_update(self, model_output, timesteps_t, timesteps_s, sample):
        return self._dpm_solver_first_order_update(model_output, timesteps_t, timesteps_s, sample)

    def _predicted_prev_given_current_ddim(self, model_output, timesteps_t, timesteps_s, sample):
        if self.prediction_type == "epsilon":
            sigmas_t = extract_into_tensor(self.sigmas, timesteps_t, sample.shape)
            alphas_t = extract_into_tensor(self.alphas, timesteps_t, sample.shape)
            pred_x_0 = (sample - sigmas_t * model_output) / alphas_t
            sigmas_s = extract_into_tensor(self.sigmas, timesteps_s, sample.shape)
            alphas_s = extract_into_tensor(self.alphas, timesteps_s, sample.shape)
            pred_x_s_t = alphas_s * pred_x_0 + sigmas_s * model_output
        elif self.prediction_type == "v_prediction":
            return NotImplementedError
        else:
            raise ValueError(f"Prediction type {self.prediction_type} currently not supported.")

        return pred_x_s_t

    def _dpm_solver_first_order_update(self, model_output, timesteps_t, timesteps_s, sample):
        sigmas_t = extract_into_tensor(self.sigmas, timesteps_t, sample.shape)
        alphas_t = extract_into_tensor(self.alphas, timesteps_t, sample.shape)

        sigmas_s = extract_into_tensor(self.sigmas, timesteps_s, sample.shape)
        alphas_s = extract_into_tensor(self.alphas, timesteps_s, sample.shape)

        lambda_t = torch.log(alphas_t) - torch.log(sigmas_t)
        lambda_s = torch.log(alphas_s) - torch.log(sigmas_s)
        
        h = lambda_s - lambda_t

        model_output = self.convert_model_output(model_output, sample, alphas_t, sigmas_t)
        if self.solver_type == "dpmsolver++":
            pred_x_s_t = (sigmas_s / sigmas_t) * sample - alphas_s * torch.expm1(-h) * model_output
        elif self.solver_type == "dpmsolver":
            pred_x_s_t = (alphas_s / alphas_t) * sample - sigmas_s * torch.expm1(h) * model_output
        # t -> s
        return pred_x_s_t

    def singlestep_dpm_solver_second_order_update(self, unet, timesteps_t, timesteps_s, sample, prompt_embeds):
        return self._singlestep_dpm_solver_second_order_update(unet, timesteps_t, timesteps_s, sample, prompt_embeds, None, False, None)

    def _singlestep_dpm_solver_second_order_update(
        self, 
        unet, 
        timesteps_t, 
        timesteps_s, 
        sample, 
        prompt_embeds, 
        added_cond_kwargs, 
        require_autocast=False, 
        weight_dtype=None,
        r1=0.5, 
    ):
        alphas_t = extract_into_tensor(self.alphas, timesteps_t, sample.shape)
        sigmas_t = extract_into_tensor(self.sigmas, timesteps_t, sample.shape)

        alphas_s = extract_into_tensor(self.alphas, timesteps_s, sample.shape)
        sigmas_s = extract_into_tensor(self.sigmas, timesteps_s, sample.shape)

        lambdas_t = torch.log(alphas_t) - torch.log(sigmas_t)
        lambdas_s = torch.log(alphas_s) - torch.log(sigmas_s)

        h = lambdas_s - lambdas_t
        lambda_s1 = lambdas_t + r1 * h

        alphas_s1 = (1 / (torch.exp(-2. * lambda_s1) + 1)) ** 0.5
        sigmas_s1 = (1 - alphas_s1 ** 2) ** 0.5
        # inverse_lambda
        timesteps_s1 = interpolate_fn(
            alphas_s1.reshape((-1, 1)), 
            torch.flip(self.alphas.reshape((1, -1)).to(timesteps_t.device), [1]), 
            torch.flip(torch.arange(0, self.num_train_timesteps).reshape(1, -1).to(timesteps_t.device), [1])
        )
        timesteps_s1 = timesteps_s1.squeeze()
        timesteps_s1 = timesteps_s1.type(timesteps_t.dtype)

        if self.solver_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)
            with torch.no_grad():
                model_t = self.model_fn(
                    unet,
                    sample,
                    timesteps_t,
                    prompt_embeds,
                    added_cond_kwargs,
                    require_autocast,
                    weight_dtype,
                )
            model_t = self.convert_model_output(model_t, sample, alphas_t, sigmas_t)

            sample_s1 = (
                (sigmas_s1 / sigmas_t) * sample
                - (alphas_s1 * phi_11) * model_t
            ).type(sample.dtype)
            
            with torch.no_grad():
                model_s1 = self.model_fn(
                    unet,
                    sample_s1,
                    timesteps_s1,
                    prompt_embeds,
                    added_cond_kwargs,
                    require_autocast,
                    weight_dtype,
                )
            model_s1 = self.convert_model_output(model_s1, sample_s1, alphas_s1, sigmas_s1)

            sample_t = (
                (sigmas_s / sigmas_t) * sample
                - (alphas_s * phi_1) * model_t
                - (0.5 / r1) * (alphas_s * phi_1) * (model_s1 - model_t)
            )
        elif self.solver_type == "dpmsolver":
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)
            
            with torch.no_grad():
                model_t = self.model_fn(
                    unet,
                    sample,
                    timesteps_t,
                    prompt_embeds,
                    added_cond_kwargs,
                    require_autocast,
                    weight_dtype,
                )
            model_t = self.convert_model_output(model_t, sample, alphas_t, sigmas_t)

            sample_s1 = (
                (alphas_s1 / alphas_t) * sample
                - (sigmas_s1 * phi_11) * model_t
            )
            with torch.no_grad():
                model_s1 = self.model_fn(
                    unet,
                    sample_s1,
                    timesteps_s1,
                    prompt_embeds,
                    added_cond_kwargs,
                    require_autocast,
                    weight_dtype,
                )
            model_s1 = self.convert_model_output(model_s1, sample_s1, alphas_s1, sigmas_s1)

            sample_t = (
                (alphas_s - alphas_t) * sample
                - (sigmas_s * phi_1) * model_t
                - (0.5 / r1) * (sigmas_s * phi_1) * (model_s1 - model_t)
            )

        return sample_t

    def model_fn(
        self,
        unet, 
        noisy_model_input, 
        timesteps, 
        prompt_embeds, 
        added_cond_kwargs, 
        require_autocast=False, 
        weight_dtype=None
    ):
        context_autocast = torch.autocast("cuda", enabled=True, dtype=weight_dtype) if require_autocast else nullcontext()

        with context_autocast:
            noise_pred = unet(
                noisy_model_input,
                timesteps,
                timestep_cond=None,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
        
        return noise_pred

    def convert_model_output(
        self,
        model_output,
        sample,
        alpha_t,
        sigma_t,
    ) -> torch.FloatTensor:
        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.solver_type == "dpmsolver++":
            if self.prediction_type == "epsilon":
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.prediction_type == "sample":
                x0_pred = model_output
            elif self.prediction_type == "v_prediction":
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction`."
                )

            return x0_pred

        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.solver_type == "dpmsolver":
            if self.prediction_type == "epsilon":
                epsilon = model_output
            elif self.prediction_type == "sample":
                epsilon = (sample - alpha_t * model_output) / sigma_t
            elif self.prediction_type == "v_prediction":
                epsilon = alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError(
                    f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction`."
                )

            return epsilon

class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000):
        # DDIM sampling parameters
        self.ddim_timesteps = (np.arange(1, timesteps + 1)).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev

