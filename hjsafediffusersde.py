from typing import Optional, Union, Callable, Dict

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import (
    at_least_ndim,
    SUPPORTED_NOISE_SCHEDULES, SUPPORTED_DISCRETIZATIONS, SUPPORTED_SAMPLING_STEP_SCHEDULE)
from .basic import DiffusionModel

class ReachableTubeReward:
    def __init__(self, beta: float = 1.0):
        """
        Initialize the reachable tube reward calculator.
        
        Args:
            beta (float): Reward scaling factor for states outside the reachable tube
        """
        self.beta = beta
        self.gamma = 2 * beta  # For gradient calculation
        
    def compute_reward(self, x_t: torch.Tensor, t: torch.Tensor, reachable_set_fn) -> torch.Tensor:
        """
        Compute the reachable tube reward for given states.
        """
        is_reachable = reachable_set_fn(x_t, t)
        reward = torch.zeros_like(t, dtype=torch.float32)
        
        # For states outside reachable set, compute penalty
        mask_unreachable = ~is_reachable
        if mask_unreachable.any():
            h_reach = self._compute_distance_to_boundary(x_t[mask_unreachable], t[mask_unreachable])
            reward[mask_unreachable] = -self.beta * (h_reach ** 2)
            
        return reward
    
    def compute_gradient(self, x_t: torch.Tensor, u_t: torch.Tensor, t: torch.Tensor, reachable_set_fn) -> torch.Tensor:
        """
        Compute the gradient of the reachable tube reward with respect to control input.
        """
        is_reachable = reachable_set_fn(x_t, t)
        grad = torch.zeros_like(u_t)
        
        mask_unreachable = ~is_reachable
        if mask_unreachable.any():
            h_reach = self._compute_distance_to_boundary(x_t[mask_unreachable], t[mask_unreachable])
            grad[mask_unreachable] = self.gamma * self._compute_boundary_gradient(
                x_t[mask_unreachable], u_t[mask_unreachable], t[mask_unreachable])
            
        return grad

    def _compute_distance_to_boundary(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute distance to the boundary of the reachable set."""
        raise NotImplementedError("Implement based on specific reachable set geometry")
    
    def _compute_boundary_gradient(self, x_t: torch.Tensor, u_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute gradient of the distance function with respect to control input."""
        raise NotImplementedError("Implement based on specific system dynamics")

class ReachableDiffusionSDE(DiffusionModel):
    def __init__(
            self,
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,
            classifier: Optional[BaseClassifier] = None,
            grad_clip_norm: Optional[float] = None,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,
            epsilon: float = 1e-3,
            diffusion_steps: int = 1000,
            discretization: Union[str, Callable] = "uniform",
            noise_schedule: Union[str, Dict[str, Callable]] = "cosine",
            noise_schedule_params: Optional[dict] = None,
            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,
            predict_noise: bool = True,
            reachable_tube_beta: float = 1.0,
            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            0, ema_rate, optim_params, device)

        self.predict_noise = predict_noise
        self.epsilon = epsilon
        self.diffusion_steps = diffusion_steps
        self.x_max = x_max.to(device) if isinstance(x_max, torch.Tensor) else x_max
        self.x_min = x_min.to(device) if isinstance(x_min, torch.Tensor) else x_min
        
        # Initialize reachable tube reward
        self.reachable_reward = ReachableTubeReward(beta=reachable_tube_beta)

        # Set up discretization
        if isinstance(discretization, str):
            if discretization in SUPPORTED_DISCRETIZATIONS.keys():
                self.t_diffusion = SUPPORTED_DISCRETIZATIONS[discretization](diffusion_steps, epsilon).to(device)
            else:
                Warning(f"Discretization method {discretization} is not supported. Using uniform discretization instead.")
                self.t_diffusion = SUPPORTED_DISCRETIZATIONS["uniform"](diffusion_steps, epsilon).to(device)
        elif callable(discretization):
            self.t_diffusion = discretization(diffusion_steps, epsilon).to(device)
        else:
            raise ValueError("discretization must be a callable or a string")

        # Set up noise schedule
        if isinstance(noise_schedule, str):
            if noise_schedule in SUPPORTED_NOISE_SCHEDULES.keys():
                self.alpha, self.sigma = SUPPORTED_NOISE_SCHEDULES[noise_schedule]["forward"](
                    self.t_diffusion, **(noise_schedule_params or {}))
            else:
                raise ValueError(f"Noise schedule {noise_schedule} is not supported.")
        elif isinstance(noise_schedule, dict):
            self.alpha, self.sigma = noise_schedule["forward"](self.t_diffusion, **(noise_schedule_params or {}))
        else:
            raise ValueError("noise_schedule must be a callable or a string")

        self.logSNR = torch.log(self.alpha / self.sigma)

    def add_noise(self, x0, t=None, eps=None):
        """Add noise to the input data."""
        t = torch.randint(self.diffusion_steps, (x0.shape[0],), device=self.device) if t is None else t
        eps = torch.randn_like(x0) if eps is None else eps

        alpha, sigma = at_least_ndim(self.alpha[t], x0.dim()), at_least_ndim(self.sigma[t], x0.dim())

        xt = alpha * x0 + sigma * eps
        xt = (1. - self.fix_mask) * xt + self.fix_mask * x0

        return xt, t, eps

    def guided_sampling_with_reachable(
            self, xt, t, alpha, sigma,
            model, reachable_set_fn,
            condition_cfg=None, w_cfg: float = 0.0,
            condition_cg=None, w_cg: float = 0.0,
            requires_grad: bool = False):
        """
        Modified guided sampling incorporating reachable tube constraints.
        """
        # Get initial prediction
        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad)

        # Apply classifier guidance
        pred, logp = self.classifier_guidance(
            xt, t, alpha, sigma, model, condition_cg, w_cg, pred)

        # Apply reachable tube guidance
        g_reach = self.reachable_reward.compute_gradient(xt, pred, t, reachable_set_fn)
        pred = pred + g_reach

        return pred, logp
    def loss(self, x0, condition=None):

        xt, t, eps = self.add_noise(x0)

        condition = self.model["condition"](condition) if condition is not None else None

        if self.predict_noise:
            loss = (self.model["diffusion"](xt, t, condition) - eps) ** 2
        else:
            loss = (self.model["diffusion"](xt, t, condition) - x0) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()
    def update(self, x0, condition=None, update_ema=True, **kwargs):
        loss = self.loss(x0, condition)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        log = {"loss": loss.item(), "grad_norm": grad_norm}

        return log
    def sample(
            self,
            prior: torch.Tensor,
            reachable_set_fn: Callable,
            solver: str = "ddpm",
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform",
            use_ema: bool = True,
            temperature: float = 1.0,
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,
            diffusion_x_sampling_steps: int = 0,
            warm_start_reference: Optional[torch.Tensor] = None,
            warm_start_forward_level: float = 0.3,
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        """
        Modified sampling method incorporating reachable tube constraints.
        """
        # Initialize sampling
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None,
            "reachable_rewards": []
        }

        model = self.model if not use_ema else self.model_ema

        # Setup initial state
        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor):
            diffusion_steps = int(warm_start_forward_level * self.diffusion_steps)
            fwd_alpha, fwd_sigma = self.alpha[diffusion_steps], self.sigma[diffusion_steps]
            xt = warm_start_reference * fwd_alpha + fwd_sigma * torch.randn_like(warm_start_reference)
        else:
            diffusion_steps = self.diffusion_steps
            xt = torch.randn_like(prior) * temperature
        
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        # Setup conditions
        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # Setup sampling schedule
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    diffusion_steps, sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule(diffusion_steps, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        alphas = self.alpha[sample_step_schedule]
        sigmas = self.sigma[sample_step_schedule]
        logSNRs = torch.log(alphas / sigmas)
        hs = torch.zeros_like(logSNRs)
        hs[1:] = logSNRs[:-1] - logSNRs[1:]
        stds = torch.zeros((sample_steps + 1,), device=self.device)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()

        buffer = []

        # Sampling loop
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):
            t = torch.full((n_samples,), sample_step_schedule[i], device=self.device)

            # Get prediction with reachable tube guidance
            pred, logp = self.guided_sampling_with_reachable(
                xt, t, alphas[i], sigmas[i],
                model, reachable_set_fn, 
                condition_vec_cfg, w_cfg, 
                condition_vec_cg, w_cg, 
                requires_grad)

            # Compute reachable tube reward for logging
            reward = self.reachable_reward.compute_reward(xt, t, reachable_set_fn)
            log["reachable_rewards"].append(reward.mean().item())

            # Clip prediction if needed
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            # Transform predictions
            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
            x_theta = pred if not self.predict_noise else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)

            # Update state based on solver
            if solver == "ddpm":
                xt = (
                    (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta) +
                    (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta
                )
                if i > 1:
                    xt += (stds[i] * torch.randn_like(xt))

            elif solver == "ddim":
                xt = (alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta)

            elif solver == "ode_dpmsolver_1":
                xt = (alphas[i - 1] / alphas[i]) * xt - sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta

            elif solver == "ode_dpmsolver++_1":
                xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "ode_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * D
                else:
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            # Fix known portion and update history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # Post-processing
        if self.classifier is not None and w_cg != 0.:
            with torch.no_grad():
                t = torch.zeros((n_samples,), dtype=torch.long, device=self.device)
                logp = self.classifier.logp(xt, t, condition_vec_cg)
            log["log_p"] = logp

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log
