"""Matched hard-initial-condition NN and PINN estimators."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value)) if value < 30 else value


class HardICNetwork(nn.Module):
    """Two-state log-space network that satisfies both ICs exactly."""

    def __init__(
        self,
        initial: tuple[float, float],
        final_time: float,
        real: bool,
        architecture: str = "single",
        initial_rates: tuple[float, float] = (0.01, 0.3),
        initial_D_abs: float = 4.0,
    ):
        super().__init__()
        if architecture not in {"single", "piecewise"}:
            raise ValueError(f"Unknown architecture: {architecture}")
        if real and architecture == "piecewise":
            raise ValueError("The treatment-only real model has no phase discontinuity")
        self.architecture = architecture
        if architecture == "single":
            self.net = self._make_net()
        else:
            self.nets = nn.ModuleList([self._make_net(), self._make_net(), self._make_net()])
        self.register_buffer("log_initial", torch.log(torch.tensor(initial)).reshape(1, 2))
        self.final_time = float(final_time)
        self.real = real
        self.raw_alpha = nn.Parameter(torch.tensor(_inverse_softplus(initial_rates[0])))
        self.raw_beta = nn.Parameter(torch.tensor(_inverse_softplus(initial_rates[1])))
        if real:
            self.raw_D = nn.Parameter(torch.tensor(_inverse_softplus(initial_D_abs)))

    @staticmethod
    def _make_net() -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2)
        )

    def states(self, time_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.architecture == "single":
            tau = time_h / self.final_time
            output = self.log_initial + tau * self.net(tau)
        else:
            tau1 = time_h / 2.0
            endpoint = torch.ones((1, 1), dtype=time_h.dtype, device=time_h.device)
            boundary1 = self.log_initial + self.nets[0](endpoint)
            tau2 = (time_h - 2.0) / 5.0
            boundary2 = boundary1 + self.nets[1](endpoint)
            tau3 = (time_h - 7.0) / 5.0
            phase1 = self.log_initial + tau1 * self.nets[0](tau1)
            phase2 = boundary1 + tau2 * self.nets[1](tau2)
            phase3 = boundary2 + tau3 * self.nets[2](tau3)
            output = torch.where(time_h <= 2.0, phase1, torch.where(time_h <= 7.0, phase2, phase3))
        return output[:, 0:1], output[:, 1:2]

    @property
    def alpha(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_alpha) + 1e-10

    @property
    def beta(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_beta) + 1e-10

    @property
    def D(self) -> torch.Tensor:
        if not self.real:
            raise AttributeError("Synthetic model has no fitted D")
        return -torch.nn.functional.softplus(self.raw_D) - 1e-10


@dataclass
class NeuralResult:
    model: HardICNetwork
    final_loss: float
    epochs: int


def _total_log(model: HardICNetwork, time_h: torch.Tensor) -> torch.Tensor:
    logN, logP = model.states(time_h)
    return torch.logsumexp(torch.cat([logN, logP], dim=1), dim=1, keepdim=True)


def _physics_residuals(
    model: HardICNetwork, collocation: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    logN, logP = model.states(collocation)
    dlogN = torch.autograd.grad(logN, collocation, torch.ones_like(logN), create_graph=True)[0]
    dlogP = torch.autograd.grad(logP, collocation, torch.ones_like(logP), create_graph=True)[0]
    ratio_PN = torch.exp(torch.clamp(logP - logN, -50.0, 50.0))
    ratio_NP = torch.exp(torch.clamp(logN - logP, -50.0, 50.0))
    alpha, beta = model.alpha, model.beta
    if model.real:
        rhs_N = model.D - alpha + beta * ratio_PN
    else:
        antibiotic = ((collocation >= 2.0) & (collocation <= 7.0)).to(collocation.dtype)
        rhs_N = 1.0 - 2.5 * antibiotic - alpha + beta * ratio_PN
    rhs_P = alpha * ratio_NP - beta
    return dlogN - rhs_N, dlogP - rhs_P


def _physics_loss(
    model: HardICNetwork,
    collocation: torch.Tensor,
    variant: str = "global",
    residual_time_scale: float = 1.0,
) -> torch.Tensor:
    residual_N, residual_P = _physics_residuals(model, collocation)
    residual_N = residual_N * residual_time_scale
    residual_P = residual_P * residual_time_scale
    if variant == "global" or model.real:
        return torch.mean(residual_N**2) + torch.mean(residual_P**2)
    if variant != "phase_balanced":
        raise ValueError(f"Unknown physics loss variant: {variant}")
    masks = (collocation < 2.0, (collocation > 2.0) & (collocation < 7.0), collocation > 7.0)
    phase_losses = []
    for mask in masks:
        selected_N = residual_N[mask]
        selected_P = residual_P[mask]
        if selected_N.numel() == 0:
            raise ValueError("Each phase must contain collocation points")
        phase_losses.append(torch.mean(selected_N**2) + torch.mean(selected_P**2))
    return torch.stack(phase_losses).mean()


def _make_collocation(
    final_time: float, points: int, strategy: str, real: bool
) -> torch.Tensor:
    if strategy == "uniform" or real:
        result = torch.linspace(0.0, final_time, points).reshape(-1, 1)
    elif strategy == "phase_stratified":
        per_phase = max(4, points // 3)
        epsilon = 1e-6
        pieces = [
            torch.linspace(epsilon, 2.0 - epsilon, per_phase),
            torch.linspace(2.0 + epsilon, 7.0 - epsilon, per_phase),
            torch.linspace(7.0 + epsilon, final_time - epsilon, per_phase),
        ]
        result = torch.cat(pieces).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown collocation strategy: {strategy}")
    result.requires_grad_(True)
    return result


def fit_neural(
    times: np.ndarray,
    log_y: np.ndarray,
    initial: tuple[float, float],
    final_time: float,
    seed: int,
    physics: bool,
    real: bool,
    epochs: int = 5000,
    data_weight: float = 0.1,
    collocation_points: int = 150,
    lbfgs_steps: int = 250,
    architecture: str = "single",
    physics_variant: str = "global",
    collocation_strategy: str = "uniform",
    initial_rates: tuple[float, float] = (0.01, 0.3),
    initial_D_abs: float = 4.0,
    residual_time_scale: float = 1.0,
) -> NeuralResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HardICNetwork(
        initial,
        final_time,
        real=real,
        architecture=architecture,
        initial_rates=initial_rates,
        initial_D_abs=initial_D_abs,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.55), int(epochs * 0.82)], gamma=0.35
    )
    t_train = torch.as_tensor(times).reshape(-1, 1)
    y_train = torch.as_tensor(log_y).reshape(-1, 1)
    collocation = _make_collocation(final_time, collocation_points, collocation_strategy, real)
    final_loss = float("nan")
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        data_loss = torch.mean((_total_log(model, t_train) - y_train) ** 2)
        if physics:
            loss = data_weight * data_loss + _physics_loss(
                model, collocation, physics_variant, residual_time_scale
            )
        else:
            loss = data_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        final_loss = float(loss.detach())

    if physics and lbfgs_steps > 0:
        optimizer_second = torch.optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=lbfgs_steps,
            max_eval=int(lbfgs_steps * 1.25),
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer_second.zero_grad(set_to_none=True)
            data_loss = torch.mean((_total_log(model, t_train) - y_train) ** 2)
            loss = data_weight * data_loss + _physics_loss(
                model, collocation, physics_variant, residual_time_scale
            )
            loss.backward()
            return loss

        final_loss = float(optimizer_second.step(closure).detach())
        final_loss = float(closure().detach())
    return NeuralResult(model=model, final_loss=final_loss, epochs=epochs)


def predict_neural(result: NeuralResult, times: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.as_tensor(np.asarray(times, dtype=float)).reshape(-1, 1)
        return _total_log(result.model, tensor).numpy().ravel()


def fitted_parameters(result: NeuralResult) -> dict[str, float]:
    parameters = {
        "alpha": float(result.model.alpha.detach()),
        "beta": float(result.model.beta.detach()),
    }
    if result.model.real:
        parameters["D"] = float(result.model.D.detach())
    return parameters


def initial_condition_error(result: NeuralResult) -> float:
    with torch.no_grad():
        time_zero = torch.zeros((1, 1))
        logN, logP = result.model.states(time_zero)
        recovered = torch.exp(torch.cat([logN, logP], dim=1)).numpy().ravel()
        expected = np.exp(result.model.log_initial.numpy()).ravel()
    return float(np.max(np.abs(recovered - expected)))


def physics_diagnostics(
    result: NeuralResult,
    final_time: float,
    points_per_phase: int = 300,
) -> dict[str, float]:
    """Return dense physical-time residual RMSE values, excluding switches."""
    model = result.model
    if model.real:
        collocation = _make_collocation(final_time, points_per_phase, "uniform", True)
        residual_N, residual_P = _physics_residuals(model, collocation)
        return {
            "all_RN_rmse_h-1": float(torch.sqrt(torch.mean(residual_N**2)).detach()),
            "all_RP_rmse_h-1": float(torch.sqrt(torch.mean(residual_P**2)).detach()),
        }
    collocation = _make_collocation(final_time, 3 * points_per_phase, "phase_stratified", False)
    residual_N, residual_P = _physics_residuals(model, collocation)
    labels = {
        "pre": collocation < 2.0,
        "treatment": (collocation > 2.0) & (collocation < 7.0),
        "post": collocation > 7.0,
    }
    diagnostics = {}
    for label, mask in labels.items():
        diagnostics[f"{label}_RN_rmse_h-1"] = float(
            torch.sqrt(torch.mean(residual_N[mask] ** 2)).detach()
        )
        diagnostics[f"{label}_RP_rmse_h-1"] = float(
            torch.sqrt(torch.mean(residual_P[mask] ** 2)).detach()
        )
    return diagnostics
