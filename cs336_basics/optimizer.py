import torch
from torch.optim import Optimizer
from typing import Iterable, Optional, Callable
import math

class AdamW(Optimizer):
    """
    AdamW optimizer implementation.
    
    AdamW decouples weight decay from the gradient-based update,
    applying weight decay directly to the parameters.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
    """
    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01 
        ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure = None):
        """
        single optimization step
        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                # get or initialize state for this param
                state = self.state[p]
                # State init
                if (len(state) == 0):
                    state['step'] = 0
                    # exp moving ave of grad values
                    state['exp_avg'] = torch.zeros_like(p)
                    # exp moving ave of sq grad vals
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1

                # update bias. 更新一阶矩 (m - 动量)：计算梯度的指数移动平均值，这可以理解为梯度的“惯性”方向 。
                exp_avg.mul_(beta1).add_(grad, alpha = 1-beta1)

                # 更新二阶矩 (v - 自适应项)：计算梯度平方的指数移动平均值，这可以理解为梯度变化的剧烈程度 。
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                
                # 偏差校正 (Bias Correction)：由于 m 和 v 初始为0，在训练初期它们会偏向于0。此步骤会计算一个校正后的学习率 α_t​来修正这个偏差。
                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2

                # update params 
                # 参数更新: 使用校正后的学习率以及 m 和 v 来更新参数。梯度小且稳定的参数会获得更大的更新步伐，反之亦然 。
                p.addcdiv_(exp_avg_corrected, exp_avg_sq_corrected.sqrt().add_(eps), value=-lr)

                # 解耦权重衰减 (Decoupled Weight Decay)：在上述步骤完成后，独立地对参数进行权重衰减，将其值向0拉近。这是 AdamW 与 Adam 的关键区别 。
                p.mul_(1-lr*weight_decay)

        return loss

class SGD(Optimizer):
    """
    stochastic gradient descent optmizer with lr decay

    Args:
        params: iterable of params to optmize
        lr: learning rate (default 1e -3)
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"] # get the lr
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # get state associated with p
                t = state.get("t", 0) # get iteration number from the state, or initial value
                grad = p.grad.date # get gradient of loss with respect of p

                # update weight tensor in-place iwth decaying lr
                p.data -= lr/math.sqrt(t+1) * grad

                state["t"] = t + 1 # increment iteration number

        return loss
