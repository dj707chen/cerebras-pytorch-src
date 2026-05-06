# Copyright 2016-2025 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import torch

import cerebras.pytorch as cstorch
from cerebras.pytorch.backend import current_backend_impl

from .optimizer import Optimizer, ParamsT


class Muon(Optimizer):
    """
    Muon optimizer with distributed Newton–Schulz functionality implemented to
    conform to execution within the constraints of the Cerebras WSE, including
    pre-initializing optimizer state. The Muon optimizer enhances standard SGD
    with momentum by applying a Newton–Schulz iteration to orthogonalize the
    update matrices. Specifically, it replaces each 2D parameter's update with
    the nearest orthogonal matrix, effectively approximating the operation G →
    UV.mT, where G is the gradient matrix and U \sum V.mT is its singular value
    decomposition. This orthogonalization step improves the conditioning of
    updates, leading to better convergence properties.
    Reference: https://kellerjordan.github.io/posts/muon/
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= weight_decay < 1.0:
            raise ValueError(
                f"Invalid weight decay parameter: {weight_decay} - should be in [0.0, 1.0]"
            )
        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                f"Invalid momentum parameter: {momentum} - should be in [0.0, 1.0]"
            )
        if ns_steps < 0:
            raise ValueError(
                f"Invalid number of steps: {ns_steps} - should be >= 0"
            )

        self.backend = current_backend_impl()

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the optimizer step. This is required for the Muon optimizer to
        function correctly on the Cerebras WSE.
        """
        for group in self.param_groups:
            if group["momentum"] != 0:
                for p in group["params"]:
                    self.state[p]["momentum_buffer"] = cstorch.zeros_like(p)

    def _adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * (max(A, B) ** 0.5)
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def _zeropower_via_newtonschulz5(
        G: torch.Tensor, steps: int = 5
    ) -> torch.Tensor:
        if G.ndim < 2:
            raise ValueError(
                f"G has {G.ndim} dimensions. Requires G has 2 or more dimensions"
            )
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        need_transpose: bool = G.size(-2) > G.size(-1)
        if need_transpose:
            X = X.mT

        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X

        if need_transpose:
            X = X.mT
        return X

    def _distributed_zeropower_via_newtonschulz5(
        self, G: torch.Tensor, steps: int = 5
    ) -> torch.Tensor:
        """
        ### Distributed Newton-Schulz Algorithm ###
        Code Segment to be distributed in Newton-Schulz:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

        ### Function Definition ###
        This function implements the Newton-Schulz algorithm to compute the
        nearest orthogonal matrix to a given matrix G.
        Args:
            G (torch.Tensor): The input matrix to be orthogonalized.
            steps (int): The number of iterations to perform.
        Expects:
            - The input matrix G should have at least 2 dimensions.
        Returns:
            torch.Tensor: The orthogonalized matrix.
        """

        mesh: cstorch.mesh.Mesh = cstorch.mesh.Mesh(0, [])

        if G.ndim < 2:
            raise ValueError(
                f"G has {G.ndim} dimensions. Requires G has 2 or more dimensions"
            )
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        need_transpose: bool = G.size(-2) > G.size(-1)
        if need_transpose:
            X = X.mT
        X = X / (torch.norm(X, dim=(-2, -1), keepdim=True) + 1e-7)

        outer_dim_sharded_array = [
            cstorch.mesh.shard(0),
            cstorch.mesh.replicate(),
        ]  # shard on outer dimension (OuterDimension)
        inner_dim_sharded_array = [
            cstorch.mesh.replicate(),
            cstorch.mesh.shard(0),
        ]  # shard on inner dimension (InnerDimension)
        unsharded_array = [cstorch.mesh.replicate(), cstorch.mesh.replicate()]
        unsharded_tensor_sharding_spec = cstorch.mesh.sharding_spec(
            unsharded_array, mesh=mesh
        )

        for _ in range(steps):
            X_gh = cstorch.mesh.distribute_tensor(
                X, unsharded_tensor_sharding_spec, names=[]
            )

            XT = X.mT
            XT_gh = cstorch.mesh.distribute_tensor(
                XT, unsharded_tensor_sharding_spec, names=[]
            )

            # first matmul:
            #   X @ X.mT
            XT_gh_sliced = cstorch.mesh.all_slice(
                XT_gh, inner_dim_sharded_array, names=["InnerDimension"]
            )
            XT_sliced_wse = cstorch.mesh.d2d_transfer(XT_gh_sliced, "WSE")

            # result of the first matmul is stored on WSE
            A_result_wse = X_gh @ XT_sliced_wse

            # second matmul:
            #   A @ A
            A_gh = cstorch.mesh.d2d_transfer(A_result_wse, "GlobalHost")
            A_gh = cstorch.mesh.all_gather(A_gh, [1])
            A_gh_sliced = cstorch.mesh.all_slice(
                A_gh, inner_dim_sharded_array, names=["InnerDimension"]
            )
            A_wse = cstorch.mesh.d2d_transfer(A_gh_sliced, "WSE")

            # result of second matmul + element-wise ops is stored on WSE
            A_matmul_result_wse = A_gh @ A_wse
            B_result_wse = b * A_wse + c * A_matmul_result_wse

            # third matmul:
            #   B @ X
            B_gh = cstorch.mesh.d2d_transfer(B_result_wse, "GlobalHost")
            B_gh = cstorch.mesh.all_gather(B_gh, [1])

            X_gh_sliced = cstorch.mesh.all_slice(
                X_gh, inner_dim_sharded_array, names=["InnerDimension"]
            )
            X_wse = cstorch.mesh.d2d_transfer(X_gh_sliced, "WSE")

            # result of third matmul + element-wise ops is stored on WSE
            B_matmul_result_wse = B_gh @ X_wse
            X_result_wse = a * X_wse + B_matmul_result_wse
            X_result_gh = cstorch.mesh.d2d_transfer(X_result_wse, "GlobalHost")
            X_result_gh = cstorch.mesh.all_gather(X_result_gh, [1])
            X = X_result_gh

        if need_transpose:
            X = X.mT
        return X

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError(
                        "Muon does not support sparse gradients."
                    )

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]

                buf = buf * group["momentum"] + grad * (1 - group["momentum"])

                if group["nesterov"]:
                    eff_grad = grad + buf * group["momentum"]
                else:
                    eff_grad = buf

                if eff_grad.ndim == 4:
                    eff_grad = eff_grad.view(len(eff_grad), -1)

                if self.backend.backend_type.is_csx:
                    eff_grad = self._distributed_zeropower_via_newtonschulz5(
                        # eff_grad, steps=group["ns_steps"]
                        eff_grad,
                        steps=1,
                    )
                else:
                    eff_grad = self._zeropower_via_newtonschulz5(
                        eff_grad, steps=group["ns_steps"]
                    )
                if p.ndim >= 2:
                    eff_grad = eff_grad.contiguous().view_as(p)

                adjusted_lr = self._adjust_lr_for_muon(group["lr"], p.shape)
                p.mul_(1 - adjusted_lr * group["weight_decay"])
                p.add_(-adjusted_lr * eff_grad)
        return loss
