import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from colossalai.kernel.triton.llama_act_combine_kernel import HAS_TRITON
from colossalai.moe._operation import MoeInGradScaler, MoeOutGradScaler
from colossalai.moe.manager import MOE_MANAGER
from colossalai.moe.utils import get_activation
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.tensor.moe_tensor.api import get_ep_rank, get_ep_size, set_moe_tensor_info

if HAS_TRITON:
    from colossalai.kernel.triton.llama_act_combine_kernel import LlamaActCombine


class MLPExperts(nn.Module):
    """
    SparseMLP is a multi-layer perceptron with sparse expert parallel layers.

    Args:
        num_experts (int): The number of experts
        hidden_size (int): The hidden size of MLP
        intermediate_size (int): The intermediate size of MLP
        expert_parallel (str, optional): The parallelism of experts. Now we have None, EP and TP.
        activation (optional): The activation function of MLP
        drop_rate (float, optional): The drop rate of MLP
        gated (bool, optional): Whether to use gated MLP
        use_kernel (bool, optional): Whether to use kernel optimization
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        expert_parallel: Optional[str] = None,
        activation: Optional[Callable] = None,
        drop_rate: Optional[float] = 0,
        gated: Optional[bool] = False,
        use_kernel: Optional[bool] = False,
    ):
        super().__init__()
        assert expert_parallel in ["EP", "TP", None]
        self.expert_parallel = expert_parallel
        self.num_total_experts = num_experts
        self.gated = gated
        self.use_kernel = use_kernel
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        # get expert parallel info
        if expert_parallel is not None:
            self.num_local_experts, self.moe_info = MOE_MANAGER.get_info(
                num_experts, use_tp=True if expert_parallel == "TP" else False
            )
            # get settings for different parallel
            self.ep_size = get_ep_size(self)
            if expert_parallel == "TP":
                intermediate_size = intermediate_size // self.ep_size
                num_experts = self.num_total_experts
            else:
                num_experts = self.num_local_experts
        else:
            self.num_local_experts = self.num_total_experts
            self.ep_size = 1

        if gated:
            print("if gated=True, activation={}".format(activation))
            self.wi_gate = nn.Parameter(
                torch.empty(
                    num_experts, hidden_size, intermediate_size * 2 if activation == "swiglu" else intermediate_size
                )
            )
            
            self.wi_up = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
            print("wi_gated shape={}".format(self.wi_gate.size()))
            print("wi shape={}".format(self.wi_up.size()))
        else:
            self.wi = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))  # 放大参数
            print("wi shape={}".format(self.wi.size()))
        self.wo = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))  # 缩放参数
        
        print("wo shape={}".format(self.wo.size()))
        self.act_name = activation
        self.act = get_activation(activation)  # 激活
        self.drop = nn.Dropout(p=drop_rate)

        if expert_parallel is not None:
            for param in self.parameters():
                set_moe_tensor_info(param, self.moe_info)

        # init param
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # expert param should be different
        if self.expert_parallel is not None:
            seed_ctx = Randomizer(get_ep_rank(self)).fork_rng(enable_cpu=True)
        else:
            seed_ctx = Randomizer(42).fork_rng(enable_cpu=True)
        with seed_ctx:
            if self.gated:
                torch.nn.init.normal_(self.wi_gate, std=math.sqrt(0.1 / self.hidden_size))
                torch.nn.init.normal_(self.wi_up, std=math.sqrt(0.1 / self.hidden_size))
            else:
                torch.nn.init.normal_(self.wi, std=math.sqrt(0.1 / self.hidden_size))
            torch.nn.init.normal_(self.wo, std=math.sqrt(0.1 / self.intermediate_size))

    def forward(
        self,
        x: torch.Tensor,
        param_slice: Tuple[slice] = (slice(None),),
        use_sparse: bool = True,
    ) -> torch.Tensor:
        """
        forward: hidden_size --> intermediate_size --> hidden_size

        Args:
            x (torch.Tensor): The input tensor of shape (num_groups, num_experts, capacity, hidden_size)

        Returns:
            torch.Tensor: The output tensor of shape (num_groups, num_experts, capacity, hidden_size)
        """
        x = MoeInGradScaler.apply(x, self.ep_size)

        e = x.size(1)
        h = x.size(-1)

        x = x.transpose(0, 1)
        inshape = x.shape
        x = x.reshape(e, -1, h)

        if self.use_kernel and use_sparse:
            seq_len = x.shape[1]
            with torch.no_grad():
                mask = x[:, :, 0] != 0.0
                mask = torch.sum(mask, dim=-1)
            x_list = []
            for i in range(e):
                x_list.append(x[i, : mask[i]])
            x = x_list

        if self.gated:
            x_gate = [torch.mm(x[i], self.wi_gate[param_slice][i]) for i in range(e)]
            x_up = [torch.mm(x[i], self.wi_up[param_slice][i]) for i in range(e)]
            print("foward self.act_name={}".format(self.act_name))
            if self.use_kernel and HAS_TRITON and self.act_name == "swiglu":
                print("self.gated & using LlamaActCombine")
                x = [LlamaActCombine.apply(x_gate[i], x_up[i]) for i in range(e)]
            else:
                print("self.gated &using self.act")
                x = [self.act(x_gate[i]) * x_up[i] for i in range(e)]
        else:
            x = [torch.mm(x[i], self.wi[param_slice][i]) for i in range(e)]
            x = [self.act(x[i]) for i in range(e)]
        x = [self.drop(x[i]) for i in range(e)]
        x = [torch.mm(x[i], self.wo[param_slice][i]) for i in range(e)]

        if self.use_kernel and use_sparse:
            for i in range(e):
                x[i] = torch.nn.functional.pad(x[i], (0, 0, 0, seq_len - x[i].shape[0]), mode="constant", value=0)

        x = torch.cat([x[i].unsqueeze(0) for i in range(e)], dim=0)
        x = x.reshape(inshape)
        x = x.transpose(0, 1).contiguous()
        x = MoeOutGradScaler.apply(x, self.ep_size)
        return x
    
    def forward_pure(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        forward: hidden_size --> intermediate_size --> hidden_size
        得到输入张量经过所有expert的输出

        Args:
            x (torch.Tensor): The input tensor of shape (num_tokens, hidden_size)

        Returns:
            torch.Tensor: The output tensor of shape (num_tokens, num_experts, hidden_size)
        """
        x = MoeInGradScaler.apply(x, self.ep_size) # num_token * h
        # print("x size {}".format(x.size()))
        x_copy = x.unsqueeze(0).repeat(self.num_experts, 1, 1) # num_expert * num_token * h
        # print("x copy size {}".format(x_copy.size()))

        # x= x.transpose(1, 2).contiguous().view(x.size()[0], x.size()[2], x.size()[1]) # num_expert  * h * num_token 
        x = torch.matmul(x_copy, self.wi_up.data) # num_expert * num_token * ffn
        # print("after enlarge {}".format(x.size()))
        # print(self.act)
        # x = self.act(x)
        x_gate = torch.matmul(x_copy, self.wi_gate.data)
        # print("x gate {}".format(x_gate.size()))
        # x = LlamaActCombine.apply(x_gate, x)
        x = self.act(x_gate) * x
        # print("after activate {}".format(x.size()))
        # x = [self.drop(x[i]) for i in range(e)]
        x = torch.matmul(x, self.wo.data) # num_expert * num_token * h
        # print("after entail {}".format(x.size()))


        # x = x.transpose(0, 1).contiguous() # num_expert * num_token * h
        x = MoeOutGradScaler.apply(x, self.ep_size)
        # print("after scale {}".format(x.size()))
        return x