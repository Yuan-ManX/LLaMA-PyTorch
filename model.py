from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    """
    模型参数配置类。

    该类定义了模型的各种配置参数，用于控制模型的结构和训练过程。
    """
    dim: int = 4096  # 模型的维度，默认为 4096
    n_layers: int = 32  # 模型的层数，默认为 32
    n_heads: int = 32  # 每个注意力头的数量，默认为 32
    n_kv_heads: Optional[int] = None  # 键值（Key-Value）注意力头的数量，可选，默认为 None
    vocab_size: int = -1  # 词汇表大小，默认为 -1，后续在构建方法中设置
    multiple_of: int = 256  # 模型维度需要是此参数的倍数，默认为 256
    ffn_dim_multiplier: Optional[float] = None  # 前馈神经网络（FFN）维度乘数，可选，默认为 None
    norm_eps: float = 1e-5 # 归一化中的小常数 epsilon，默认为 1e-5

    # 以下参数用于键值缓存（KV Cache）
    max_batch_size: int = 32 # 最大批次大小，默认为 32
    max_seq_len: int = 2048 # 最大序列长度，默认为 2048

    device: str = None # 设备类型，可选，默认为 None


class RMSNorm(nn.Module):
    """
    均方根归一化（RMSNorm）模块。

    RMSNorm 是一种归一化方法，类似于 LayerNorm，但只对输入张量的均方根进行归一化，不使用偏置（bias）。
    这种方法在某些情况下可以提高训练稳定性和模型性能。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初始化 RMSNorm 模块。

        参数:
            dim (int): 输入张量的维度。
            eps (float, 可选): 小常数 epsilon，用于数值稳定性，默认为 1e-6。
        """
        super().__init__()
        # 设置 epsilon
        self.eps = eps
        # 初始化 gamma 参数，作为可学习的权重
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        """
        执行 RMSNorm 的核心归一化操作。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, Seq_Len, Dim)。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        # 计算输入张量的均方根 (RMS)
        # x.pow(2).mean(-1, keepdim=True) 计算每个样本的均方值
        # torch.rsqrt 计算均方根的倒数 (1 / sqrt(x))
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        # 对输入张量进行归一化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) 

    def forward(self, x: torch.Tensor):
        """
        前向传播方法，执行 RMSNorm。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        # 将输入张量转换为浮点类型
        # 执行归一化操作
        # 将归一化后的张量乘以 gamma 参数
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x) # 返回最终的归一化结果，并保持输入张量的数据类型


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    """
    预计算旋转位置嵌入的频率参数。

    该函数根据给定的头维度、序列长度和设备，预计算旋转位置嵌入所需的频率参数。
    这些频率参数用于在注意力机制中引入位置信息。

    参数:
        head_dim (int): 每个注意力头的维度。
        seq_len (int): 序列长度。
        device (str): 设备类型（如 'cuda' 或 'cpu'）。
        theta (float, 可选): 控制频率衰减的基值，默认为 10000.0。

    返回:
        torch.Tensor: 预计算好的频率复数张量，形状为 (Seq_Len, Head_Dim / 2)。
    """
    # 根据论文第3.2.2节的描述，维度必须是偶数
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    # 构建 theta 参数
    # 根据公式 theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # 形状: (Head_Dim / 2)
    # 生成偶数索引 [0, 2, 4, ..., head_dim-2]
    theta_numerator = torch.arange(0, head_dim, 2).float()

    # 计算 theta_i，形状: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)

    # 构建位置参数（"m" 参数）
    # 形状: (Seq_Len)
    # 生成位置索引 [0, 1, 2, ..., seq_len-1]
    m = torch.arange(seq_len, device=device)
    
    # 使用外积计算频率
    # (Seq_Len) 外积 (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()

    # 计算复数形式的频率
    # 我们可以使用极坐标形式 c = R * exp(m * theta)，其中 R = 1，如下所示:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    # 将频率转换为复数形式
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    # 返回复数形式的频率张量
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    """
    应用旋转位置嵌入。

    该函数将预计算的频率参数应用于输入张量，以引入位置信息。
    通过将输入张量的实部和虚部分离，并进行复数乘法，实现旋转。

    参数:
        x (torch.Tensor): 输入张量，形状为 (B, Seq_Len, H, Head_Dim)。
        freqs_complex (torch.Tensor): 预计算的频率复数张量，形状为 (Seq_Len, Head_Dim / 2)。
        device (str): 设备类型。

    返回:
        torch.Tensor: 应用旋转后的张量，形状与输入相同。
    """
    # 将输入张量的最后一个维度分割为两个部分，分别代表复数的实部和虚部
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # 将实部和虚部视为复数

    # 重塑 freqs_complex 张量以匹配 x_complex 的形状。需要添加批次维度和头维度
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2) # 重塑频率张量

    # 将每个复数与对应的频率复数相乘，实现旋转
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex # 进行复数乘法，实现旋转

    # 将复数转换回实数形式
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated) 

    # 重塑回原始形状
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)

    # 返回旋转后的张量，并保持输入张量的数据类型
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复键值（KV）张量。

    该函数将键值张量在键值头维度上进行重复，以增加键值头的数量。
    这在某些模型架构中用于扩展键值头的数量。

    参数:
        x (torch.Tensor): 输入键值张量，形状为 (B, Seq_Len, N_KV_Heads, Head_Dim)。
        n_rep (int): 重复次数。

    返回:
        torch.Tensor: 重复后的键值张量，形状为 (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)。
    """
    # 获取输入张量的形状
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        # 如果重复次数为1，则直接返回原张量
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]  # 在键值头维度上增加一个维度
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)  # 扩展张量以进行重复
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)  # 重塑张量形状
    ) # 返回重复后的键值张量


class SelfAttention(nn.Module):
    """
    自注意力机制模块。

    该模块实现了自注意力机制，是 Transformer 模型中的核心组件之一。
    支持键值（Key-Value）头的重复，以适应不同的多头设置。
    """
    def __init__(self, args: ModelArgs):
        """
        初始化自注意力模块。

        参数:
            args (ModelArgs): 模型参数配置，包含模型的各种配置参数。
        """
        super().__init__()

        # 键值（Key-Value）头的数量。如果未指定，则默认为查询（Query）头的数量
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        # 查询（Query）头的数量
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        # 键值头的重复次数，等于查询头的数量除以键值头的数量
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # 每个头的维度，即每个头负责的嵌入部分
        self.head_dim = args.dim // args.n_heads

        # 定义线性变换层，将输入嵌入向量投影到查询、键和值空间
        # 查询线性变换
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        # 键线性变换
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 值线性变换
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 输出线性变换
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 初始化键和值的缓存，用于存储历史键和值，以支持长序列处理
        # 键缓存
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        # 值缓存
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor
    ):
        """
        前向传播方法，执行自注意力机制。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, 1, Dim)。
            start_pos (int): 当前序列的起始位置。
            freqs_complex (torch.Tensor): 预计算的旋转位置嵌入频率复数张量。

        返回:
            torch.Tensor: 自注意力机制的输出张量，形状为 (B, 1, Dim)。
        """
        # 获取批次大小 (B)、序列长度 (1) 和维度 (Dim)
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)

        # 将输入张量投影到查询、键和值空间
        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xv = self.wv(x)

        # 重塑张量形状以适应多头注意力
        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 应用旋转位置嵌入
        # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # (B, 1, H_KV, Head_Dim) --> (B, 1, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # 将当前键和值替换到缓存中
        # 更新键缓存
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        # 更新值缓存
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # 获取缓存中的键和值
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.
        # 由于每个查询组共享相同的键和值头，因此需要为每个查询重复键和值头。
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # 重塑张量形状以进行矩阵乘法
        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # 计算注意力分数
        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 对注意力分数进行 softmax 归一化
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # 计算加权的值
        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # 重塑输出张量形状
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        # 应用输出线性变换，得到最终输出
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)


class FeedForward(nn.Module):
    """
    前馈神经网络模块。

    该模块实现了前馈神经网络，通常作为 Transformer 模型中的子层，用于处理和转换输入特征。
    采用 Swish 激活函数（也称为 SiLU 激活函数），并包含两个线性变换层和一个额外的线性层用于门控机制。
    """
    def __init__(
        self,
        args: ModelArgs
    ):
        """
        初始化前馈神经网络模块。

        参数:
            args (ModelArgs): 模型参数配置，包含模型的各种配置参数。
        """
        super().__init__()

        # 计算隐藏层的维度，初始为模型维度的 4 倍
        hidden_dim = 4 * args.dim
        # 将隐藏层维度调整为模型维度的 8/3 倍
        hidden_dim = int(2 * hidden_dim / 3)
        # 如果指定了 ffn_dim_multiplier，则进一步调整隐藏层维度
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        # 将隐藏层维度向上取整到 multiple_of 的倍数
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # 定义第一个线性变换层，将输入维度映射到隐藏层维度
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        # 定义第二个线性变换层，将隐藏层维度映射回输入维度
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        # 定义第三个线性变换层，用于门控机制，输入和输出维度相同
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        """
        前向传播方法，执行前馈神经网络的计算。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, Seq_Len, Dim)。

        返回:
            torch.Tensor: 前馈神经网络处理后的输出张量，形状为 (B, Seq_Len, Dim)。
        """
        # 应用第一个线性变换层，将输入张量映射到隐藏层维度
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))

        # 应用第三个线性变换层，用于门控机制
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)

        # 将 Swish 激活后的张量与门控张量相乘，实现门控机制
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V

        # 应用第二个线性变换层，将隐藏层维度映射回输入维度
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        # 返回前馈神经网络处理后的输出张量
        return x


class EncoderBlock(nn.Module):
    """
    Transformer 编码器块。

    该类实现了 Transformer 编码器中的一个块，包括自注意力机制、前馈神经网络以及层归一化。
    每个编码器块执行以下操作：
        1. 对输入进行层归一化。
        2. 应用自注意力机制。
        3. 将注意力输出与原始输入进行残差连接。
        4. 对结果进行层归一化。
        5. 应用前馈神经网络。
        6. 将前馈输出与步骤 3 的结果进行残差连接。
    """

    def __init__(self, args: ModelArgs):
        """
        初始化 Transformer 编码器块。

        参数:
            args (ModelArgs): 模型参数配置，包含模型的各种配置参数。
        """
        super().__init__()

        # 注意力头的数量
        self.n_heads = args.n_heads
        # 模型的维度
        self.dim = args.dim
        # 每个头的维度
        self.head_dim = args.dim // args.n_heads

        # 初始化自注意力机制模块
        self.attention = SelfAttention(args)
        # 初始化前馈神经网络模块
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the attention block
        # 在注意力块之前的层归一化
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward block
        # 在前馈神经网络块之前的层归一化
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        """
        前向传播方法，执行编码器块的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, Seq_Len, Dim)。
            start_pos (int): 当前序列的起始位置。
            freqs_complex (torch.Tensor): 预计算的旋转位置嵌入频率复数张量。

        返回:
            torch.Tensor: 编码器块处理后的输出张量，形状为 (B, Seq_Len, Dim)。
        """
        # 应用层归一化和自注意力机制
        # self.attention.forward(norm_x, start_pos, freqs_complex) -> (B, Seq_Len, Dim)
        # 将注意力输出与原始输入进行残差连接
        # 对残差连接后的结果进行层归一化
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )

        # 应用前馈神经网络
        # self.feed_forward.forward(norm_h) -> (B, Seq_Len, Dim)
        # 将前馈输出与步骤 3 的结果进行残差连接
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        # 返回编码器块的输出
        return out


class Transformer(nn.Module):
    """
    Transformer 模型类。

    该类实现了 Transformer 模型，包括词嵌入、多个编码器层、层归一化以及输出线性变换。
    每个输入的 token 都会被依次通过所有编码器层进行处理，最后通过输出线性变换得到预测结果。
    """

    def __init__(self, args: ModelArgs):
        """
        初始化 Transformer 模型。

        参数:
            args (ModelArgs): 模型参数配置，包含模型的各种配置参数。
        """
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        # 保存模型参数配置
        self.args = args
        # 词汇表大小
        self.vocab_size = args.vocab_size
        # 编码器层的数量
        self.n_layers = args.n_layers
        # 定义词嵌入层，将词汇表中的 token 转换为维度为 args.dim 的嵌入向量
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        # 初始化编码器层列表
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            # 添加一个编码器块到列表中
            self.layers.append(EncoderBlock(args))

        # 定义层归一化层，用于对编码器层的输出进行归一化
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 定义输出线性变换层，将归一化后的嵌入向量转换为词汇表大小的 logits
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # 预计算旋转位置嵌入的频率参数，用于在注意力机制中引入位置信息
        # 参数说明：
        #   head_dim = args.dim // args.n_heads：每个注意力头的维度
        #   seq_len = args.max_seq_len * 2：预计算的位置范围，假设为最大序列长度的两倍
        #   device = self.args.device：设备类型（如 'cuda' 或 'cpu'）
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        前向传播方法，执行 Transformer 模型的前向传播。

        参数:
            tokens (torch.Tensor): 输入的 token 张量，形状为 (B, Seq_Len)。
            start_pos (int): 当前序列的起始位置，用于检索对应的旋转位置嵌入频率。

        返回:
            torch.Tensor: Transformer 模型的输出 logits，形状为 (B, Seq_Len, Vocab_Size)。
        """
        # 获取批次大小 (B) 和序列长度 (Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        # 将输入的 token 转换为嵌入向量，形状为 (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        # 检索对应于位置 [start_pos, start_pos + seq_len] 的 (m, theta) 对
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        # Consecutively apply all the encoder layers
        # 依次应用所有编码器层
        for layer in self.layers:
            # 将当前层的输出作为下一层的输入
            h = layer(h, start_pos, freqs_complex)

        # 对编码器层的输出进行层归一化
        h = self.norm(h)
        # 通过输出线性变换层将归一化后的嵌入向量转换为词汇表大小的 logits
        output = self.output(h).float()
        # 返回输出 logits
        return output
