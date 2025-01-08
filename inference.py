from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer


class LLaMA:
    """
    LLaMA 模型类。

    该类封装了 Transformer 模型和 SentencePiece 分词器，用于构建和加载 LLaMA 模型。
    提供了一个静态方法 `build` 用于从检查点目录和分词器路径构建 LLaMA 模型。
    """

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        """
        初始化 LLaMA 模型。

        参数:
            model (Transformer): 预训练的 Transformer 模型。
            tokenizer (SentencePieceProcessor): 分词器，用于处理文本数据。
            model_args (ModelArgs): 模型参数配置，包含模型的各种配置参数。
        """
        # Transformer 模型
        self.model = model
        # 分词器
        self.tokenizer = tokenizer
        # 模型参数配置
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        """
        构建 LLaMA 模型。

        该静态方法从指定的检查点目录和分词器路径加载模型参数和分词器，并构建 LLaMA 模型。

        参数:
            checkpoints_dir (str): 检查点目录路径，包含模型的检查点文件。
            tokenizer_path (str): 分词器路径，包含分词器的模型文件。
            load_model (bool): 是否加载预训练的模型参数。
            max_seq_len (int): 模型支持的最大序列长度。
            max_batch_size (int): 模型支持的最大批次大小。
            device (str): 设备类型，如 'cuda' 或 'cpu'。

        返回:
            LLaMA: 构建好的 LLaMA 模型实例。
        """
        # 记录当前时间，用于计时
        prev_time = time.time()
        if load_model:
            # 从checkpoints目录中获取所有 .pth 文件，并按名称排序
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            # 选择第一个checkpoints文件
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            # 更新计时起点
            prev_time = time.time()

        # 从checkpoints目录中读取 params.json 文件，获取模型参数
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        # 初始化模型参数配置，使用 max_seq_len, max_batch_size, device 和 params 中的其他参数
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        # 初始化 SentencePiece 分词器并加载分词器模型
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        # 设置词汇表大小为分词器的词汇表大小
        model_args.vocab_size = tokenizer.vocab_size()
        
        # 根据设备类型设置默认的张量类型
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        
        # 构建 Transformer 模型并将其移动到指定设备
        model = Transformer(model_args).to(device)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            # 删除checkpoint中不匹配的关键字（如果有的话）
            del checkpoint['rope.freqs']
            # 加载模型状态字典，严格模式确保所有参数都匹配
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
        
        # 返回构建好的 LLaMA 模型实例
        return LLaMA(model, tokenizer, model_args)

    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        """
        执行文本补全任务。

        该方法根据给定的提示生成文本，支持温度采样和 Top-p 采样。

        参数:
            prompts (List[str]): 输入的提示列表。
            temperature (float, 可选): 采样温度，用于控制生成文本的随机性，默认为 0.6。
            top_p (float, 可选): Top-p 采样参数，用于控制生成文本的多样性，默认为 0.9。
            max_gen_len (Optional[int], 可选): 最大生成长度，可选，默认为 None。

        返回:
            Tuple[List[List[int]], List[str]]: 生成的 token 列表和对应的文本列表。
        """
        if max_gen_len is None:
            # 如果未指定最大生成长度，则设置为最大序列长度减 1
            max_gen_len = self.args.max_seq_len - 1

        # Convert each prompt into tokens
        # 将每个提示转换为 token 列表
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure the batch size is not too large
        # 确保批次大小不超过最大批次大小
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        # 获取提示的最大长度
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)

        # Make sure the prompt length is not larger than the maximum sequence length
        # 确保提示长度不超过最大序列长度
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        # 计算总长度
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        # 创建用于存储生成 token 的张量，并初始化为填充 ID
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            # 用提示的 token 填充初始部分
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        
        # 初始化 EOS 标记列表
        eos_reached = torch.tensor([False] * batch_size, device=device)
        # 标记哪些位置是提示 token
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                # 获取当前位置的 logits
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                # The temperature is applied before the softmax
                # 应用温度参数进行采样
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                # 使用 Top-p 采样选择下一个 token
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the max probability
                # 贪婪选择下一个 token
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token
            # 仅在当前位置是填充位置时替换 token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            # 如果当前 token 是 EOS，并且位置不是提示部分，则标记为 EOS 已到达
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                # 如果所有样本都到达 EOS，则停止生成
                break
        
        # 初始化输出 token 列表
        out_tokens = []
        # 初始化输出文本列表
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            # 如果存在 EOS token，则截断到 EOS token
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            # 解码 token 并添加到输出文本列表
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        # 返回生成的 token 和文本
        return (out_tokens, out_text)
    
    def _sample_top_p(self, probs, p):
        """
        执行 Top-p 采样。

        该方法根据给定的概率分布和 Top-p 参数，选择下一个 token。

        参数:
            probs (torch.Tensor): 概率分布张量，形状为 (B, vocab_size)。
            p (float): Top-p 参数，用于控制生成文本的多样性。

        返回:
            torch.Tensor: 选择的下一个 token 的索引，形状为 (B, 1)。
        """
        # 对概率进行排序，并获取排序后的索引
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # 计算累积概率
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)

        # 创建掩码，标记哪些 token 属于 Top-p 集合
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p 

        # 将不在 Top-p 集合中的 token 的概率设置为 0
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0 

        # 重新归一化概率，使其和为 1
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

        # 从 Top-p 分布中采样下一个 token 的索引
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)

        # 根据索引获取 token 在词汇表中的位置
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token) 
        return next_token


if __name__ == '__main__':

    # 设置随机种子，以确保结果可复现
    torch.manual_seed(0)

    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    # 定义一组提示，用于生成文本
    prompts = []

    # 构建 LLaMA 模型实例
    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True, 
        max_seq_len=1024, # 模型支持的最大序列长度
        max_batch_size=len(prompts), # 批次大小，这里设置为提示的数量
        device=device
    )

    # 使用文本补全方法生成文本
    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)

    # 遍历生成的文本并输出
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)
