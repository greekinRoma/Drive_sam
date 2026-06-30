from .encoders import OurEncoder
from torch import nn
from safetensors.torch import save_file
from safetensors import safe_open
import torch
import  numpy  as np
class LoRA_qkv(nn.Module):
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super(LoRA_qkv, self).__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.d_model = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)                     # 原始投影输出
        q_ba = self.linear_b_q(self.linear_a_q(x))   # LoRA 对 query 的增量
        v_ba = self.linear_b_v(self.linear_a_v(x))   # LoRA 对 value 的增量

        if qkv.dim() == 4:
            # 四维形状: (batch, heads, seq_len, head_dim * 3) 或类似
            qkv[:, :, :, :self.d_model] += q_ba
            qkv[:, :, :, -self.d_model:] += v_ba
        elif qkv.dim() == 3:
            # 三维形状: (batch, seq_len, d_model * 3)
            qkv[:, :, :self.d_model] += q_ba
            qkv[:, :, -self.d_model:] += v_ba
        else:
            raise ValueError(f"Unsupported qkv dimension: {qkv.dim()}. Expected 3 or 4.")

        return qkv
    
class LoRA_encoder(nn.Module):
    def __init__(self, encoder_type, checkpoint_path, rank, *args, **kwargs):
        super(LoRA_encoder, self).__init__(*args, **kwargs)
        self.rank = rank
        assert rank > 0
        self.encoder = build_encoder(encoder_type=encoder_type, check_point_path=checkpoint_path)
        self.lora_layer = list(range(len(self.encoder.blocks)))

        # --- 修改点 1: 使用 nn.ModuleList ---
        self.A_weights = nn.ModuleList()
        self.B_weights = nn.ModuleList()

        for param in self.encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(self.encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            self.d_model = w_qkv_linear.in_features

            w_a_linear_q = nn.Linear(self.d_model, self.rank, bias=False)
            w_b_linear_q = nn.Linear(self.rank, self.d_model, bias=False)
            w_a_linear_v = nn.Linear(self.d_model, self.rank, bias=False)
            w_b_linear_v = nn.Linear(self.rank, self.d_model, bias=False)

            # 使用 append 添加到 ModuleList
            self.A_weights.append(w_a_linear_q)
            self.B_weights.append(w_b_linear_q)
            self.A_weights.append(w_a_linear_v)
            self.B_weights.append(w_b_linear_v)

            blk.attn.qkv = LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v
            )
        self.reset_parameters()

    @property
    def device(self):
        # 通过查看模型参数的 device 来确定模型所在的设备
        return next(self.parameters())
    
    def reset_parameters(self):
        """
        Initialize the LoRA A and B matrices like in the paper
        """
        # Initalisation like in the paper
        for w_A in self.A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights:
            nn.init.zeros_(w_B.weight)

        
    def save_lora_parameters(self, filename: str):
        """
        Save the LoRA wieghts applied to the attention model as safetensors.

        Arguments:
            filenmame: Name of the file that will be saved
        
        Return:
            None: Saves a safetensors file
        """
        num_layer = len(self.A_weights)
        # sufix 03:d -> allows to have a name 1 instead of 001
        a_tensors = {f"w_a_{i:03d}": self.A_weights[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.B_weights[i].weight for i in range(num_layer)}
        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)


    def load_lora_parameters(self, filename: str):
        """
        Load a safetensor file of LoRA weights for the attention modules

        Arguments:
            filename: Name of the file containing the saved weights
        
        Return:
            None: Loads the weights to the LoRA_sam class
        """
        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.A_weights):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = nn.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.B_weights):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = nn.Parameter(saved_tensor)
                
    def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
        for n, p in model.named_parameters():
            # print(n)
            if 'linear_' not in n:
                p.requires_grad = False
        if bias == 'none':
            return
        elif bias == 'all':
            for n, p in model.named_parameters():
                if 'bias' in n:
                    p.requires_grad = True
        elif bias == 'lora_only':
            for m in model.modules():
                if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                        m.bias.requires_grad = True
        else:
            raise NotImplementedError
    
class LORA(nn.Module):
    """
    Class that takes the image encoder of SAM and add the lora weights to the attentions blocks

    Arguments:
        sam_model: Sam class of the segment anything model
        rank: Rank of the matrix for LoRA
        lora_layer: List of weights exisitng for LoRA
    
    Return:
        None

    """

    def __init__(self, model, rank, lora_layer=None):
        super(LORA, self).__init__()
        self.model = model
        self.rank = rank
        assert rank > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            # In each block, you have an attention block => total blocks -> nb lora layers
            self.lora_layer = list(range(len(model.blocks)))
        
        self.A_weights = []
        self.B_weights = []

        # freeze parameters of the image encoder
        for param in model.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(model.blocks):
            # if only lora on few layers
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            self.d_model = w_qkv_linear.in_features

            w_a_linear_q = nn.Linear(self.d_model, self.rank, bias=False)
            w_b_linear_q = nn.Linear(self.rank, self.d_model, bias=False)
            w_a_linear_v = nn.Linear(self.d_model, self.rank, bias=False)
            w_b_linear_v = nn.Linear(self.rank, self.d_model, bias=False)
            

            self.A_weights.append(w_a_linear_q)
            self.B_weights.append(w_b_linear_q)
            self.A_weights.append(w_a_linear_v)
            self.B_weights.append(w_b_linear_v)

            blk.attn.qkv = LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v
            )

        self.reset_parameters()
        self.sam = model
        self.lora_vit = model


    def reset_parameters(self):
        """
        Initialize the LoRA A and B matrices like in the paper
        """
        # Initalisation like in the paper
        for w_A in self.A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights:
            nn.init.zeros_(w_B.weight)


    def save_lora_parameters(self, filename: str):
        """
        Save the LoRA wieghts applied to the attention model as safetensors.

        Arguments:
            filenmame: Name of the file that will be saved
        
        Return:
            None: Saves a safetensors file
        """
        num_layer = len(self.A_weights)
        # sufix 03:d -> allows to have a name 1 instead of 001
        a_tensors = {f"w_a_{i:03d}": self.A_weights[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.B_weights[i].weight for i in range(num_layer)}
        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)


    def load_lora_parameters(self, filename: str):
        """
        Load a safetensor file of LoRA weights for the attention modules

        Arguments:
            filename: Name of the file containing the saved weights
        
        Return:
            None: Loads the weights to the LoRA_sam class
        """
        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.A_weights):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = nn.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.B_weights):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = nn.Parameter(saved_tensor)
    def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
        for n, p in model.named_parameters():
            # print(n)
            if 'linear_' not in n:
                p.requires_grad = False
        if bias == 'none':
            return
        elif bias == 'all':
            for n, p in model.named_parameters():
                if 'bias' in n:
                    p.requires_grad = True
        elif bias == 'lora_only':
            for m in model.modules():
                if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                        m.bias.requires_grad = True
        else:
            raise 
    def forward(self,input):
        return self.model(input)
    