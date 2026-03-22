"""
Co-Training module for Qwen3 model.
Based on co_train_qwen.py with modifications for Qwen3 architecture.

Key differences from Qwen2:
- Qwen3 has q_norm and k_norm (RMSNorm applied to Q and K after projection)
- head_dim is configurable (default 128)
"""
import torch
import wandb
import os
import transformers

from torch import nn
from torch.nn.functional import linear, embedding
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config, Qwen3Model, Qwen3ForCausalLM, Qwen3DecoderLayer,
    Qwen3MLP, Qwen3Attention, Qwen3RMSNorm, Qwen3RotaryEmbedding,
    apply_rotary_pos_emb, repeat_kv
)
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.cache_utils import Cache, DynamicCache
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from tools.log import main_logger
from dataclasses import dataclass
from tools.global_state import hyper_params, data_cls_reversed_dict, ban_losses, ban_layers
from accelerate import Accelerator
from typing import Optional, Tuple, List
from functools import partial


accelerator = Accelerator()


class BigValueFirstLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mseloss = nn.MSELoss(reduction="none")

    def forward(self, output, target):
        return torch.mean(torch.abs(target + 1e-2) * self.mseloss(output, target))


class MSELossV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mseloss = nn.MSELoss(reduction="none")
    
    def forward(self, output, target):
        return self.mseloss(output, target).sum(dim=-1).mean()


class L1LossV2(nn.Module):
    def forward(self, output, target):
        return ((output - target).abs().sum(dim=-1)/50.0).mean()


LOSS_DICT = {
    "mseloss": nn.MSELoss,
    "mseloss_v2": MSELossV2,
    "l1loss": nn.L1Loss,
    "l1loss_v2": L1LossV2,
}


class CustomConfig(Qwen3Config):
    def set_custom_kwargs(self, **kwargs):
        # required
        self.target_hidden_size = kwargs["target_hidden_size"]
        self.use_attn_map = kwargs.get("use_attn_map", False)
        self.target_rms_norm_eps = kwargs.get("target_rms_norm_eps", self.rms_norm_eps)
        self.use_aux_loss = kwargs.get("use_aux_loss", True)
        self.use_std_like_attn = kwargs.get("use_std_like_attn", False)
        self.use_logits_loss = kwargs.get("use_logits_loss", True)
        self.use_ntp_loss = kwargs.get("use_ntp_loss", True)
        self.check_data_cls_loss = kwargs.get("check_data_cls_loss", False)
        self.kl_temperature = kwargs.get("kl_temperature", 10.0)
        self.aux_loss_type = kwargs.get("aux_loss_type", "mseloss")
        self.student_attn_from_scratch = kwargs.get("student_attn_from_scratch", False)
        self.tie_word_emb_proj = kwargs.get("tie_word_emb_proj", False)
        self.del_layers = kwargs.get("del_layers", [])
        self.use_in_out_mlp = kwargs.get("use_in_out_mlp", False)
        self.use_all_attn = kwargs.get("use_all_attn", False)


class DebugRMSNorm(nn.Module):
    """RMSNorm for student model"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def reinit_weight(module: nn.Module):
    if type(module) == nn.Linear:
        if module.weight.requires_grad:
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    if type(module) == DebugRMSNorm:
        if module.weight.requires_grad:
            module.weight.data.fill_(1.0)


class Attn(nn.Module):
    """
    Custom Attention module for Qwen3 co-training.
    Qwen3 specific: includes q_norm and k_norm (RMSNorm on Q and K)
    """
    def __init__(self, config: CustomConfig, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        
        # Original projections (teacher)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        
        # Qwen3 specific: Q and K normalization
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        # Rotary embedding
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        
        # Co-training zoom projections
        self.zoom_up = nn.Linear(config.target_hidden_size, self.hidden_size, bias=False)
        self.zoom_down = nn.Linear(self.hidden_size, config.target_hidden_size, bias=False)
        
        self.mseloss = LOSS_DICT[config.aux_loss_type]()

    def _compute_attention(self, query_states, key_states, value_states, attention_mask=None):
        """Compute attention output using eager attention"""
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout if self.training else 0.0, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        compressed_hidden_states: torch.Tensor,
        loss_dict: dict,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        hidden_shape = (bsz, q_len, -1, self.head_dim)
        
        # Teacher forward
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Compute teacher attention
        raw_out, raw_attn_weights = self._compute_attention(query_states, key_states, value_states, attention_mask)
        raw_out = raw_out.reshape(bsz, q_len, -1).contiguous()
        raw_out = self.o_proj(raw_out)
        
        # Student forward (with zoom projections)
        zoomed_hs = self.zoom_up(compressed_hidden_states)
        s_query_states = self.q_norm(self.q_proj(zoomed_hs).view(hidden_shape)).transpose(1, 2)
        s_key_states = self.k_norm(self.k_proj(zoomed_hs).view(hidden_shape)).transpose(1, 2)
        s_value_states = self.v_proj(zoomed_hs).view(hidden_shape).transpose(1, 2)
        
        # Apply rotary embeddings to student
        s_query_states, s_key_states = apply_rotary_pos_emb(s_query_states, s_key_states, cos, sin)
        
        # Compute student attention
        out, _ = self._compute_attention(s_query_states, s_key_states, s_value_states, attention_mask)
        out = out.reshape(bsz, q_len, -1).contiguous()
        out = self.o_proj(out)
        compressed_hidden_states = self.zoom_down(out)
        
        # Compute auxiliary losses
        if "attn-in-sim-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict["attn-in-sim-loss"] = self.mseloss(zoomed_hs, hidden_states)
        if "attn-out-sim-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict["attn-out-sim-loss"] = self.mseloss(out, raw_out)
            
        return raw_out, compressed_hidden_states, raw_attn_weights, None, loss_dict
    
    def merge_weight(self):
        """Merge zoom projections into weights for inference"""
        self.q_proj.weight.data = (self.q_proj.weight.data @ self.zoom_up.weight.data).contiguous()
        self.k_proj.weight.data = (self.k_proj.weight.data @ self.zoom_up.weight.data).contiguous()
        self.v_proj.weight.data = (self.v_proj.weight.data @ self.zoom_up.weight.data).contiguous()
        self.o_proj.weight.data = (self.zoom_down.weight.data @ self.o_proj.weight.data).contiguous()


class AllAttn(Attn):
    """
    Alternative attention with separate zoom for Q, K, V.
    """
    def __init__(self, config: CustomConfig, layer_idx=None):
        super().__init__(config, layer_idx)
        # Override zoom projections with separate Q, K, V zooms
        del self.zoom_up
        self.zoom_q = nn.Linear(config.target_hidden_size, self.hidden_size, bias=False)
        self.zoom_k = nn.Linear(config.target_hidden_size, self.hidden_size, bias=False)
        self.zoom_v = nn.Linear(config.target_hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        compressed_hidden_states: torch.Tensor,
        loss_dict: dict,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        hidden_shape = (bsz, q_len, -1, self.head_dim)
        
        # Get position embeddings
        cos, sin = position_embeddings
        
        # Teacher forward
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        raw_out, raw_attn_weights = self._compute_attention(query_states, key_states, value_states, attention_mask)
        raw_out = raw_out.reshape(bsz, q_len, -1).contiguous()
        raw_out = self.o_proj(raw_out)
        
        # Student forward with separate zoom projections
        _up_hs_q = self.q_proj(self.zoom_q(compressed_hidden_states))
        _up_hs_k = self.k_proj(self.zoom_k(compressed_hidden_states))
        _up_hs_v = self.v_proj(self.zoom_v(compressed_hidden_states))
        
        s_query_states = self.q_norm(_up_hs_q.view(hidden_shape)).transpose(1, 2)
        s_key_states = self.k_norm(_up_hs_k.view(hidden_shape)).transpose(1, 2)
        s_value_states = _up_hs_v.view(hidden_shape).transpose(1, 2)
        s_query_states, s_key_states = apply_rotary_pos_emb(s_query_states, s_key_states, cos, sin)
        
        out, _ = self._compute_attention(s_query_states, s_key_states, s_value_states, attention_mask)
        out = out.reshape(bsz, q_len, -1).contiguous()
        out = self.o_proj(out)
        compressed_hidden_states = self.zoom_down(out)
        
        # Compute auxiliary losses
        teacher_q = self.q_proj(hidden_states)
        teacher_k = self.k_proj(hidden_states)
        teacher_v = self.v_proj(hidden_states)
        
        if "attn-q-sim-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict["attn-q-sim-loss"] = self.mseloss(_up_hs_q, teacher_q)
        if "attn-k-sim-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict["attn-k-sim-loss"] = self.mseloss(_up_hs_k, teacher_k)
        if "attn-v-sim-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict["attn-v-sim-loss"] = self.mseloss(_up_hs_v, teacher_v)
        if "attn-out-sim-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict["attn-out-sim-loss"] = self.mseloss(out, raw_out)
            
        return raw_out, compressed_hidden_states, raw_attn_weights, None, loss_dict
    
    def merge_weight(self):
        self.q_proj.weight.data = (self.q_proj.weight.data @ self.zoom_q.weight.data).contiguous()
        self.k_proj.weight.data = (self.k_proj.weight.data @ self.zoom_k.weight.data).contiguous()
        self.v_proj.weight.data = (self.v_proj.weight.data @ self.zoom_v.weight.data).contiguous()
        self.o_proj.weight.data = (self.zoom_down.weight.data @ self.o_proj.weight.data).contiguous()


class MLP(Qwen3MLP):
    def __init__(self, config: CustomConfig, layer_idx=None):
        super().__init__(config)
        self.zoom_up = nn.Linear(self.hidden_size, config.target_hidden_size, bias=False)
        self.zoom_gate = nn.Linear(self.hidden_size, config.target_hidden_size, bias=False)
        self.zoom_down = nn.Linear(self.hidden_size, config.target_hidden_size, bias=False)
        self.mseloss = LOSS_DICT[config.aux_loss_type]()
        self.layer_idx = layer_idx

    def small_forward(self, compressed_x, raw_gate, raw_act_gate, raw_up, raw_x, raw_out, loss_dict: dict):
        Wup = self.zoom_up(self.up_proj.weight)
        Wgate = self.zoom_gate(self.gate_proj.weight)
        Wdown = self.zoom_down(self.down_proj.weight.T).T
        gate = linear(compressed_x, Wgate)
        act_gate = self.act_fn(gate)
        up = linear(compressed_x, Wup)
        down = linear(act_gate * up, Wdown)

        # Calculate losses
        if "mlp-gate-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict[f"mlp-gate-loss"] = self.mseloss(gate, raw_gate)
        if "mlp-up-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict[f"mlp-up-loss"] = self.mseloss(up, raw_up)
        if "mlp-out-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict[f"mlp-out-loss"] = self.mseloss(down, self.zoom_down(raw_out))

        return down

    def forward(self, x, compressed_x, loss_dict: dict):
        gate = self.gate_proj(x)
        act_gate = self.act_fn(gate)
        up = self.up_proj(x)
        down = self.down_proj(act_gate * up)

        return down, self.small_forward(compressed_x, gate, act_gate, up, x, down, loss_dict), loss_dict
    
    def merge_weight(self):
        self.gate_proj.weight.data = self.zoom_gate(self.gate_proj.weight.data).contiguous()
        self.up_proj.weight.data = self.zoom_up(self.up_proj.weight.data).contiguous()
        self.down_proj.weight.data = self.zoom_down(self.down_proj.weight.data.T).T.contiguous()


class CustomLayer(nn.Module):
    def __init__(self, config: CustomConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Create attention module
        if self.config.use_all_attn:
            self.self_attn = AllAttn(config, layer_idx)
        else:
            self.self_attn = Attn(config, layer_idx)
        
        # Create MLP module
        self.mlp = MLP(config, layer_idx)
        
        # Teacher layer norms
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Student layer norms
        self.target_input_layernorm = DebugRMSNorm(config.target_hidden_size, eps=config.target_rms_norm_eps)
        self.target_post_attention_layernorm = DebugRMSNorm(config.target_hidden_size, eps=config.target_rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        compressed_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        compressed_residual = compressed_hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        compressed_hidden_states = self.target_input_layernorm(compressed_hidden_states)

        # Self Attention
        hidden_states, compressed_hidden_states, self_attn_weights, present_key_value, loss_dict = self.self_attn(
            hidden_states=hidden_states,
            compressed_hidden_states=compressed_hidden_states,
            loss_dict={},
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = residual + hidden_states
        compressed_hidden_states = compressed_hidden_states + compressed_residual

        # Fully Connected (MLP)
        residual = hidden_states
        compressed_residual = compressed_hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        compressed_hidden_states = self.target_post_attention_layernorm(compressed_hidden_states)

        hidden_states, compressed_hidden_states, loss_dict = self.mlp(hidden_states, compressed_hidden_states, loss_dict)

        hidden_states = residual + hidden_states
        compressed_hidden_states = compressed_hidden_states + compressed_residual

        outputs = (hidden_states, compressed_hidden_states, loss_dict)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
    def merge_weight(self):
        self.input_layernorm.weight.data = self.target_input_layernorm.weight.data
        self.post_attention_layernorm.weight.data = self.target_post_attention_layernorm.weight.data


@dataclass
class IIModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    compressed_hidden_states: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class Model(Qwen3Model):
    _no_split_modules = ["CustomLayer"]

    def __init__(self, config: CustomConfig):
        super().__init__(config)

        self.zoom = nn.Linear(config.hidden_size, config.target_hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [CustomLayer(config, layer_idx) if layer_idx not in config.del_layers else Qwen3DecoderLayer(config, layer_idx)
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_norm = DebugRMSNorm(config.target_hidden_size, eps=config.target_rms_norm_eps)
        self.cur_step = 0

        self.post_init()

    def merge_weight(self):
        self.embed_tokens.weight.data = self.zoom(self.embed_tokens.weight.data).contiguous()
        self.norm.weight.data = self.target_norm.weight.data

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            raise NotImplementedError

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # For co-training, we don't use causal mask
        assert attention_mask is None

        # Embed positions
        hidden_states = inputs_embeds
        Wemb = self.zoom(self.embed_tokens.weight).to(device=input_ids.device)
        if os.environ.get("DEBUG", False):
            print("emb token", Wemb[0, :6])
        compressed_hidden_states = embedding(input_ids, Wemb)

        # Create position embeddings (Qwen3 style)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        aux_loss = 0

        # Set state for logging loss
        grad_acumulation_steps = hyper_params["gradient_accumulation_steps"]
        cur_train_step = None
        if (self.cur_step + 1) % (grad_acumulation_steps * 20) == 0:
            cur_train_step = (self.cur_step + 1) // grad_acumulation_steps
        self.cur_step += 1
        
        for layer_idx, decoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                raise NotImplementedError
            
            if layer_idx not in self.config.del_layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    compressed_hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

                compressed_hidden_states = layer_outputs[1]
                loss_dict = layer_outputs[2]
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            
            hidden_states = layer_outputs[0]

            if layer_idx not in self.config.del_layers:
                _log_dict = {}
                for k, v in loss_dict.items():
                    if self.config.use_aux_loss:
                        if isinstance(aux_loss, torch.Tensor):
                            aux_loss = aux_loss.to(v.device)
                        aux_loss = aux_loss + v * hyper_params["aux_loss_scale_factor"]
                    main_logger.debug(f"L{decoder_layer.layer_idx}-{k}: {v.item()}")
                    
                    if cur_train_step:
                        _log_dict[f"L{decoder_layer.layer_idx}-{k}"] = v.item()
                
                if cur_train_step and (os.environ.get("LOCAL_RANK", 0) == 0 or accelerator.is_main_process) and len(_log_dict) > 0:
                    wandb.log(_log_dict, cur_train_step)

        hidden_states = self.norm(hidden_states)
        compressed_hidden_states = self.target_norm(compressed_hidden_states)

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, all_self_attns] if v is not None)
        
        return IIModelOutput(
            last_hidden_state=hidden_states,
            compressed_hidden_states=compressed_hidden_states,
            aux_loss=aux_loss,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


def calculate_language_loss(lgts, labels, vocab_size):
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = lgts[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
    return loss


class CoTrainLM(Qwen3ForCausalLM):
    # Remove _tied_weights_keys to avoid conflict with newer transformers
    # We don't actually tie weights, we use zoom projections instead
    
    def __init__(self, config: CustomConfig):
        super().__init__(config)
        self.model = Model(config)
        if not config.tie_word_emb_proj:
            self.zoom_down = nn.Linear(config.hidden_size, config.target_hidden_size, bias=False)
            self.zoom_down.weight.data.normal_(mean=0.0, std=0.01)
        self.mseloss = LOSS_DICT[config.aux_loss_type]()
        self.kl_temperature = self.config.kl_temperature
        self.cur_step = 0
        self.cur_loss_accumulation = 0
        self.cur_logit_loss_accumulation = 0
        self.check_data_cls_loss = config.check_data_cls_loss
        self.data_cls_losses = [0] * 8
        self.data_cls_cnt = [0] * 8
        self.post_init()

    def merge_weight(self):
        if not self.config.tie_word_emb_proj:
            self.lm_head.weight.data = self.zoom_down(self.lm_head.weight.data).contiguous()
        else:
            self.lm_head.weight.data = self.model.zoom(self.lm_head.weight.data).contiguous()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        data_cls=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Decoder outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        compressed_hidden_states = outputs[1]
        aux_loss = outputs[2]

        logits = self.lm_head(hidden_states)
        if not self.config.tie_word_emb_proj:
            Whead = self.zoom_down(self.lm_head.weight)
        else:
            Whead = self.model.zoom(self.lm_head.weight)
        if os.environ.get("DEBUG", False):
            print("head weight", Whead[0, :6])
        target_logits = linear(compressed_hidden_states, Whead)

        if self.config.use_logits_loss:
            target_logp = F.log_softmax(target_logits / self.kl_temperature, dim=-1)
            raw_logp = F.log_softmax(logits / self.kl_temperature, dim=-1)
            logits_loss = F.kl_div(target_logp, raw_logp, log_target=True, reduction="batchmean")
            aux_loss = aux_loss + logits_loss
            main_logger.debug(f"logits_loss: {round(logits_loss.item(), 4)}")
        
        raw_loss = calculate_language_loss(logits.float(), labels, self.config.vocab_size)
        target_loss = calculate_language_loss(target_logits.float(), labels, self.config.vocab_size)
        main_logger.debug(f"raw_loss: {round(raw_loss.item(), 4)}, target_loss: {round(target_loss.item(), 4)}")

        # Wandb logging
        self.cur_loss_accumulation += target_loss.item()
        if self.config.use_logits_loss:
            self.cur_logit_loss_accumulation += logits_loss.item()
        loss_log_steps = hyper_params["gradient_accumulation_steps"] * 5
        if self.check_data_cls_loss:
            assert hidden_states.shape[0] == 1, "only appliable in bs = 1"
            spec_cls = data_cls[0].item()
            self.data_cls_cnt[spec_cls] += 1
            self.data_cls_losses[spec_cls] += target_loss.item()
        if (self.cur_step + 1) % loss_log_steps == 0:
            cur_train_step = (self.cur_step + 1) // hyper_params["gradient_accumulation_steps"]
            _log_dict = {"target_loss": self.cur_loss_accumulation / loss_log_steps}
            if self.config.use_logits_loss:
                _log_dict["logits_loss"] = self.cur_logit_loss_accumulation / loss_log_steps
            if self.check_data_cls_loss:
                _log_dict.update({
                    f"{data_cls_reversed_dict[i]}_loss": loss / self.data_cls_cnt[i] 
                    for i, loss in enumerate(self.data_cls_losses) if self.data_cls_cnt[i] > 0
                })
            if (os.environ.get("LOCAL_RANK", 0) == 0 or accelerator.is_main_process):
                wandb.log(_log_dict, step=cur_train_step)
            self.cur_loss_accumulation = 0
            self.cur_logit_loss_accumulation = 0
            self.data_cls_cnt = [0] * 8
            self.data_cls_losses = [0] * 8
        self.cur_step += 1

        if not return_dict:
            raise NotImplementedError

        return CausalLMOutputWithPast(
            loss=target_loss + aux_loss if self.config.use_ntp_loss else aux_loss,
            logits=target_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def freeze_original_model(self):
        key_words = ["zoom", "target"]
        for n, p in self.named_parameters():
            flag = False

            for key in key_words:
                if key in n:
                    flag = True

            p.requires_grad = flag

    def tie_custom_weights(self, tie_n):
        raise ValueError("low perf")

    def tie_word_emb_proj(self):
        self.zoom_down.weight = self.model.zoom.weight

    def get_trained_params(self):
        state_dict = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                state_dict[n] = p
        return state_dict

    def save_pretrained(self, *args, **kwargs):
        if kwargs.get("only_save_trainable", True):
            state_dict = self.get_trained_params()
            kwargs["state_dict"] = state_dict
        return super().save_pretrained(*args, **kwargs)
