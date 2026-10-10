import math

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import functional as F

"""
实现一个修改版本的GPT2，结合VAR的概念。
主要修改包括：
    1. attention mask的构造
    2. 层次化的位置编码
    3. 特殊token的处理（SOS, EOS, PAD）
    4. parallel sampling

SRC: [SOS, L1, L2, Ln-1, Ln (variable length)]
DST: [L1, L2, ..., Ln-1, Ln (variable length), EOS]

"""

class NewGELU(nn.Module):

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )

class CausalSelfAttention(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)

        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

    def forward(self, x, attn_mask=None):
        B, T, C = (
            x.size()
        )

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if attn_mask is not None:

            att = att.masked_fill(attn_mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )

        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                c_proj=nn.Linear(4 * cfg.n_embd, cfg.n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(cfg.resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x, attn_mask=None):

        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT_Normal(nn.Module):

    def __init__(self, cfg: DictConfig):

        super().__init__()
        assert cfg.vocab_size is not None
        self.cfg = cfg

        self.enable_lvl_emb = cfg.get("enable_lvl_emb", True)
        self.enable_var_mask = cfg.get("enable_var_mask", False)

        print("Enable Level Embeddings:", self.enable_lvl_emb)
        print("Enable Level Attention Mask:", self.enable_var_mask)

        self.pad_token_id = cfg.pad_token_id
        self.eos_token_id = cfg.eos_token_id

        self.patch_nums = cfg.patch_nums
        self.n_fixed_levels = len(self.patch_nums)

        self.n_levels = self.n_fixed_levels + 1

        self.level_lengths = [pn for pn in self.patch_nums] + [cfg.max_var_len]

        self.block_size = sum(self.level_lengths) + 1

        self.transformer = nn.ModuleDict(
            dict(

                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),

                lvl_emb=nn.Embedding(self.n_levels, cfg.n_embd),

                pos_emb=nn.Embedding(self.block_size, cfg.n_embd),

                drop=nn.Dropout(cfg.embd_pdrop),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embd),
            )
        )
        self.sos = nn.Parameter(torch.randn(1, 1, cfg.n_embd))

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer)
                )

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def create_masks(self, idx, lvl_ids):

        if not self.enable_var_mask:
            B, T = idx.size()

            causal_mask = torch.tril(torch.ones(T, T, device=idx.device)).view(
                1, 1, T, T
            )
            causal_mask = causal_mask.bool()

            pad_mask = (
                (idx != self.pad_token_id).unsqueeze(1).unsqueeze(2)
            )

            combined_mask = causal_mask & pad_mask

            return combined_mask
        else:
            B, T = idx.size()
            causal_mask = torch.tril(torch.ones(T, T, device=idx.device)).view(
                1, 1, T, T
            )
            causal_mask = causal_mask.bool()
            pad_mask = (idx != self.pad_token_id).unsqueeze(1).unsqueeze(2)
            lvl_ids_q = lvl_ids.unsqueeze(-1)
            lvl_ids_k = lvl_ids.unsqueeze(-2)

            var_mask = (lvl_ids_q > lvl_ids_k).unsqueeze(1)

            var_mask_range = sum(self.level_lengths[:-1])
            combined_mask = causal_mask.repeat(B, 1, 1, 1)
            combined_mask[:, :, :var_mask_range, :var_mask_range] = var_mask[
                :, :, :var_mask_range, :var_mask_range
            ]

            combined_mask = combined_mask & pad_mask
            return combined_mask

    def forward(self, idx, lvl_ids, targets=None):

        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        )

        tok_emb = self.transformer.wte(idx)

        tok_emb[:, 0, :] = self.sos.squeeze(0)

        lvl_emb = self.transformer.lvl_emb(lvl_ids)

        seq_indices = torch.arange(t, dtype=torch.long, device=device).unsqueeze(
            0
        )
        pos_emb = self.transformer.pos_emb(seq_indices)

        if self.enable_lvl_emb:
            x = tok_emb + pos_emb + lvl_emb
        else:
            x = tok_emb + pos_emb

        x = self.transformer.drop(x)

        attn_mask = self.create_masks(idx, lvl_ids)

        for block in self.transformer.h:
            x = block(x, attn_mask=attn_mask)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.pad_token_id,
            )

        return logits, loss

    def configure_optimizers(self, cfg_opt: DictConfig):

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("transformer.sos")
        no_decay.add("transformer.pos_emb")
        no_decay.add("transformer.lvl_emb.weight")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            f"parameters {inter_params} made it into both decay/no_decay sets!"
        )
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {param_dict.keys() - union_params} were not separated!"
        )

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": cfg_opt.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=cfg_opt.learning_rate, betas=cfg_opt.betas
        )
        return optimizer

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k=None,
    ):

        idx = torch.full(
            (batch_size, 1),
            fill_value=0,
            device=device,
            dtype=torch.long,
        )
        level_indices = (
            torch.repeat_interleave(
                torch.arange(len(self.level_lengths)),
                torch.tensor(self.level_lengths),
            )
            .unsqueeze(0)
            .to(device)
        )
        level_indices = torch.cat(
            [torch.tensor([[0]], device=device), level_indices], dim=1
        )
        len_to_add = max_new_tokens + 1 - level_indices.shape[1]
        level_indices = torch.cat(
            [
                level_indices,
                torch.full((1, len_to_add), level_indices[:, -1].item(), device=device),
            ],
            dim=1,
        )

        lvl_ids_full = level_indices.expand(batch_size, -1)

        generate_ids = self.generate(
            idx, lvl_ids_full, max_new_tokens, temperature, do_sample, top_k
        )

        real_token_ind = 1 + sum(self.level_lengths[:-1])
        real_tokens = generate_ids[:, real_token_ind:]

        list_tokens = []
        for i in range(real_tokens.size(0)):
            eos_positions = (real_tokens[i] == self.eos_token_id).nonzero(
                as_tuple=True
            )[0]
            if len(eos_positions) > 0:
                first_eos_pos = eos_positions[0].item()
                list_tokens.append(real_tokens[i, :first_eos_pos].cpu().numpy())
            else:
                list_tokens.append(real_tokens[i].cpu().numpy())

        return list_tokens

    @torch.no_grad()
    def generate(
        self,
        idx,
        lvl_ids_full,

        max_new_tokens,
        temperature=1.0,
        do_sample=False,
        top_k=None,
    ):

        b = idx.size(0)
        t_prompt = idx.size(1)
        device = idx.device

        assert lvl_ids_full.size(0) == b, "Batch size mismatch for lvl_ids_full"
        assert lvl_ids_full.size(1) == t_prompt + max_new_tokens, (
            f"lvl_ids_full has wrong length: {lvl_ids_full.size(1)}. "
            f"Expected T_prompt ({t_prompt}) + max_new_tokens ({max_new_tokens}) = {t_prompt + max_new_tokens}"
        )

        is_finished = torch.zeros(b, dtype=torch.bool, device=device)

        for i in range(max_new_tokens):

            current_len_in = idx.size(1)

            idx_cond = (
                idx if current_len_in <= self.block_size else idx[:, -self.block_size :]
            )

            lvl_ids_cond = (
                lvl_ids_full[:, :current_len_in]
                if current_len_in <= self.block_size
                else lvl_ids_full[
                    :, (current_len_in - self.block_size) : current_len_in
                ]
            )

            assert idx_cond.size() == lvl_ids_cond.size(), (
                f"Shape mismatch: idx_cond {idx_cond.size()} vs lvl_ids_cond {lvl_ids_cond.size()}"
            )

            logits, _ = self(idx_cond, lvl_ids_cond)

            logits_next = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits_next, top_k)
                logits_next[logits_next < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits_next, dim=-1)

            if do_sample:
                idx_sampled = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_sampled = torch.topk(probs, k=1, dim=-1)

            newly_finished = (~is_finished) & (
                idx_sampled.squeeze(1) == self.eos_token_id
            )

            idx_to_append = torch.where(
                is_finished.unsqueeze(1),
                torch.full_like(idx_sampled, self.pad_token_id),
                idx_sampled,
            )

            is_finished = is_finished | newly_finished

            idx = torch.cat((idx, idx_to_append), dim=1)

            if is_finished.all():
                break

        return idx

    @torch.no_grad()
    def sample_2(
        self,
        batch_size: int,
        device: torch.device,
        max_new_tokens: int,
        temp_struct: float = 1.0,
        temp_content: float = 1.0,
        do_sample: bool = False,
        top_k=None,
    ):

        idx = torch.full(
            (batch_size, 1),
            fill_value=0,
            device=device,
            dtype=torch.long,
        )

        level_indices = (
            torch.repeat_interleave(
                torch.arange(len(self.level_lengths)),
                torch.tensor(self.level_lengths),
            )
            .unsqueeze(0)
            .to(device)
        )

        level_indices = torch.cat(
            [torch.tensor([[0]], device=device), level_indices], dim=1
        )

        len_to_add = max_new_tokens + 1 - level_indices.shape[1]
        if len_to_add > 0:
            level_indices = torch.cat(
                [
                    level_indices,
                    torch.full(
                        (1, len_to_add), level_indices[:, -1].item(), device=device
                    ),
                ],
                dim=1,
            )

        lvl_ids_full = level_indices.expand(batch_size, -1)

        generate_ids = self.generate_dual_temp(
            idx,
            lvl_ids_full,
            max_new_tokens,
            temp_struct,
            temp_content,
            do_sample,
            top_k,
        )

        real_token_ind = 1 + sum(self.level_lengths[:-1])
        real_tokens = generate_ids[:, real_token_ind:]

        list_tokens = []
        for i in range(real_tokens.size(0)):
            eos_positions = (real_tokens[i] == self.eos_token_id).nonzero(
                as_tuple=True
            )[0]
            if len(eos_positions) > 0:
                first_eos_pos = eos_positions[0].item()
                list_tokens.append(real_tokens[i, :first_eos_pos].cpu().numpy())
            else:
                list_tokens.append(real_tokens[i].cpu().numpy())

        return list_tokens

    @torch.no_grad()
    def generate_dual_temp(
        self,
        idx,
        lvl_ids_full,
        max_new_tokens,
        temp_struct=1.0,
        temp_content=1.0,
        do_sample=False,
        top_k=None,
    ):

        b = idx.size(0)
        t_prompt = idx.size(1)
        device = idx.device

        var_level_id = self.n_levels - 1

        is_finished = torch.zeros(b, dtype=torch.bool, device=device)

        for i in range(max_new_tokens):

            current_len_in = idx.size(1)

            current_level = lvl_ids_full[0, current_len_in].item()

            if current_level == var_level_id:

                current_temp = temp_content
            else:

                current_temp = temp_struct

            idx_cond = (
                idx if current_len_in <= self.block_size else idx[:, -self.block_size :]
            )
            lvl_ids_cond = (
                lvl_ids_full[:, :current_len_in]
                if current_len_in <= self.block_size
                else lvl_ids_full[
                    :, (current_len_in - self.block_size) : current_len_in
                ]
            )

            logits, _ = self(idx_cond, lvl_ids_cond)
            logits_next = logits[:, -1, :]

            if current_temp == 0:

                current_temp = 1e-8

            logits_next = logits_next / current_temp

            if top_k is not None:
                v, _ = torch.topk(logits_next, top_k)
                logits_next[logits_next < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits_next, dim=-1)

            if do_sample:
                idx_sampled = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_sampled = torch.topk(probs, k=1, dim=-1)

            newly_finished = (~is_finished) & (
                idx_sampled.squeeze(1) == self.eos_token_id
            )
            idx_to_append = torch.where(
                is_finished.unsqueeze(1),
                torch.full_like(idx_sampled, self.pad_token_id),
                idx_sampled,
            )
            is_finished = is_finished | newly_finished
            idx = torch.cat((idx, idx_to_append), dim=1)

            if is_finished.all():
                break

        return idx
