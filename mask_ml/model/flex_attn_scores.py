import torch
from torch.nn.attention.flex_attention import _score_mod_signature, _mask_mod_signature
from torch import Tensor
from torch._inductor.lowering import make_pointwise, register_lowering

# Some internal torch.compile details
from torch._inductor.virtualized import ops
from functools import partial
from  typing import List, Union
from enum import Enum




def generate_alibi_bias(H: int) -> _score_mod_signature:
    """Returns an alibi bias score_mod given the number of heads H

    Args:
        H: number of heads

    Returns:
        alibi_bias: alibi bias score_mod
    """

    def alibi_mod(score, b, h, q_idx, kv_idx):
        scale = torch.exp2(-((h + 1) * 8.0 / H))
        bias = (kv_idx - q_idx) * scale
        return score + bias

    return alibi_mod


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx




def generate_dilated_sliding_window(window_size: int, dilation: int) -> _mask_mod_signature:
    """Generates a dilated sliding window attention mask.
    Args:
        window_size: The size of the sliding window.
        dilation: The dilation factor for the sliding window.

    Note:
        Query at position i can only attend to keys within a window of size `window_size`
        centered around i, where the keys are at positions j such that:
        * abs(i - j) <= window_size
        * abs(i - j) % dilation == 0
    """

    def dilated_sliding_window(b, h, q_idx, kv_idx):
        diff = torch.abs(q_idx - kv_idx)
        in_window = diff <= window_size
        is_dilated = (diff % dilation) == 0
        return in_window & is_dilated

    dilated_sliding_window.__name__ = f"dilated_sliding_window_{window_size}_dilation_{dilation}"
    return dilated_sliding_window


def _offsets_to_doc_ids_tensor(offsets):
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def length_to_offsets(lengths: List[int], device: Union[str, torch.device]) -> Tensor:
    """Converts a list of lengths to a list of offsets.

    Args:
        lengths: A list of lengths.

    """
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets


def generate_doc_mask_mod(mask_mod: _mask_mod_signature, offsets: Tensor) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        offsets: This tensor should be of shape(num_documents + 1)
            this should contain the cumulative counts of document tokens.
            e.g. if you have 3 documents of length 2, 4, 3 then
            offsets = [0, 2, 6, 9]

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.
    """
    document_id = _offsets_to_doc_ids_tensor(offsets)

    def doc_mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        q_logical = q_idx - offsets[document_id[q_idx]]
        kv_logical = kv_idx - offsets[document_id[kv_idx]]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask

    return doc_mask_mod


@torch.library.custom_op("approx::tanh", mutates_args=())
def _tanh_approx(inp: Tensor) -> Tensor:
    return torch.tanh(inp)


@_tanh_approx.register_fake
def _(inp: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inp)


def _tanh_approx_lowering(inp):
    fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
    return make_pointwise(fn)(inp)


register_lowering(torch.ops.approx.tanh)(_tanh_approx_lowering)


class _TanhApprox(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.ops.approx.tanh(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        result = output
        ctx.save_for_backward(result)

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * (1 - result * result)

    @staticmethod
    def vmap(info, in_dims, x):
        return torch.tanh(x), 0


_tanh_approx = _TanhApprox.apply


def generate_tanh_softcap(soft_cap: int, approx: bool = False) -> _score_mod_signature:
    """Returns an tanh bias score_mod given the number of heads H

    Args:
        soft_cap: The soft cap value to use for normalizing logits
        approx: Whether to use the `tanh.approx.` ptx instruction

    Returns:
        tanh_softcap: score_mod
    """
    tanh = _tanh_approx if approx else torch.tanh

    def tanh_softcap(score, b, h, q_idx, kv_idx):
        return soft_cap * tanh(score / soft_cap)

    prefix = "tanh_softcap_approx" if approx else "tanh_softcap"
    tanh_softcap.__name__ = f"{prefix}_{soft_cap}"

    return tanh_softcap



from typing import Dict, Callable


class MaskType(Enum):
    CAUSAL = "causal"
    GLOBAL = "global"
    LOCAL = "local"
    SLIDING_WINDOW = "sliding_window"
    DILATED_SLIDING_WINDOW = "dilated_sliding_window"
    ALIBI_BIAS = "alibi_bias"
    DOC_MASK = "doc_mask"
    TANH_SOFTCAP = "tanh_softcap"



mask_types: Dict[MaskType, Callable] = {
    MaskType.CAUSAL: causal_mask,
    MaskType.TANH_SOFTCAP: generate_tanh_softcap,
    MaskType.DILATED_SLIDING_WINDOW: generate_dilated_sliding_window,
}
