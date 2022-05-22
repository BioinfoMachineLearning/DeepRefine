import math
from collections import namedtuple
from functools import partial, reduce
from math import gcd
from operator import itemgetter
from operator import mul

import dgl
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from linformer import LinformerSelfAttention
from local_attention import LocalAttention
from product_key_memory import PKM
from torch import nn, einsum
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
from torch_geometric.data import Data


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from Linear Attention Transformer (https://github.com/lucidrains/linear-attention-transformer):
# -------------------------------------------------------------------------------------------------------------------------------------
# For routing arguments into the functions of the reversible layer
def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args


def layer_drop(layers, prob):
    to_drop = torch.empty(len(layers)).uniform_(0, 1) < prob
    blocks = [block for block, drop in zip(layers, to_drop) if not drop]
    blocks = layers[:1] if len(blocks) == 0 else blocks
    return blocks


# Following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


# Heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=2)

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)

        return x, dx


class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, args):
        ctx.args = args
        for block, kwarg in zip(blocks, args):
            x = block(x, **kwarg)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        args = ctx.args
        for block, kwargs in zip(ctx.blocks[::-1], args[::-1]):
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route={}, layer_dropout=0.):
        super().__init__()
        assert all(len(route) == len(layers) for route in
                   args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        if self.training and self.layer_dropout > 0:
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x


class ReversibleSequence(nn.Module):
    def __init__(self, blocks, args_route={}, layer_dropout=0.):
        super().__init__()
        self.args_route = args_route
        self.layer_dropout = layer_dropout
        self.blocks = nn.ModuleList([ReversibleBlock(f=f, g=g) for f, g in blocks])

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim=-1)

        blocks = self.blocks
        args = route_args(self.args_route, kwargs, len(blocks))
        args = list(map(lambda x: {'f_args': x[0], 'g_args': x[1]}, args))

        layers_and_args = list(zip(blocks, args))

        if self.training and self.layer_dropout > 0:
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)
            blocks, args = map(lambda ind: list(map(itemgetter(ind), layers_and_args)), (0, 1))

        out = _ReversibleFunction.apply(x, blocks, args)
        return torch.stack(out.chunk(2, dim=-1)).mean(dim=0)


# Namedtuple settings
LinformerSettings = namedtuple('LinformerSettings', ['k'])
LinformerContextSettings = namedtuple('LinformerContextSettings', ['seq_len', 'k'])


# Helper functions
def exists(val):
    return val is not None


def default(value, d):
    return d if not exists(value) else value


def always(value):
    return lambda *args, **kwargs: value


def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val


def safe_div(n, d, eps=1e-6):
    return n.div_(d + eps)


def lcm(*numbers):
    return int(reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))


def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def expand_dim(t, dim, k, unsqueeze=True):
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


# Helper classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)


class ProjectInOut(nn.Module):
    def __init__(self, fn, dim_in, dim_out, project_out=True):
        super().__init__()
        self.fn = fn
        self.project_in = nn.Linear(dim_in, dim_out)
        self.project_out = nn.Linear(dim_out, dim_in) if project_out else nn.Identity()

    def forward(self, x, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, **kwargs)
        x = self.project_out(x)
        return x


# Token shifting helper classes
def shift(t, amount, mask=None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value=0.)


class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask=mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim=-1)
        return self.fn(x, **kwargs)


# Positional embeddings
class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)[None, :, :]


# Sinusoidal positional embeddings
class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)


# Rotary positional embedding helpers
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d j -> ... (d j)')


def apply_rotory_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j=2)
    sin, cos = sinu_pos.unbind(dim=-2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j=2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k


# Feedforward layers
class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., activation=None, glu=False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


# Self-attention layers
def linear_attn(q, k, v, kv_mask=None):
    dim = q.shape[-1]

    if exists(kv_mask):
        mask_value = max_neg_value(q)
        mask = kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, mask_value)
        v = v.masked_fill_(~mask, 0.)
        del mask

    q = q.softmax(dim=-1)
    k = k.softmax(dim=-2)

    q = q * dim ** -0.5

    context = einsum('bhnd,bhne->bhde', k, v)
    attn = einsum('bhnd,bhde->bhne', q, context)
    return attn.reshape(*q.shape)


def causal_linear_attn(q, k, v, kv_mask=None, bucket_size=None, eps=1e-3):
    b, h, n, e, dtype = *q.shape, q.dtype
    bucket_size = default(bucket_size, 64)
    bucket_size = max(bucket_size, 1)
    assert bucket_size == 0 or (
            n % bucket_size) == 0, f'sequence length {n} must be divisible by the bucket size {bucket_size} for causal linear attention'

    q = q.softmax(dim=-1)
    k = torch.exp(k).type(dtype).clone()

    q = q * e ** -0.5

    if exists(kv_mask):
        mask = kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, 0.)
        v = v.masked_fill_(~mask, 0.)
        del mask

    bucket_fn = lambda x: x.reshape(*x.shape[:-2], -1, bucket_size, e)
    b_q, b_k, b_v = map(bucket_fn, (q, k, v))

    b_k_sum = b_k.sum(dim=-2)
    b_k_cumsum = b_k_sum.cumsum(dim=-2).type(dtype)

    context = einsum('bhund,bhune->bhude', b_k, b_v)
    context = context.cumsum(dim=-3).type(dtype)

    if bucket_size > 1:
        context = F.pad(context, (0, 0, 0, 0, 1, 0), value=0.)
        context, _ = split_at_index(2, -1, context)

        b_k_cumsum = F.pad(b_k_cumsum, (0, 0, 1, 0), value=0.)
        b_k_cumsum, _ = split_at_index(2, -1, b_k_cumsum)

    D_inv = 1. / einsum('bhud,bhund->bhun', b_k_cumsum, b_q).clamp(min=eps)
    attn = einsum('bhund,bhude,bhun->bhune', b_q, context, D_inv)
    return attn.reshape(*q.shape)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, causal=False, dim_head=None, blindspot_size=1, n_local_attn_heads=0,
                 local_attn_window_size=128, receives_context=False, dropout=0., attn_dropout=0.):
        super().__init__()
        assert dim_head or (dim % heads) == 0, 'embedding dimension must be divisible by number of heads'
        d_heads = default(dim_head, dim // heads)

        self.heads = heads
        self.d_heads = d_heads
        self.receives_context = receives_context

        self.global_attn_heads = heads - n_local_attn_heads
        self.global_attn_fn = linear_attn if not causal else partial(causal_linear_attn, bucket_size=blindspot_size)

        self.local_attn_heads = n_local_attn_heads
        self.local_attn = LocalAttention(local_attn_window_size, causal=causal, dropout=attn_dropout)

        self.to_q = nn.Linear(dim, d_heads * heads, bias=False)

        kv_heads = heads

        self.kv_heads = kv_heads
        self.to_k = nn.Linear(dim, d_heads * kv_heads, bias=False)
        self.to_v = nn.Linear(dim, d_heads * kv_heads, bias=False)

        self.to_out = nn.Linear(d_heads * heads, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_mask=None, context=None, context_mask=None, pos_emb=None, **kwargs):
        assert not (self.receives_context and not exists(
            context)), 'context must be supplied if self attention is in receives context mode'

        if not self.receives_context:
            q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
        else:
            q, k, v = (self.to_q(x), self.to_k(context), self.to_v(context))

        b, t, e, h, dh = *q.shape, self.heads, self.d_heads

        merge_heads = lambda x: x.reshape(*x.shape[:2], -1, dh).transpose(1, 2)

        q, k, v = map(merge_heads, (q, k, v))

        if exists(pos_emb) and not self.receives_context:
            q, k = apply_rotory_pos_emb(q, k, pos_emb)

        out = []

        split_index_fn = partial(split_at_index, 1, self.local_attn_heads)

        (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))

        has_local, has_global = map(lambda x: x.shape[1] > 0, (lq, q))

        if has_local:
            local_out = self.local_attn(lq, lk, lv, input_mask=input_mask)
            out.append(local_out)

        if has_global:
            kv_mask = input_mask if not self.receives_context else context_mask
            global_out = self.global_attn_fn(q, k, v, kv_mask=kv_mask)
            out.append(global_out)

        attn = torch.cat(out, dim=1)
        attn = attn.transpose(1, 2).reshape(b, t, -1)
        return self.dropout(self.to_out(attn))


# Transformer class
class FoldAxially(nn.Module):
    def __init__(self, axial_dim, fn):
        super().__init__()
        self.fn = fn
        self.axial_dim = axial_dim

    def forward(self, x, input_mask=None, **kwargs):
        b, t, d, ax = *x.shape, self.axial_dim
        x = x.reshape(b, -1, ax, d).transpose(1, 2).reshape(b * ax, -1, d)

        mask = None
        if exists(input_mask):
            mask = input_mask.reshape(b, -1, ax).transpose(1, 2).reshape(b * ax, -1)

        x = self.fn(x, input_mask=mask, **kwargs)
        x = x.reshape(b, ax, -1, d).transpose(1, 2).reshape(b, t, d)
        return x


class LinearAttentionTransformer(nn.Module):
    """A global multi-head attention (MHA) mechanism as a PyTorch module.
    Inspired by the implementation of the Linear Attention Transformer by
    lucidrains (https://github.com/lucidrains/linear-attention-transformer)."""

    def __init__(
            self,
            dim,
            depth,
            max_seq_len,
            heads=8,
            dim_head=None,
            causal=False,
            ff_chunks=1,
            ff_glu=False,
            ff_dropout=0.,
            attn_layer_dropout=0.,
            attn_dropout=0.,
            reversible=False,
            blindspot_size=1,
            n_local_attn_heads=0,
            local_attn_window_size=128,
            receives_context=False,
            attend_axially=False,
            pkm_layers=tuple(),
            linformer_settings=None,
            context_linformer_settings=None,
            shift_tokens=False
    ):
        super().__init__()
        assert not (causal and exists(
            linformer_settings)), 'Linformer self attention layer can only be used for non-causal networks'
        assert not exists(linformer_settings) or isinstance(linformer_settings,
                                                            LinformerSettings), 'Linformer self-attention settings must be a LinformerSettings namedtuple'
        assert not exists(context_linformer_settings) or isinstance(context_linformer_settings,
                                                                    LinformerContextSettings), 'Linformer contextual self-attention settings must be a LinformerSettings namedtuple'

        if type(n_local_attn_heads) is not tuple:
            n_local_attn_heads = tuple([n_local_attn_heads] * depth)

        assert len(n_local_attn_heads) == depth, 'local attention heads tuple must have the same length as the depth'
        assert all([(local_heads <= heads) for local_heads in
                    n_local_attn_heads]), 'number of local attn heads must be less than the maximum number of heads'

        layers = nn.ModuleList([])

        for ind, local_heads in zip(range(depth), n_local_attn_heads):
            layer_num = ind + 1
            use_pkm = layer_num in cast_tuple(pkm_layers)

            parallel_net = Chunk(ff_chunks, FeedForward(dim), along_dim=1) if not use_pkm else PKM(dim)

            if not exists(linformer_settings):
                attn = SelfAttention(dim, heads, causal, dim_head=dim_head, blindspot_size=blindspot_size,
                                     n_local_attn_heads=local_heads, local_attn_window_size=local_attn_window_size,
                                     dropout=attn_layer_dropout, attn_dropout=attn_dropout)
            else:
                attn = LinformerSelfAttention(dim, max_seq_len, heads=heads, dim_head=dim_head, dropout=attn_dropout,
                                              **linformer_settings._asdict())

            if shift_tokens:
                shifts = (1, 0, -1) if not causal else (1, 0)
                attn, parallel_net = map(partial(PreShiftTokens, shifts), (attn, parallel_net))

            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, parallel_net)
            ]))

            if attend_axially:
                layers.append(nn.ModuleList([
                    PreNorm(dim, FoldAxially(local_attn_window_size,
                                             SelfAttention(dim, heads, causal, dropout=attn_layer_dropout,
                                                           attn_dropout=attn_dropout))),
                    PreNorm(dim, Chunk(ff_chunks, FeedForward(dim, glu=ff_glu, dropout=ff_dropout), along_dim=1))
                ]))

            if receives_context:
                if not exists(context_linformer_settings):
                    attn = SelfAttention(dim, heads, dim_head=dim_head, dropout=attn_layer_dropout,
                                         attn_dropout=attn_dropout, receives_context=True)
                else:
                    attn = LinformerSelfAttention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout,
                                                  **context_linformer_settings._asdict())

                layers.append(nn.ModuleList([
                    PreNorm(dim, attn),
                    PreNorm(dim, Chunk(ff_chunks, FeedForward(dim, glu=ff_glu, dropout=ff_dropout), along_dim=1))
                ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        axial_layer = ((True, False),) if attend_axially else tuple()
        attn_context_layer = ((True, False),) if receives_context else tuple()
        route_attn = ((True, False), *axial_layer, *attn_context_layer) * depth
        route_context = ((False, False), *axial_layer, *attn_context_layer) * depth

        context_route_map = {'context': route_context, 'context_mask': route_context} if receives_context else {}
        attn_route_map = {'input_mask': route_attn, 'pos_emb': route_attn}
        self.layers = execute_type(layers, args_route={**attn_route_map, **context_route_map})

        self.pad_to_multiple = lcm(
            1 if not causal else blindspot_size,
            1 if all([(h == 0) for h in n_local_attn_heads]) else local_attn_window_size
        )

    def forward(self, feats: torch.Tensor, **kwargs):
        """
        Compute attention scores with a PyTorch Tensor of feature values.

        Parameters
        ----------
        feats: torch.Tensor
            Input representations.
        Returns
        ----------
        torch.Tensor
            MHA representations.
        """
        return self.layers(feats, **kwargs)


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepRefine (https://github.com/BioinfoMachineLearning/DeepRefine):
# -------------------------------------------------------------------------------------------------------------------------------------
def convert_dgl_graph_to_pyg_graph(dgl_graph: dgl.DGLGraph, ca_only: bool, labels=True) -> Data:
    """Convert a given DGLGraph into its corresponding representation as a PyTorch Geometric graph."""
    edge_index = torch.vstack([dgl_graph.edges()[0], dgl_graph.edges()[1]]).long().contiguous()
    pyg_graph = Data(
        ndata=dgl_graph.ndata['f'],
        edge_index=edge_index,
        edata=dgl_graph.edata['f'],
        y=dgl_graph.ndata['x_true'] if labels else torch.zeros_like(dgl_graph.ndata['x_pred']),
        pos=dgl_graph.ndata['x_pred'],
        atom_type=dgl_graph.ndata['atom_type'],
        labeled=dgl_graph.ndata['labeled'],
        interfacing=dgl_graph.ndata['interfacing'],
        covalent_radius=dgl_graph.ndata['covalent_radius'],
        chain_id=dgl_graph.ndata['chain_id'],
        residue_number=dgl_graph.ndata['residue_number'],
        surf_prox=dgl_graph.ndata['surf_prox'],
        is_ca_atom=dgl_graph.ndata['is_ca_atom'],
        dihedral_angles=dgl_graph.ndata['dihedral_angles'] if ca_only else torch.zeros((dgl_graph.num_nodes(), 6)),
        pos_enc=dgl_graph.edata['pos_enc'],
        rel_pos=dgl_graph.edata['rel_pos'],
        r=dgl_graph.edata['r'],
        bond_type=torch.zeros((dgl_graph.num_nodes(), 1)) if ca_only else dgl_graph.edata['bond_type'],
        in_same_chain=dgl_graph.edata['in_same_chain'],
        rel_geom_feats=dgl_graph.edata['rel_geom_feats']
    )
    return pyg_graph
