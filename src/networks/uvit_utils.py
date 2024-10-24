import math
from abc import abstractmethod
from inspect import isfunction

import torch
import torch as th
import torch.nn.functional as F
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from torch import nn, einsum

if hasattr(torch, "_dynamo"):
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 1024

allow_ops_in_compiled_graph()


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, skip_h=None, res_emb=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb, skip_h=skip_h, res_emb=res_emb)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, Transformer_Block):
                x = layer(x, emb, context=context)
            else:
                x = layer(x)
        return x


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


### REMARK: Change to 4
def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, channels_output, use_conv, output_res, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        self.output_res = output_res
        stride = 2
        if use_conv:
            self.op = conv_nd(
                dims, channels, channels_output, 3, stride=stride, padding=1
            )
        else:
            self.op = avg_pool_nd(dims, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D
    """

    def __init__(self, channels, channels_output, use_conv, output_res, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        self.output_res = output_res
        if use_conv:
            self.conv = conv_nd(dims, channels, channels_output, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 1024

        x = F.interpolate(x, scale_factor=2, mode="trilinear")
        if self.use_conv:
            x = self.conv(x)

        if self.output_res < x.size(-1):
            x = x[..., :-1]
        if self.output_res < x.size(-2):
            x = x[..., :-1, :]
        if self.output_res < x.size(-3):
            x = x[..., :-1, :, :]

        return x


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        embedding_dim,
        expansion_factor=4,
        dropout=0.0,
        activation=nn.SiLU(),
    ):
        super().__init__()
        self.transformer_dropout = dropout
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.dense1 = nn.Linear(hidden_size, hidden_size * expansion_factor)
        self.scale = nn.Sequential(
            nn.SiLU(), nn.Linear(embedding_dim, hidden_size * expansion_factor)
        )
        self.shift = nn.Sequential(
            nn.SiLU(), nn.Linear(embedding_dim, hidden_size * expansion_factor)
        )
        self.out = nn.Linear(hidden_size * expansion_factor, hidden_size)
        self.activation = activation

        # Apply zero initialization
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # Initialize biases to zero

    def forward(self, x, emb):
        B, HWL, C = x.shape
        x = self.norm(x)
        mlp_h = self.dense1(x)
        scale = self.scale(emb)
        shift = self.shift(emb)
        mlp_h = self.activation(mlp_h)  # F.silu(mlp_h)
        mlp_h = mlp_h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        if self.transformer_dropout > 0.0:
            mlp_h = nn.functional.dropout(
                mlp_h, p=self.transformer_dropout, training=self.training
            )
        out = self.out(mlp_h)
        return out


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=4, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.norm = nn.LayerNorm(query_dim, elementwise_affine=False)

        self.heads = heads

        self.to_q = nn.Sequential(
            nn.SiLU(), nn.Linear(query_dim, inner_dim)
        )  # nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Sequential(
            nn.SiLU(), nn.Linear(context_dim, inner_dim)
        )  # nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Sequential(
            nn.SiLU(), nn.Linear(context_dim, inner_dim)
        )  # nn.Linear(context_dim, inner_dim, bias=False)

        self.norm_q = nn.LayerNorm(inner_dim)
        self.norm_k = nn.LayerNorm(inner_dim)

        self.to_out = nn.Sequential(
            nn.SiLU(), zero_module(nn.Linear(inner_dim, query_dim))
        )

        # Apply zero initialization
        # nn.init.zeros_(self.to_out.weight)
        # nn.init.zeros_(self.to_out.bias)

    def forward(self, x, context=None):
        B, HWL, C = x.shape
        if context is not None:
            B, T, TC = context.shape
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)

        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = q * q.shape[-1] ** -0.5

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        weights = einsum("b i d, b j d -> b i j", q, k)  ## unnormalized att maps.
        weights = weights.softmax(dim=-1)  ## normalized att maps
        attn_vals = einsum("b i j, b j d -> b i d", weights, v)
        out = rearrange(attn_vals, "(b h) n d -> b n (h d)", h=h)

        return self.to_out(out)


class Transformer_Block(nn.Module):
    def __init__(
        self,
        query_dim,
        emb_dim,
        context_dim=None,
        heads=4,
        dropout=0.0,
        activation=nn.SiLU(),
    ):
        super().__init__()
        self.mlp = MLP(
            query_dim,
            emb_dim,
            expansion_factor=4,
            dropout=dropout,
            activation=activation,
        )
        self.att = Attention(
            query_dim, context_dim=context_dim, heads=heads, dim_head=64
        )

    def forward(self, x, emb, context=None):
        x = x + self.mlp(x, emb)
        x = x + self.att(x, context=context)
        return x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        activation=SiLU(),
        skip_h=None,
        learnable_skip_r=None,
        res_emb_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.res_emb_channels = res_emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        if skip_h is not None:
            self.skip_norm = normalization(channels)
            self.learnable_skip_r = learnable_skip_r
            if learnable_skip_r is not None:
                self.skip_learn_f = nn.Sequential(
                    nn.Linear(channels, channels // learnable_skip_r),
                    nn.ReLU(),
                    nn.Linear(channels // learnable_skip_r, channels),
                    nn.Sigmoid(),
                )

        self.in_norm = normalization(channels)
        self.act1 = activation
        self.in_conv = conv_nd(dims, channels, self.out_channels, 3, padding=1)

        self.emb_layers = nn.Sequential(
            activation,
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            activation,
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )
        if self.res_emb_channels is not None:
            self.linear_res_emb = linear(self.res_emb_channels, self.out_channels)

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, skip_h=None, res_emb=None):
        B, H, W, L, C = x.shape
        h = self.in_norm(x)
        if skip_h is not None:
            skip_h = self.skip_norm(skip_h)
            if self.learnable_skip_r is not None:
                # print(skip_h.shape)
                averaged_skip = skip_h.mean(dim=(-3, -2, -1))
                # print(averaged_skip.shape)
                k = self.skip_learn_f(averaged_skip)
                # print(k.shape, h.shape, skip_h.shape)
                h = h + k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * skip_h
                # print(h.shape)
                # raise "err"
            else:
                h = (h + skip_h) / math.sqrt(2)

        h = self.act1(h)
        h = self.in_conv(h)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)

            if self.res_emb_channels is not None:
                torch._assert(res_emb is not None, "res_emb is None")
                z_res = self.linear_res_emb(res_emb)
                while len(z_res.shape) < len(scale.shape):
                    z_res = z_res[..., None]
                h = (out_norm(h) * (1 + scale) + shift) * z_res
            else:
                h = out_norm(h) * (1 + scale) + shift

            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class Condition_UVIT(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        input_size,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        activation=None,
        context_dim=None,
        use_single_context=False,
        context_emb_dim=None,
        with_self_att=False,
        num_transformer_blocks=8,
        add_num_register=0,
        learnable_skip_r=None,
        add_condition_time_ch=None,
        add_condition_input_ch=None,
        no_cross_attention=False,
    ):
        super().__init__()

        self.with_self_att = with_self_att
        self.activation = activation if activation is not None else SiLU()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.no_cross_attention = no_cross_attention

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.LayerNorm(time_embed_dim),
            self.activation,
            # linear(time_embed_dim, time_embed_dim),
        )

        self.add_condition_time_ch = add_condition_time_ch

        if add_condition_time_ch is not None:
            self.mapping_layer = nn.Sequential(
                linear(context_dim, 256),
                self.activation,
                linear(256, 256),
                self.activation,
            )
            self.time_cond_proj = nn.Sequential(
                linear(256, 128), self.activation, linear(128, add_condition_time_ch)
            )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.add_condition_input_ch = add_condition_input_ch

        if self.add_condition_input_ch is not None:
            self.mapping_input_layer = nn.Sequential(
                linear(context_dim, 256),
                self.activation,
                linear(256, 256),
                self.activation,
            )
            self.input_cond_proj = nn.Sequential(
                linear(256, 128), self.activation, linear(128, add_condition_input_ch)
            )
            in_channels = in_channels + self.add_condition_input_ch

        if use_single_context:
            # create a few layers of embedding like the time embedding
            self.context_embed = nn.Sequential(
                linear(context_emb_dim, time_embed_dim),
                self.activation,
                linear(time_embed_dim, time_embed_dim),
            )

        att_size = input_size
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        res_channel = self.add_condition_time_ch
        self.output_reses = [att_size]
        for level in range(len(num_res_blocks)):
            mult = channel_mult[level]
            layers = []
            for _ in range(num_res_blocks[level]):
                layers.append(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        activation=self.activation,
                        res_emb_channels=res_channel,
                    )
                )

                ch = mult * model_channels
                # self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            # if level != len(num_res_blocks) - 1:
            att_size = (att_size - 3 + 2) // 2 + 1
            self.output_reses.append(att_size)

            layers.append(
                Downsample(ch, ch, conv_resample, output_res=att_size, dims=dims)
            )
            self.input_blocks.append(TimestepEmbedSequential(*layers))
            input_block_chans.append(ch)
            ds *= 2
            # att_size //= 2

        self.att_size = att_size
        self.add_num_register = add_num_register
        if self.add_num_register > 0:
            self.pos_emb = nn.init.trunc_normal_(
                nn.Parameter(
                    torch.zeros(
                        1, (att_size * att_size * att_size) + self.add_num_register, ch
                    )
                ),
                0.0,
                0.01,
            )
            self.register_emb = nn.init.trunc_normal_(
                nn.Parameter(torch.zeros(1, self.add_num_register, ch)), 0.0, 0.01
            )
        else:
            self.pos_emb = nn.init.trunc_normal_(
                nn.Parameter(torch.zeros(1, att_size * att_size * att_size, ch)),
                0.0,
                0.01,
            )

        self.self_middle_blocks = nn.ModuleList([])
        self.middle_blocks = nn.ModuleList([])

        for _ in range(num_transformer_blocks):
            if self.with_self_att:
                self.self_middle_blocks.append(
                    TimestepEmbedSequential(
                        Transformer_Block(
                            ch,
                            time_embed_dim,
                            context_dim=None,
                            heads=num_heads,
                            dropout=dropout,
                            activation=nn.SiLU(),
                        )
                    )
                )

            cross_attn_dim = None if no_cross_attention else context_dim
            self.middle_blocks.append(
                TimestepEmbedSequential(
                    Transformer_Block(
                        ch,
                        time_embed_dim,
                        context_dim=cross_attn_dim,
                        heads=num_heads,
                        dropout=dropout,
                        activation=nn.SiLU(),
                    )
                )
            )

        self.output_blocks = nn.ModuleList([])

        for level in reversed(range(len(num_res_blocks))):

            ### fixing transition
            mult = channel_mult[level]
            layers = [
                Upsample(
                    ch,
                    model_channels * mult,
                    conv_resample,
                    self.output_reses[level],
                    dims=dims,
                )
            ]
            ch = model_channels * mult

            for _ in range(num_res_blocks[level]):
                layers.append(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        activation=self.activation,
                        skip_h=True,
                        learnable_skip_r=learnable_skip_r,
                        res_emb_channels=res_channel,
                    )
                )
                ch = model_channels * mult
                # print(ch)
            self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            self.activation,
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return torch.float32  # FIXED
        # return next(self.input_blocks.parameters()).dtype

    def forward(
        self,
        x,
        timesteps=None,
        y=None,
        low_cond=None,
        latent_codes=None,
        single_context=None,
    ):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param low_cond: an [N x C x ...]  Tensor of condition.
        :return: an [N x C x ...] Tensor of outputs.
        """

        if timesteps is None:
            timesteps = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # print("aa",x.shape, latent_codes.shape, y, low_cond, timesteps)
        ## concat the condition
        if low_cond is not None:
            x = th.cat((x, low_cond), dim=1)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.add_condition_input_ch is not None:
            latent_codes_ = self.mapping_input_layer(latent_codes)
            average_latent_codes = torch.mean(latent_codes_, 1)
            # print(average_latent_codes.shape)
            input_cond = (
                self.input_cond_proj(average_latent_codes)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            # print(input_cond.shape, x.shape)
            x = th.cat(
                (x, input_cond.repeat(1, 1, x.size(2), x.size(3), x.size(4))), dim=1
            )

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.add_condition_time_ch is not None:
            latent_codes_ = self.mapping_layer(latent_codes)
            # print(latent_codes_.shape)
            average_latent_codes = torch.mean(latent_codes_, 1)
            # print(average_latent_codes.shape)
            res_emb = self.time_cond_proj(average_latent_codes)
            # print(input_cond.shape, emb.shape)
            # emb = th.cat((emb, input_cond), dim = -1)
            # print(emb.shape)
        else:
            res_emb = None

        if single_context is not None:
            emb_context = self.context_embed(single_context)
            emb = emb + emb_context

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x  # .type(self.inner_dtype)

        if latent_codes is not None:
            latent_codes = latent_codes.type(h.type())
        else:
            latent_codes = None

        # print("input_num", len(self.input_blocks))
        hs = [h]

        for module in self.input_blocks:
            module_rest, module_downsample = module[:-1], module[-1]
            h = module_rest(h, emb, context=latent_codes, res_emb=res_emb)
            # print("in", h.shape)
            hs.append(h)

            # downsample only
            h = module_downsample(h)

        h = h.permute(0, 2, 3, 4, 1)  # BCHWL --> BHWLC
        # print("mid", h.shape)
        h = h.view(h.size(0), -1, h.size(-1))  # BHWLC --> B(HWL)C

        if self.add_num_register > 0:
            # print("reg", self.register_emb.shape)
            h = torch.cat([h, self.register_emb.repeat(h.size(0), 1, 1)], dim=1)

        # print("mid", h.shape)
        h = h + self.pos_emb
        # print("mid", h.shape, self.pos_emb.shape)
        for idx, module in enumerate(self.middle_blocks):
            if self.with_self_att:
                h = self.self_middle_blocks[idx](h, emb, context=None)

            if self.no_cross_attention:
                h = module(h, emb, context=None)
            else:
                h = module(h, emb, context=latent_codes)
            # print("T", h.shape)

        if self.add_num_register > 0:
            h = h[:, : -self.add_num_register, :]

        h = h.permute(0, 2, 1)  # B(HWL)C --> BC(HWL)
        h = h.view(
            h.size(0), h.size(1), self.att_size, self.att_size, self.att_size
        )  # B(HWL)C --> BCHWL
        # print("T", h.shape)

        # print("output_num", len(self.output_blocks))
        for idx, module in enumerate(self.output_blocks):
            # handling for non-even inputs
            # h = F.interpolate(h, size= hs[-1].size()[-3:], mode='trilinear')
            # skip_h = hs[-1]
            # print("out", h.shape, skip_h.shape)
            # cat_in = th.cat([h, hs.pop()], dim=1)
            # print("out", cat_in.shape, emb.shape)

            # print(f"{idx} : {h.size()} : {hs[-1].size()}")
            h = module(h, emb, context=latent_codes, skip_h=hs[-1], res_emb=res_emb)
            hs.pop()
            # print("out", h.shape)
        # print(len(hs))
        h = h.type(x.dtype)
        return self.out(h)
