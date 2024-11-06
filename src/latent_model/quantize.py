import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from abc import ABC, abstractmethod


EPS = 1e-6


class GumbelQuantizer(nn.Module):
    def __init__(
        self,
        n_e,
        e_dim,
        num_hiddens,
        straight_through=False,
        kl_weight=5e-4,
        temp_init=1.0,
    ):
        super().__init__()
        self.codebook_size = n_e  # number of embeddings
        self.emb_dim = e_dim  # dimension of embedding
        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.proj = nn.Conv3d(
            num_hiddens, codebook_size, 1
        )  # projects last encoder layer to quantized logits
        self.embed = nn.Embedding(codebook_size, emb_dim)

    def forward(self, z):
        hard = self.straight_through if self.training else True

        logits = self.proj(z)

        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)

        z_q = torch.einsum(
            "b n l h w, n d -> b d l h w", soft_one_hot, self.embed.weight
        )

        qy = F.softmax(logits, dim=1)

        diff = (
            self.kl_weight
            * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=1).mean()
        )

        min_encoding_indices = soft_one_hot.argmax(dim=1)

        return z_q, diff, {"min_encoding_indices": min_encoding_indices}


class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=False,
        normalize=None,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)

        if normalize == "l2_norm":
            self.embedding.weight.data.normal_()
            self.norm = lambda x: F.normalize(x, dim=-1)
        else:
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape
        self.normalize = normalize

    def init_embedding(self, init_embs):
        with torch.no_grad():
            self.embedding.weight.copy_(init_embs.clone())

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        # print(z.shape)
        # z = z.permute(0,2,3,4,1).contiguous()
        # print(z.shape)
        z = rearrange(z, "b c l h w -> b l h w c").contiguous()

        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        embedding_weight = self.embedding.weight

        if self.normalize == "l2_norm":
            embedding_weight = self.norm(embedding_weight)
            z_flattened = self.norm(z_flattened)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding_weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(embedding_weight, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None
        unique = len(torch.unique(min_encoding_indices))

        if self.normalize == "l2_norm":
            z_q = self.norm(z_q)
            z = self.norm(z)

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, "b l h w c -> b c l h w").contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(
                z.shape[0], -1
            )  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        # if self.sane_index_shape:
        # min_encoding_indices = min_encoding_indices.reshape(
        #        z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4])

        return z_q, loss, (unique, perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q


class VectorQuantizer_latent_set(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=False,
        normalize=None,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)

        if normalize == "l2_norm":
            self.embedding.weight.data.normal_()
            self.norm = lambda x: F.normalize(x, dim=-1)
        else:
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape
        self.normalize = normalize

    def init_embedding(self, init_embs):
        with torch.no_grad():
            self.embedding.weight.copy_(init_embs.clone())

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        # print(z.shape)
        # z = z.permute(0,2,3,4,1).contiguous()
        # print(z.shape)

        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        embedding_weight = self.embedding.weight

        if self.normalize == "l2_norm":
            embedding_weight = self.norm(embedding_weight)
            z_flattened = self.norm(z_flattened)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding_weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(embedding_weight, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None
        unique = len(torch.unique(min_encoding_indices))

        if self.normalize == "l2_norm":
            z_q = self.norm(z_q)
            z = self.norm(z)

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(
                z.shape[0], -1
            )  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        # if self.sane_index_shape:
        # min_encoding_indices = min_encoding_indices.reshape(
        #        z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4])

        return z_q, loss, (unique, perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q


class VectorQuantizer2_partial(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=False,
        comp_channel=4,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape
        self.comp_channel = comp_channel

    def init_embedding(self, init_embs):
        with torch.no_grad():
            self.embedding.weight.copy_(init_embs.clone())

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        # print(z.shape)
        # z = z.permute(0,2,3,4,1).contiguous()
        # print(z.shape)
        z = rearrange(z, "b c l h w -> b l h w c").contiguous()

        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened[:, : self.comp_channel] ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight[:, : self.comp_channel] ** 2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn",
                z_flattened[:, : self.comp_channel],
                rearrange(self.embedding.weight[:, : self.comp_channel], "n d -> d n"),
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None
        unique = len(torch.unique(min_encoding_indices))

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, "b l h w c -> b c l h w").contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(
                z.shape[0], -1
            )  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        # if self.sane_index_shape:
        # min_encoding_indices = min_encoding_indices.reshape(
        #        z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4])

        return z_q, loss, (unique, perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q


class VectorQuantizer_Norm(nn.Module):

    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=False,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def normalize_prototypes(self):
        # normalize the prototypes sss
        with torch.no_grad():
            w = self.embedding.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.embedding.weight.copy_(w)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        self.normalize_prototypes()

        z = rearrange(z, "b c l h w -> b l h w c").contiguous()
        z = nn.functional.normalize(z, dim=-1, p=2)
        # print(z.shape)
        # raise "err"
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None
        unique = len(torch.unique(min_encoding_indices))
        # print(min_encoding_indices.shape, unique)

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, "b l h w c -> b c l h w").contiguous()

        return z_q, loss, (unique, perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q


class VectorQuantizer_chamfer(nn.Module):

    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        div_hyp=1.0,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=False,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.div_hyp = div_hyp

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):

        z = rearrange(z, "b c l h w -> b l h w c").contiguous()

        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n")
            )
        )

        # print(d.shape)
        # raise "err"
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None
        unique = len(torch.unique(min_encoding_indices))

        min_encoding_indices_emb = torch.argmin(d.t(), dim=1)
        z_a = z_flattened[min_encoding_indices_emb]
        # print(min_encoding_indices_emb.shape, min_encoding_indices.shape, z_a.shape,z_q.shape, z.shape, self.embedding.weight.shape)
        # raise "err"

        loss = (
            torch.mean((z_q.detach() - z) ** 2)
            + self.beta * torch.mean((z_q - z.detach()) ** 2)
            + self.div_hyp * torch.mean((z_a - self.embedding.weight.detach()) ** 2)
        )

        #  torch.mean((z_q.detach()-z)**2) + self.beta * \
        # loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
        #       torch.mean((z_a - self.embedding.weight.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, "b l h w c -> b c l h w").contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(
                z.shape[0], -1
            )  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4]
            )

        return z_q, loss, (unique, perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q


class Sinkhorn_VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim, args):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim

        # self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding = nn.Linear(self.e_dim, self.n_e, bias=False)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.temperature = args.temperature
        self.sinkhorn_iterations = args.sinkhorn_iterations
        self.improve_numerical_stability = args.improve_numerical_stability
        self.epsilon = args.epsilon

    def normalize_prototypes(self):
        # normalize the prototypes sss
        with torch.no_grad():
            w = self.embedding.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.embedding.weight.copy_(w)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):

        # primary stuff
        softmax = nn.Softmax(dim=1).to(z.device)
        self.normalize_prototypes()

        # reshape z -> (batch, length, height, width, channel) and flatten and normalize
        z = rearrange(z, "b c l h w -> b l h w c").contiguous()
        z_flattened = z.view(-1, self.e_dim)
        z_norm = nn.functional.normalize(z_flattened, dim=1, p=2)
        # print(z_norm.shape)
        # snikorn steps
        output = self.embedding(z_norm)  ## (B l h w) x n_e
        # print(output.shape)
        with torch.no_grad():
            q = output / self.epsilon
            if self.improve_numerical_stability:
                M = torch.max(q)  # dist.all_reduce(M, m#op=dist.ReduceOp.MAX)
                q -= M
            q = torch.exp(q).t()
            q = sinkhorn(q, self.sinkhorn_iterations)
            # print(q.shape)

        p = softmax(output / self.temperature)

        indices = torch.argmax(q, dim=1)
        # print(indices.shape, self.embedding.weight.shape)
        z_q = self.embedding.weight[indices].view(z.shape)
        unique = len(torch.unique(indices))
        # print(z_q.shape, indices)
        # raise "err"
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, "b l h w c -> b c l h w").contiguous()

        loss = -torch.mean(q * torch.log(p + EPS))

        return z_q, loss, (unique, None)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q


#################################################################################################################################


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(
            new_cluster_size, alpha=1 - self.decay
        )

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
            (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )
        # normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)


class EMAVectorQuantizer(nn.Module):
    def __init__(
        self, n_e, e_dim, beta, decay=0.99, eps=1e-5, remap=None, unknown_index="random"
    ):
        super().__init__()
        self.codebook_dim = e_dim
        self.num_tokens = n_e
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        # z, 'b c h w -> b h w c'
        z = rearrange(z, "b c l h w -> b l h w c")
        z_flattened = z.reshape(-1, self.codebook_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            z_flattened.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * torch.einsum("bd,nd->bn", z_flattened, self.embedding.weight)
        )  # 'n d -> d n'

        encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        unique = len(torch.unique(encoding_indices))

        if self.training and self.embedding.update:
            # EMA cluster size
            encodings_sum = encodings.sum(0)
            self.embedding.cluster_size_ema_update(encodings_sum)
            # EMA embedding average
            embed_sum = encodings.transpose(0, 1) @ z_flattened
            self.embedding.embed_avg_ema_update(embed_sum)
            # normalize embed_avg and update weight
            self.embedding.weight_update(self.num_tokens)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        # z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, "b l h w c -> b c l h w")
        return z_q, loss, (unique, perplexity, encodings, encoding_indices)


class EMAVectorQuantizer_Eff(nn.Module):
    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        grid_size,
        decay=0.99,
        eps=1e-5,
        remap=None,
        unknown_index="random",
    ):
        super().__init__()
        self.codebook_dim = e_dim * grid_size
        self.num_tokens = n_e
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        # z, 'b c h w -> b h w c'
        z = rearrange(z, "b c l h w -> b l h w c")
        z_flattened = z.reshape(-1, self.codebook_dim)
        # print(z_flattened.shape)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            z_flattened.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * torch.einsum("bd,nd->bn", z_flattened, self.embedding.weight)
        )  # 'n d -> d n'

        encoding_indices = torch.argmin(d, dim=1)
        print(encoding_indices.shape, encoding_indices.max())
        z_q = self.embedding(encoding_indices).view(z.shape)
        # print(z_q.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        unique = len(torch.unique(encoding_indices))

        if self.training and self.embedding.update:
            # EMA cluster size
            encodings_sum = encodings.sum(0)
            self.embedding.cluster_size_ema_update(encodings_sum)
            # EMA embedding average
            embed_sum = encodings.transpose(0, 1) @ z_flattened
            self.embedding.embed_avg_ema_update(embed_sum)
            # normalize embed_avg and update weight
            self.embedding.weight_update(self.num_tokens)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        # z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, "b l h w c -> b c l h w")
        return z_q, loss, (unique, perplexity, encodings, encoding_indices)


class BaseVectorQuantizer(ABC, nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        :param num_embeddings: size of the latent dictionary (num of embedding vectors).
        :param embedding_dim: size of a single tensor in dict.
        """

        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # create the codebook of the desired size
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # wu/decay init (may never be used)
        self.kl_warmup = None
        self.temp_decay = None

    def init_codebook(self) -> None:
        """
        uniform initialization of the codebook
        """
        nn.init.uniform_(
            self.codebook.weight, -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.IntTensor, float):
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return quantized_x (B, D, H, W), detached codes (B, H*W), latent_loss
        """
        pass

    @abstractmethod
    def vec_to_codes(self, x: torch.Tensor) -> torch.IntTensor:
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return flat codebook indices (B, H * W)
        """
        pass

    @torch.no_grad()
    def get_codebook(self) -> torch.nn.Embedding:
        return self.codebook.weight

    @torch.no_grad()
    def codes_to_vec(self, codes: torch.IntTensor) -> torch.Tensor:
        """
        :param codes: int tensors to decode (B, N).
        :return flat codebook indices (B, N, D)
        """

        quantized = self.get_codebook()[codes]
        return quantized

    def get_codebook_usage(self, index_count: torch.Tensor):
        """
        :param index_count: (n, ) where n is the codebook size, express the number of times each index have been used.
        :return: prob of each index to be used: (n, ); perplexity: float; codebook_usage: float 0__1
        """

        # get used idx as probabilities
        used_indices = index_count / torch.sum(index_count)

        # perplexity
        perplexity = (
            torch.exp(
                -torch.sum(used_indices * torch.log(used_indices + 1e-10), dim=-1)
            )
            .sum()
            .item()
        )

        # get the percentage of used codebook
        n = index_count.shape[0]
        used_codebook = (torch.count_nonzero(used_indices).item() * 100) / n

        return perplexity, used_codebook


class EMAVectorQuantizer_2(BaseVectorQuantizer):

    def __init__(
        self,
        n_e,
        e_dim,
        commitment_cost: float = 0.25,
        decay: float = 0.95,
        epsilon: float = 1e-5,
    ):
        """
        EMA ALGORITHM
        Each codebook entry is updated according to the encoder outputs who selected it.
        The important thing is that the codebook updating is not a loss term anymore.
        Specifically, for every codebook item wi, the mean code mi and usage count Ni are tracked:
        Ni ← Ni · γ + ni(1 − γ),
        mi ← mi · γ + Xnij e(xj )(1 − γ),
        wi ← mi Ni
        where γ is a discount factor

        :param num_embeddings: size of the latent dictionary (num of embedding vectors).
        :param embedding_dim: size of a single tensor in dictionary
        :param commitment_cost: scaling factor for e_loss
        :param decay: decay for EMA updating
        :param epsilon: smoothing parameters for EMA weights
        """

        super().__init__(n_e, e_dim)

        num_embeddings = n_e
        embedding_dim = e_dim

        self.num_tokens = n_e
        self.commitment_cost = commitment_cost

        # EMA does not require grad
        self.codebook.requires_grad_(False)

        # ema parameters
        # ema usage count: total count of each embedding trough epochs
        self.register_buffer("ema_count", torch.zeros(num_embeddings))

        # same size as dict, initialized with normal
        # the updated means
        self.register_buffer(
            "ema_weight", torch.empty((self.num_embeddings, self.embedding_dim))
        )
        self.ema_weight.data.normal_()

        self.decay = decay
        self.epsilon = epsilon

    def forward(self, x):
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return quantized_x (B, D, H, W), detached codes (B, H*W), latent_loss
        """

        b, c, h, w, l = x.shape
        device = x.device

        # Flat input to vectors of embedding dim = C.
        flat_x = rearrange(x, "b c h w l -> (b h w l) c")

        # Calculate distances of each vector w.r.t the dict
        # distances is a matrix (B*H*W, codebook_size)
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_x, self.codebook.weight.t())
        )

        # Get indices of the closest vector in dict, and create a mask on the correct indexes
        # encoding_indices = (num_vectors_in_batch, 1)
        # Mask = (num_vectors_in_batch, codebook_dim)
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=device
        )
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # Quantize and un-flat
        quantized = torch.matmul(encodings, self.codebook.weight)

        # Use EMA to update the embedding vectors
        # Update a codebook vector as the mean of the encoder outputs that are closer to it
        # Calculate the usage count of codes and the mean code, then update the codebook vector dividing the two
        if self.training:
            with torch.no_grad():
                ema_count = self.get_buffer("ema_count") * self.decay + (
                    1 - self.decay
                ) * torch.sum(encodings, 0)

                # Laplace smoothing of the ema count
                self.ema_count = (
                    (ema_count + self.epsilon)
                    / (b + self.num_embeddings * self.epsilon)
                    * b
                )

                dw = torch.matmul(encodings.t(), flat_x)
                self.ema_weight = (
                    self.get_buffer("ema_weight") * self.decay + (1 - self.decay) * dw
                )

                self.codebook.weight.data = self.get_buffer(
                    "ema_weight"
                ) / self.get_buffer("ema_count").unsqueeze(1)

        # Loss function (only the inputs are updated)
        e_loss = self.commitment_cost * F.mse_loss(quantized.detach(), flat_x)

        # during backpropagation quantized = inputs (copy gradient trick)
        quantized = flat_x + (quantized - flat_x).detach()

        quantized = rearrange(quantized, "(b h w l) c -> b c h w l", b=b, h=h, w=w)
        # encoding_indices = rearrange(encoding_indices, '(b h w)-> b (h w)', b=b, h=h, w=w).detach()

        encodings = F.one_hot(encoding_indices, self.num_tokens).type(x.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        unique = len(torch.unique(encoding_indices))

        return quantized, e_loss, (unique, perplexity, encodings, encoding_indices)

    @torch.no_grad()
    def vec_to_codes(self, x: torch.Tensor) -> torch.IntTensor:
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return flat codebook indices (B, H * W)
        """
        b, c, h, w = x.shape

        # Flat input to vectors of embedding dim = C.
        flat_x = rearrange(x, "b c h w -> (b h w) c")

        # Calculate distances of each vector w.r.t the dict
        # distances is a matrix (B*H*W, codebook_size)
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_x, self.codebook.weight.t())
        )

        # Get indices of the closest vector in dict
        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = rearrange(
            encoding_indices, "(b h w) -> b (h w)", b=b, h=h, w=w
        )

        return encoding_indices
