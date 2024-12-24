"""
revise from https://github.com/dongyangli-del/EEG_Image_decode ATM_S50, tnks!

notes 1: for 250hz downsample to 200hz, changed Proj_eeg's embedding dim from 1440 to 1040
"""
import os
from scipy.stats import rankdata
import torch
import numpy as np
import torch.nn as nn
from einops.layers.torch import Rearrange
from lavis.models.clip_models.loss import ClipLoss
from torch import Tensor
from torch_geometric.nn import GATConv
import math

# OK
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + 1])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer('pe', pe) # register_buffer 后，不会看作参数

    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1).to(x.device)
        x = x + pe
        return x

# OK
class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        #或者在tokenize之后添加pos_encoder?
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]

# NO
class ChannelConv(nn.Module):
    def __init__(self, channel=None):
        super().__init__()
        # revised from shallownet
        self.channelconv = nn.Sequential(
            nn.Conv2d(40, 40, (channel, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.channelconv(x)


class ImageProjection(nn.Module):
    """
    reviseded from ImageProjection in IP-Adapter
    挪用到自己的unet上，给embedding赋予seq
    """
    def __init__(
        self,
        image_embed_dim: int = 1024,
        cross_attention_dim: int = 1024,
        num_image_text_embeds: int = 4,
    ):
        super().__init__()

        self.num_image_text_embeds = num_image_text_embeds
        self.image_embeds = nn.Linear(image_embed_dim, self.num_image_text_embeds * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.FloatTensor):
        batch_size = image_embeds.shape[0]
        shape_revise = image_embeds.shape[1:]
        # image
        image_embeds = self.image_embeds(image_embeds)
        image_embeds = image_embeds.reshape(batch_size, self.num_image_text_embeds, -1)
        image_embeds = self.norm(image_embeds)

        return image_embeds


# NO
class EEG_GAT(nn.Module):
    def __init__(self, in_channels=250, out_channels=250):
        super(EEG_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GATConv(in_channels=in_channels, out_channels=out_channels, heads=1) #
        # self.conv2 = GATConv(in_channels=out_channels, out_channels=out_channels, heads=1)

        self.num_channels = 64
        # Create a list of tuples representing all possible edges between channels
        self.edge_index_list = torch.Tensor(
            [(i, j) for i in range(self.num_channels) for j in range(self.num_channels) if i != j])
        # Convert the list of tuples to a tensor
        self.edge_index = self.edge_index_list.clone().detach().to(torch.long).t().contiguous()
        # self.edge_index = torch.tensor(self.edge_index_list, dtype=torch.long).t().contiguous()

    def forward(self, x):
        self.edge_index = self.edge_index.to(x.device)
        x = x.unsqueeze(1)
        batch_size, _, num_channels, num_features = x.size()
        x = x.view(batch_size * num_channels, num_features)
        x = self.conv1(x, self.edge_index)
        x = x.view(batch_size, num_channels, -1)
        return x

# OK
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, use_channel_attn = False): # use_ori = False
        super().__init__()
        # revised from shallownet
        self.use_channel_attn = use_channel_attn
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)), # (batch, 40, 63, 226)
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.1)
        )
        if use_channel_attn is True:
            self.channel_length = [7, 3, 4, 5, 3, 4, 3, 3, 3, 4, 3, 6, 4, 3, 8]
            self.func_area = [
                [0, 1, 2, 3, 4, 5, 6],
                [7, 8, 9], [15, 16, 17, 18],
                [10, 11, 19, 20, 21], [12, 13, 14],
                [22, 23, 24, 25], [26, 27, 28],
                [29, 30, 31], [32, 33, 34], [35, 36, 37, 38],
                [46, 47, 48], [39, 40, 41, 49, 50, 51], [42, 43, 44, 45],
                [52, 53, 54], [56, 57, 58, 60, 61, 62, 55, 59],
            ]  # 17 regions
            self.blocks = nn.ModuleList(ChannelConv(channel=channel_num) for i, channel_num in enumerate(self.channel_length))
            self.sumChannelConv = ChannelConv(channel=17)
        else:
            self.tsconv = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Conv2d(40, 40, (63, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(0.1),
            )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)  # b, 1, 63, 250 = x.shape
        if self.use_channel_attn is True:
            x = self.temporalconv(x) # batch, 40, 63, 36
            x_new = []
            for i, area in enumerate(self.func_area):
                x_new.append([])
                print(len(area))
                for j, element in enumerate(area):
                    x_new[i].append(x[:, :, self.func_area[i][j], :])
                x_new[i] = torch.stack(x_new[i], dim=2)
            del x
            for i, blk in enumerate(self.blocks):
                x_new[i] = blk(x_new[i])
            x_new = torch.cat(x_new, dim=2)
            x = self.sumChannelConv(x_new)
        else:
            x = self.tsconv(x)
        x = self.projection(x)
        return x

# OK
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.1):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    """
    guess this is important finetuning step
    """
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.1):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return x

class Proj_text(nn.Sequential):
    """
    guess this is an important finetuning step
    """
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.1):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return x

class Proj_depth(nn.Sequential):
    """
    guess this is an important finetuning step
    """
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.1):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return x

# NO
class Fusion_Encoder(nn.Module):
    """
    guess this is an important finetuning step
    """
    def __init__(self, embedding_dim=3072, proj_dim=1024, drop_proj=0.1):
        super().__init__()
        self.arc = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, img_embed=None, text_embed=None, eeg_embed=None):
        if img_embed is None and text_embed is None and eeg_embed is None:
            raise ValueError("At least one of the inputs should not be None")
        if img_embed is not None:
            batchsize = img_embed.shape[0]
        if text_embed is not None:
            batchsize = text_embed.shape[0]
        if eeg_embed is not None:
            batchsize = eeg_embed.shape[0]
        output = self.arc(torch.cat((img_embed if img_embed is not None else torch.zeros(batchsize, 1024),
                                     text_embed if text_embed is not None else torch.zeros(batchsize, 1024),
                                     eeg_embed if eeg_embed is not None else torch.zeros(batchsize, 1024)
                                     ), dim=1))
        return output

# NO
class SwitchTransformersTop1Router(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self,
                 num_experts = 4,
                 expert_capacity = 200,
                 router_jitter_noise = 0.01,
                 router_ignore_padding_tokens = False,
                 router_dtype = "float32",
                 hidden_size = 250,
                 router_bias = False
                 ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.jitter_noise = router_jitter_noise
        self.ignore_padding_tokens = router_ignore_padding_tokens
        self.dtype = getattr(torch, router_dtype)
        self.classifier = nn.Linear(hidden_size, self.num_experts, bias=router_bias).to(self.dtype)

    def _compute_router_probabilities(self, hidden_states: torch.Tensor):
        r"""
        Computes router probabilities from input hidden states.

        Args:
            hidden_states (`torch.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
        Returns:
            router_probabilities (`torch.Tensor`):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`torch.Tensor`):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        # float32 is used to ensure stability. See the discussion of "selective precision" in
        # https://arxiv.org/abs/2101.03961.
        # We also store the previous dtype to cast back the output to the previous dtype
        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)

        if self.training and self.jitter_noise > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            # todo see this work or not
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # Shape: [num_groups, tokens_per_group, num_experts]
        router_logits = self.classifier(hidden_states)

        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return router_probabilities, router_logits

    def forward(self, hidden_states: torch.Tensor):
        r"""
        Generic forward function for every Router class. Each Router expects to have the same input hidden states
        (`hidden_states`) corresponding to the hidden states for each token, the `expert_capacity` corresponding to the
        number of tokens the Router will send to each expert, some Routers can send up to few tokens to each expert.

        Each Router works as the following: it expects the hidden states for each token, gets the `router_probs` and
        `router_logits` from the `router_weights`. This will assign for each token, the raw probability to be assigned
        to an expert. Then each Router class will have to define its own `_compute_routing_instructions`.

        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        Returns:
            Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`] Tuple containing the expert index, the router probs
            and the router logits. The router probabilities and logits are required to compute the loss.
        """
        router_probs, router_logits = self._compute_router_probabilities(hidden_states) # cal probs

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Mask tokens outside expert capacity. Sum over each sequence
        token_priority = torch.cumsum(expert_index, dim=-2) #
        # mask if the token routed to the expert will overflow
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits

# NO
class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

# NO
class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_labels=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, 40),
        )
    def forward(self, x):
        cls_out = self.fc(x)
        return x, cls_out

# OK
class Cogcap(nn.Module):
    """
    revise from ATM
    """
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=10, num_features=64, num_latents=1024,
                 num_blocks=1):
        super(Cogcap, self).__init__()
        # self.regionmodule = ResidualAdd(
        #     nn.Sequential(
        #         EEG_GAT(),
        #         nn.Dropout(0.1),
        #     )
        # )
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x): # todo
        # x = self.regionmodule(x)
        x = self.attention_model(x)
        x = self.subject_wise_linear[0](x) # how to deal with this
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out

# NO
class RouteModel(nn.Module):
    def __init__(self, sequence_length=250, num_subjects=10, embedding_dim=1024, proj_dim=3,
                 drop_proj=0.1):
        super(RouteModel, self).__init__()
        self.CogCap = Cogcap(num_subjects=num_subjects, sequence_length=sequence_length)
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
        self.simple_linear = nn.Linear(embedding_dim, proj_dim)

    def forward(self, x):
        x = self.CogCap(x)
        # x = self.linear(x) # todo see which is better
        x = self.simple_linear(x)
        return x

def get_model_structure(model):
    structure = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.GELU, nn.Dropout, nn.LayerNorm)):
            layer_info = {
                'Name': name,
                'Type': type(module).__name__,
                'Parameters': sum(p.numel() for p in module.parameters() if p.requires_grad)
            }
            structure.append(layer_info)
    return structure

if __name__ == '__main__':
    input = torch.randn(1, 1024).to("cuda:1")
    eeg = torch.randn(1, 63, 250).to("cuda:1")
    model = Cogcap(num_subjects=10, sequence_length=250).to("cuda:1")
    stu = get_model_structure(model)
    model_5 = ImageProjection().to("cuda:1")
    sum(p.numel() for p in Cogcap().attention_model.parameters())
    output = model(eeg)
    output = model_5(input)
    print(output.shape)