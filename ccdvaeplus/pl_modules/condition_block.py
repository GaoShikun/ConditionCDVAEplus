from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal_, orthogonal
from torch_scatter import scatter

from ccdvaeplus.pl_modules.decoder import build_mlp
from ccdvaeplus.pl_modules.gemnet.layers.embedding_block import AtomEmbedding


class GaussianExpansion(nn.Module):
    r"""Expansion layer using a set of Gaussian functions.

    https://github.com/atomistic-machine-learning/cG-SchNet/blob/53d73830f9fb1158296f060c2f82be375e2bb7f9/nn_classes.py#L687)

    Args:
        start (float): center of first Gaussian function, :math:`\mu_0`.
        stop (float): center of last Gaussian function, :math:`\mu_{N_g}`.
        n_gaussians (int, optional): total number of Gaussian functions, :math:`N_g`
            (default: 50).
        trainable (bool, optional): if True, widths and offset of Gaussian functions
            are adjusted during training process (default: False).
        widths (float, optional): width value of Gaussian functions (provide None to
            set the width to the distance between two centers :math:`\mu`, default:
            None).
    """

    def __init__(self, start, stop, n_gaussians=50, trainable=False, width=None):
        super(GaussianExpansion, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        if width is None:
            widths = (offset[1] - offset[0]) * torch.ones_like(offset)
        else:
            widths = width * torch.ones_like(offset)
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, property):
        """Compute expanded gaussian property values.
        Args:
            property (torch.Tensor): property values of (N_b x 1) shape.
        Returns:
            torch.Tensor: layer output of (N_b x N_g) shape.
        """
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(self.widths, 2)[None, :]
        # Use advanced indexing to compute the individual components
        diff = property - self.offsets[None, :]
        # compute expanded property values
        return torch.exp(coeff * torch.pow(diff, 2))


class comp_embedding(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.comp_dim = n_out
        self.emb = AtomEmbedding(self.comp_dim)

    def forward(self, atom_types, num_atoms):
        batch = torch.repeat_interleave(
            torch.arange(num_atoms.size(0), device=num_atoms.device),
            num_atoms,
        )

        atom_emb = self.emb(atom_types)
        comp_emb = scatter(atom_emb, batch, dim=0, reduce="mean")
        return comp_emb


# single scalar, one sample is (1,)
class ScalarEmbedding(nn.Module):
    def __init__(
        self,
        prop_name: str,
        # batch norm
        batch_norm: bool = False,
        # gaussian expansion
        no_expansion: bool = False,
        n_basis: int = None,  # num gaussian basis
        start: float = None,
        stop: float = None,
        trainable_gaussians: bool = False,
        width: float = None,
        # out mlp
        no_mlp: bool = False,
        hidden_dim: int = None,
        fc_num_layers: int = None,
        n_out: int = None,
    ):
        super().__init__()
        self.n_out = n_out
        self.prop_name = prop_name

        if batch_norm:
            self.bn = nn.BatchNorm1d(1)
        else:
            self.bn = nn.Identity()

        if no_expansion:
            self.expansion_net = nn.Identity()
        else:
            self.expansion_net = GaussianExpansion(
                start, stop, n_basis, trainable_gaussians, width
            )

        if no_mlp:
            self.mlp = nn.Identity()
        else:
            self.mlp = build_mlp(None, hidden_dim, fc_num_layers, n_out)

    def forward(self, prop):
        prop = self.bn(prop)
        prop = self.expansion_net(prop)  # expanded prop
        out = self.mlp(prop)
        return out



def multply(*args):
    """Returns the product of the given arguments."""
    total = 1
    for arg in args:
        total *= arg
    return total

class LMFConditionEmbedding(nn.Module):
    '''
    Low-rank Multimodal Fusion
    https://github.com/Justin1904/Low-rank-Multimodal-Fusion

    '''

    def __init__(self, input_dims, dropouts, output_dim, rank=4, condition_prop=None, condition_formula=True):
        '''
        Args:

        Output:
        '''
        super(LMFConditionEmbedding, self).__init__()


        self.output_dim = output_dim
        self.rank = rank

        self.z_dim = input_dims[0]

        self.z_prob = dropouts[0]
        self.ef_prob = dropouts[1]
        self.comp_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        # define the pre-fusion subnetworks
        if condition_prop:
            self.emb_dim = input_dims[1]
            self.label_embedding = ScalarEmbedding(
                prop_name='formation_energy_per_atom',
                batch_norm=False,
                no_expansion=False,
                n_basis=50,
                start=-2,
                stop=2,
                trainable_gaussians=False,
                no_mlp=True,
            )
            self.emb_factor = Parameter(torch.Tensor(self.rank, self.emb_dim + 1, self.output_dim))

        if condition_formula:
            self.emb_dim = input_dims[2]
            self.label_embedding  = comp_embedding(self.emb_dim)
            self.emb_factor = Parameter(torch.Tensor(self.rank, self.emb_dim + 1, self.output_dim))

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.z_factor = Parameter(torch.Tensor(self.rank, self.z_dim + 1, self.output_dim))

        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        self.output_layer = nn.Sequential(nn.Linear(self.output_dim, self.output_dim),
                                          nn.Tanh())

        # init teh factors
        xavier_normal_(self.z_factor)
        xavier_normal_(self.emb_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
        # nn.init.xavier_normal_(self.output_layer[0].weight)
        # nn.init.zeros_(self.output_layer[0].bias)

    def forward(self, z, fe_c=None, atom_type=None, num_atoms=None):
        '''
        Args:
            z: tensor of shape (batch_size, z_dim)
            fe_c: tensor of shape (batch_size, 1)
            atom_type: tensor of shape (batch_size*num_atoms, 1)
            num_atoms: tensor of shape (batch_size, 1)
        '''
        batch_size = z.size(0)
        if fe_c is not None:
            condition_emb = self.label_embedding(fe_c)
        if atom_type is not None and num_atoms is not None:
            condition_emb = self.label_embedding(atom_type, num_atoms)

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product

        z_h = torch.cat((Variable(torch.ones(size=(batch_size, 1), device=z.device), requires_grad=False), z),
                               dim=1)
        condition_emb_h = torch.cat((Variable(torch.ones(size=(batch_size, 1), device=z.device), requires_grad=False), condition_emb),
                               dim=1)
        # comp_h = torch.cat((Variable(torch.ones(size=(batch_size, 1), device=z.device), requires_grad=False), composition),
        #                        dim=1)

        fusion_z = torch.matmul(z_h, self.z_factor)
        fusion_emb_c = torch.matmul(condition_emb_h, self.emb_factor)
        # fusion_composition = torch.matmul(comp_h, self.comp_factor)
        fusion_zy = fusion_z * fusion_emb_c

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)

        return output


#test
if __name__ == "__main__":
    z = torch.randn(16,256).cuda()
    fe_c = torch.randn(16, 1).cuda()
    atom_type = torch.randint(1, 101, (16 * 8,)).cuda()
    tensor_of_sixes = torch.full((16,), 8).cuda()

    # TFN = TFNConditionEmbedding(
    #     z_dim=256, post_fusion_dim=1024, condition_dim=50, composition_dim=128
    # ).cuda()
    LMF = LMFConditionEmbedding(input_dims=[256, 50, 128],
                                dropouts=[0, 0.2, 0.3, 0.0003],
                                output_dim=256,
                                rank=4).cuda()

    # out = TFN(z, fe_c, atom_type, tensor_of_sixes)
    out2 = LMF(z=z, atom_type=atom_type,num_atoms=tensor_of_sixes)



    print()