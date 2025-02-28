import math

import torch
import torch.nn as nn
from e3nn.o3 import Linear
from torch_geometric.data import Data

from ccdvaeplus.pl_modules.embeddings import MAX_ATOMIC_NUM
from ccdvaeplus.pl_modules.equiformer_v2.equiformer_v2 import EquiformerV2
from ccdvaeplus.pl_modules.gemnet.gemnet import GemNetT


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)



class GemNetTDecoder(nn.Module):
    """Decoder with GemNetT."""

    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        max_neighbors=20,
        radius=6.,
        scale_file=None,
    ):
        super(GemNetTDecoder, self).__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors

        self.gemnet = GemNetT(
            num_targets=1,
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            regress_forces=True,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            otf_graph=True,
            scale_file=scale_file,
        )
        self.fc_atom = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

    def forward(self, z, pred_frac_coords, pred_atom_types, num_atoms,
                lengths, angles):
        """
        args:
            z: (N_cryst, num_latent)
            pred_frac_coords: (N_atoms, 3)
            pred_atom_types: (N_atoms, ), need to use atomic number e.g. H = 1
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3)
            angles: (N_cryst, 3)
        returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """
        # (num_atoms, hidden_dim) (num_crysts, 3)
        h, pred_cart_coord_diff = self.gemnet(
            z=z,
            frac_coords=pred_frac_coords,
            atom_types=pred_atom_types,
            num_atoms=num_atoms,
            lengths=lengths,
            angles=angles,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
        )
        pred_atom_types = self.fc_atom(h)
        return pred_cart_coord_diff, pred_atom_types


class EquiformerV2Decoder(nn.Module):
    def __init__(
        self,
            max_num_neighbors=20,
            radius=7.0,
    ):

        super(EquiformerV2Decoder, self).__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_num_neighbors
        self.EquiformerV2 = EquiformerV2(
            num_atoms=None,  # not used
            bond_feat_dim=None,  # not used
            num_targets=None,
            use_pbc=True,
            regress_forces=True,
            otf_graph=True,
            max_neighbors=20,
            max_radius=7.0,
            max_num_elements=100,

            num_layers=8,
            sphere_channels=128,
            attn_hidden_channels=64,
            num_heads=8,
            attn_alpha_channels=64,
            attn_value_channels=16,
            ffn_hidden_channels=128,
            energy_block_out_channels=128,

            norm_type='layer_norm_sh',

            lmax_list=[4],
            mmax_list=[2],
            grid_resolution=18,

            num_sphere_samples=128,

            edge_channels=128,
            decoder=True,
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            use_m_share_rad=False,
            distance_function="gaussian",
            num_distance_basis=512,

            attn_activation='silu',
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation='silu',
            use_gate_act=False,
            use_grid_mlp=True,
            use_sep_s2_act=True,

            alpha_drop=0.1,
            drop_path_rate=0.1,
            proj_drop=0.0,

            weight_init='uniform'
        )
        self.fc_atom = Linear(irreps_in='128x0e',
                              irreps_out='{}x0e'.format(MAX_ATOMIC_NUM))

    def forward(self,z, noisy_frac_coords, rand_atom_types, num_atoms, pred_lengths, pred_angles):
        data = Data(
            z = z,
            frac_coords = noisy_frac_coords,
            atom_types = rand_atom_types,
            num_atoms = num_atoms,
            lengths = pred_lengths,
            angles = pred_angles
        )

        node_feature, pred_cart_coord_diff = self. EquiformerV2(data)

        pred_atom_types = self.fc_atom(node_feature)
        return pred_cart_coord_diff, pred_atom_types




