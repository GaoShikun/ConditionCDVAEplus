"""This module is adapted from https://github.com/Open-Catalyst-Project/ocp/tree/master/ocpmodels/models
"""

import torch
import torch.nn as nn
from fairchem.core.common.utils import conditional_grad
from torch_scatter import scatter

from ccdvaeplus.pl_modules.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20, _AVG_NUM_NODES
from ccdvaeplus.pl_modules.equiformer_v2.so3 import SO3_Embedding

try:
    from torch_geometric.nn.acts import swish
except ImportError:
    from torch_geometric.nn.resolver import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer,
    EmbeddingBlock,
    ResidualLayer,
    SphericalBasisLayer,
)
from torch_sparse import SparseTensor

from ccdvaeplus.common.data_utils import (
    get_pbc_distances,
    frac_to_cart_coords,
    radius_graph_pbc_wrapper,
)

try:
    import sympy as sym
except ImportError:
    sym = None


class InteractionPPBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act=swish,
    ):
        super(InteractionPPBlock, self).__init__()
        self.act = act

        # Transformations of Bessel and spherical basis representations.
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(
            num_spherical * num_radial, basis_emb_size, bias=False
        )
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)

        # Dense transformations of input messages.
        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets.
        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection.
        self.layers_before_skip = torch.nn.ModuleList(
            [
                ResidualLayer(hidden_channels, act)
                for _ in range(num_before_skip)
            ]
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [
                ResidualLayer(hidden_channels, act)
                for _ in range(num_after_skip)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        # Initial transformations.
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis.
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down-project embeddings and generate interaction triplet embeddings.
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis.
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings.
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputPPBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_layers,
        act=swish,
    ):
        super(OutputPPBlock, self).__init__()
        self.act = act

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def forward(self, x, rbf, i, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)

class AtomLSTMEmbedding(nn.Module):
    def __init__(self, vocab_size=None, embedding_dim=None, hidden_dim=None, embedding_max_length=None):
        super(AtomLSTMEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_max_length)
        self.act = swish

    def forward(self, x):
        x = self.embedding(x.long())
        lstm_out, (h_n, c_n) = self.lstm(x)
        x= self.fc(h_n.permute(1, 0, 2))
        return self.act(x)


class DimeNetPlusPlus(torch.nn.Module):
    r"""DimeNet++ implementation based on https://github.com/klicperajo/dimenet.
    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Embedding size used for interaction triplets
        basis_emb_size (int): Embedding size used in the basis transformation
        out_emb_channels(int): Embedding size used for atoms in the output block
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (function, optional): The activation funtion.
            (default: :obj:`swish`)
    """

    url = "https://github.com/klicperajo/dimenet/raw/master/pretrained"

    def __init__(
        self,
        hidden_channels,
        out_channels,
        num_blocks,
        int_emb_size,
        basis_emb_size,
        out_emb_channels,
        num_spherical,
        num_radial,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=swish,
    ):
        super(DimeNetPlusPlus, self).__init__()

        self.cutoff = cutoff

        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, cutoff, envelope_exponent
        )

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = torch.nn.ModuleList(
            [
                OutputPPBlock(
                    num_radial,
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionPPBlock(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def triplets(self, edge_index, num_nodes):
        row, col = edge_index  # j, i

        # row, col = col, row  # Swap because my definition of edge_index is i->j

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]  #mask 用于过滤掉那些 i和k相同的情况,避免三元组的起点和终点是同一个节点，相当于产生了一个自循环或环形路径。

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, z, pos, batch=None):
        """"""
        raise NotImplementedError


class DimeNetPlusPlusWrap(DimeNetPlusPlus):
    def __init__(
        self,
        num_targets,
        hidden_channels=128,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        otf_graph=False,
        cutoff=10.0,
        max_num_neighbors=20,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        readout='mean',
    ):
        self.num_targets = num_targets
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.otf_graph = otf_graph

        self.readout = readout

        super(DimeNetPlusPlusWrap, self).__init__(
            hidden_channels=hidden_channels,
            out_channels=num_targets,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
        )

    def forward(self, data):
        batch = data.batch

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc_wrapper(
                data, self.cutoff, self.max_num_neighbors, data.num_atoms.device
            )
            data.edge_index = edge_index
            data.to_jimages = cell_offsets
            data.num_bonds = neighbors

        pos = frac_to_cart_coords(
            data.frac_coords,
            data.lengths,
            data.angles,
            data.num_atoms)

        out = get_pbc_distances(
            data.frac_coords,
            data.edge_index,
            data.lengths,
            data.angles,
            data.to_jimages,
            data.num_atoms,
            data.num_bonds,
            return_offsets=True
        )

        edge_index = out["edge_index"]
        dist = out["distances"]
        offsets = out["offsets"]

        j, i = edge_index

        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=data.atom_types.size(0)
        )

        # Calculate angles.
        pos_i = pos[idx_i].detach()
        pos_j = pos[idx_j].detach()
        pos_ji, pos_kj = (
            pos[idx_j].detach() - pos_i + offsets[idx_ji],
            pos[idx_k].detach() - pos_j + offsets[idx_kj],
        )

        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)    #RBF径向基函数  捕捉节点之间的距离信息  (Nrbf，6)
        sbf = self.sbf(dist, angle, idx_kj)     #SBF球面基函数   用于捕捉节点之间的角度信息

        # Embedding block.
        x = self.emb(data.atom_types.long(), rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i, num_nodes=pos.size(0))

        # Use mean
        if batch is None:
            if self.readout == 'mean':
                energy = P.mean(dim=0)
            elif self.readout == 'sum':
                energy = P.sum(dim=0)
            elif self.readout == 'cat':
                import pdb
                pdb.set_trace()
                energy = torch.cat([P.sum(dim=0), P.mean(dim=0)])
            else:
                raise NotImplementedError
        else:
            # TODO: if want to use cat, need two lines here
            energy = scatter(P, batch, dim=0, reduce=self.readout)

        return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class EquiformerV2Encoder(EquiformerV2_OC20):
    def __init__(
        self,
        num_targets=None,
        otf_graph=False,
        cutoff=10.0,
        max_num_neighbors=20,
        readout='mean',
    ):
        self.num_targets = num_targets
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.otf_graph = otf_graph

        self.readout = readout

        super(EquiformerV2Encoder, self).__init__(
            num_atoms=None,  # not used
            bond_feat_dim=None,  # not used
            num_targets=None,
            use_pbc=True,
            regress_forces=False,
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
            energy_block_out_channels=256,

            norm_type='layer_norm_sh',

            lmax_list=[4],
            mmax_list=[2],
            grid_resolution=18,

            num_sphere_samples=128,

            edge_channels=128,
            decoder=False,
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


    @conditional_grad(torch.enable_grad())
    def forward(self, data, z=None):
        self.batch_size = len(data.num_atoms)
        self.dtype = data.frac_coords.dtype
        self.device = data.frac_coords.device

        atomic_numbers = data.atom_types.long()
        num_atoms = len(atomic_numbers)

        if z is not None:
            z = z.repeat_interleave(data.num_atoms[:, ], dim=0)
        pos = frac_to_cart_coords(
            data.frac_coords,
            data.lengths,
            data.angles,
            data.num_atoms)

        out = get_pbc_distances(
            data.frac_coords,
            data.edge_index,
            data.lengths,
            data.angles,
            data.to_jimages,
            data.num_atoms,
            data.num_bonds,
            return_offsets=True,
            return_distance_vec=True
        )

        edge_index = out["edge_index"]
        dist = out["distances"]
        offsets = out["offsets"]
        edge_distance_vec = out["distance_vec"]

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                )[:, offset: offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(dist)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            if self.decoder:
                edge_distance = torch.cat(
                    (edge_distance, z[edge_index[0]], z[edge_index[1]], source_embedding, target_embedding), dim=1)
            else:
                edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index,
        )
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch,  # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        ###############################################################
        # node_feature estimation
        ###############################################################



        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x)
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        energy = torch.zeros((len(data.num_atoms), node_energy.size(-1)), device=node_energy.device,
                             dtype=node_energy.dtype)
        energy.index_add_(0, data.batch, node_energy.squeeze(1))
        energy = energy / _AVG_NUM_NODES

        ###############################################################
        # Force estimation
        ###############################################################
        if self.regress_forces:
            forces = self.force_block(x,
                                      atomic_numbers,
                                      edge_distance,
                                      edge_index,
                                      )
            forces = forces.embedding.narrow(1, 1, 3)
            forces = forces.view(-1, 3)

        if not self.regress_forces:
            return energy
        else:
            return energy, forces