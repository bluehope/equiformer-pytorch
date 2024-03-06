# %%
import torch
from equiformer_pytorch import Equiformer
# %%


import torch
from equiformer_pytorch import Equiformer

model = Equiformer(
    num_tokens = 24,
    dim = (4, 4, 2),               # dimensions per type, ascending, length must match number of degrees (num_degrees)
    dim_head = (4, 4, 4),          # dimension per attention head
    heads = (2, 2, 2),             # number of attention heads
    num_linear_attn_heads = 0,     # number of global linear attention heads, can see all the neighbors
    num_degrees = 3,               # number of degrees
    depth = 4,                     # depth of equivariant transformer
    attend_self = True,            # attending to self or not
    reduce_dim_out = True,         # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
    l2_dist_attention = False      # set to False to try out MLP attention
).cuda()

feats = torch.randint(0, 24, (1, 128), requires_grad=False).cuda()
coors = torch.randn(1, 128, 3,requires_grad=True).cuda()
mask  = torch.ones(1, 128,requires_grad=True).bool().cuda()

coors.retain_grad()
out = model(feats, coors, mask) # (1, 128)

print(out.type0.shape, out.type1.shape)

# Get gradients of out.type0.shape by coors
energy = torch.sum( out.type0)

# retain_graph=True is needed to call backward multiple times
energy.backward(retain_graph=True)
force = coors.grad
print('force', force)
# %%

# %%
import torch
from equiformer_pytorch import Equiformer
# %%


import torch
from equiformer_pytorch import Equiformer

model_lattice = Equiformer(
    num_tokens = 24,
    dim = (4, 4, 2),               # dimensions per type, ascending, length must match number of degrees (num_degrees)
    dim_head = (4, 4, 4),          # dimension per attention head
    heads = (2, 2, 2),             # number of attention heads
    num_linear_attn_heads = 0,     # number of global linear attention heads, can see all the neighbors
    num_degrees = 3,               # number of degrees
    depth = 4,                     # depth of equivariant transformer
    attend_self = True,            # attending to self or not
    reduce_dim_out = True,         # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
    l2_dist_attention = False,      # set to False to try out MLP attention
    use_pbc=True                   # 
).cuda()

N = 5 # 128
feats = torch.randint(0, 24, (1, N)).cuda()
coors = 10 * torch.randn(1, N, 3, requires_grad=True).cuda()
coors.retain_grad()
mask  = torch.ones(1, N).bool().cuda()

lattice = torch.tensor([[[10.0, 0, 0], [0, 10.0, 0], [0, 1, 10.0]]]).float().cuda()
out = model_lattice(feats, coors, mask, lattice_vectors=lattice) # (1, 128)

print(out.type0.shape, out.type1.shape)

# Get gradients of out.type0.shape by coors
energy = torch.sum( out.type0)

energy.backward()
force = coors.grad
print('force', force)


# %%
from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange
import ase
import ase.atoms
import numpy as np

lattice_vectors = torch.tensor([[[10.0, 0, 0], [10, 10.0, 0], [0, -10, 10.0]]]).float().cuda()

atoms = ase.Atoms(positions=coors[0].cpu().detach().numpy(), symbols=feats[0].cpu().detach().numpy(),
                  pbc=True, cell=lattice_vectors[0].cpu().detach().numpy())

position_frac = torch.matmul(coors, torch.linalg.inv(lattice_vectors))
position_cart = torch.matmul(position_frac, lattice_vectors)

max_diff_frac = np.max(np.abs(position_frac[0].cpu().detach().numpy() - atoms.get_scaled_positions()))

# cal to diff between coors and position_cart
max_diff = torch.max(torch.abs(coors - position_cart))
print("max_diff", max_diff.item(), max_diff_frac)

rel_pos  = rearrange(coors, 'b n d -> b n 1 d') - rearrange(coors, 'b n d -> b 1 n d')
frac_rel_pos = torch.matmul(rel_pos, torch.linalg.inv(lattice_vectors))

rel_pos_pbc  = torch.matmul(frac_rel_pos - torch.round(frac_rel_pos), lattice_vectors)
print("ref pos orig", rel_pos)
print("ref pos pbc", rel_pos_pbc)

# %%
