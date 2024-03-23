import torch
from torch import nn

import todos
import pdb

BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]])

def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    # tensor [xyz] size: [65536, 3], min: -2.342431, max: 2.053263, mean: 0.015732

    box_min, box_max = bounding_box
    box_min = box_min.to(xyz.device)
    box_max = box_max.to(xyz.device)
    
    keep_mask = xyz==torch.max(torch.min(xyz, box_max), box_min)
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution
    
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0]).to(xyz.device)*grid_size

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS.to(xyz.device)
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)
    # tensor [voxel_indices] size: [65536, 8, 3], min: 3.0, max: 13.0, mean: 8.016531
    # tensor [hashed_voxel_indices] size: [65536, 8], min: 24676.0, max: 517691.0, mean: 279805.5

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices #, keep_mask


class HashEmbedder(nn.Module):
    '''
    "encoding": {
        "otype": "HashGrid",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 1.44
    },
    '''
    def __init__(self, 
            # bounding_box=(
            #     torch.tensor([-3.972000, -4.003262, -3.328434]), 
            #     torch.tensor([4.012635, 4.006973, 3.338517])
            # ), 
            bounding_box=(
                torch.tensor([-3.0, -3.0, -3.0]), 
                torch.tensor([ 3.0,  3.0,  3.0])
            ), 
            n_levels=16, 
            n_features_per_level=2,
            log2_hashmap_size=19, 
            base_resolution=16, 
            finest_resolution=512,
        ):
        super().__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.b = torch.exp((torch.log(self.finest_resolution) - torch.log(self.base_resolution))/(n_levels-1))
        self.n_output_dims = n_levels * n_features_per_level

        self.embeddings = nn.ModuleList([
            nn.Embedding(2**self.log2_hashmap_size, n_features_per_level) for i in range(n_levels)])
        # (Pdb) self.embeddings
        # ModuleList(
        #   (0-15): 16 x Embedding(524288, 2)
        # )

        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)


    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    def forward(self, x):
        # x is 3D point position: B x 3
        # tensor [x] size: [65536, 3], min: -2.342431, max: 2.053263, mean: 0.015732

        x_embedded_all = []
        for i in range(self.n_levels): # self.n_levels -- 16
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = get_voxel_vertices(
                    x, self.bounding_box, resolution, self.log2_hashmap_size)
            
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        out = torch.cat(x_embedded_all, dim=-1) #, keep_mask
        # tensor [item] size: [65536, 32], min: -9.8e-05, max: 9.8e-05, mean: 0.0

        return out


if __name__=="__main__":
    model = HashEmbedder()
    model.eval()
    print("model ...", model)

    x = torch.randn(819200, 3)
    with torch.no_grad():
        y = model(x)

    todos.debug.output_var("x", x)
    todos.debug.output_var("y", y) # [921600, 32]
 