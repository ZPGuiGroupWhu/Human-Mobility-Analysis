import torch
import torch.nn as nn


class Input_Module(nn.Module):
    """
    Embedding and concatenating the spatiotemporal semantics， trajectory points, and  driving status
    """
    embed_dim = [("weekday", 7, 3), ("start_time", 48, 6)]

    def __init__(self):
        super(Input_Module, self).__init__()
        self.sem_pt_embed = nn.Embedding(9, 3, padding_idx=0)
        self.map = nn.Sequential(
            nn.Linear(8, 3, False)
        )
        for name, num_embeddings, embedding_dim in Input_Module.embed_dim:
            self.add_module(name + '_embed', nn.Embedding(num_embeddings, embedding_dim))

    def end_dim(self):
        end_dim = 0
        for name, num_embeddings, embedding_dim in Input_Module.embed_dim:
            end_dim += embedding_dim
        end_dim += 5 + 3 + 3  # trajectory dim + departure region dim + trajectory point semantics dim
        return end_dim

    def sem_dim(self):
        # departure spatiotemporal semantics dim -> departure region dim + weekday dim + start_time dim
        return 3 + 6 + 3

    def forward(self, attr, traj):
        lngs = traj['lngs'].unsqueeze(-1)
        lats = traj['lats'].unsqueeze(-1)
        locs = torch.cat((lngs, lats), dim=2)
        time_semantic = []
        for name, num_embeddings, embedding_dim in Input_Module.embed_dim:
            embed = getattr(self, name + '_embed')

            _attr = torch.squeeze(embed(attr[name].view(-1, 1)), dim=1)
            time_semantic.append(_attr)

        time_semantic.append(self.map(attr['sem_O']))
        traj_semantic = torch.cat(time_semantic, dim=1)

        expand_traj_semantic = traj_semantic.unsqueeze(dim=1).expand(
            locs.size()[:2] + (traj_semantic.size()[-1],))

        sem_pt = torch.squeeze(self.sem_pt_embed(traj['sem_pt'].unsqueeze(-1)), dim=2)

        semantic = torch.cat((expand_traj_semantic, sem_pt), dim=2)

        input_tensor = torch.cat((locs,
                                  traj['travel_dis'].unsqueeze(-1),
                                  traj['spd'].unsqueeze(-1),
                                  traj['azimuth'].unsqueeze(-1),
                                  semantic), dim=2)

        return input_tensor, traj, traj_semantic