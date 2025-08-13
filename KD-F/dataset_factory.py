import torch
from torch.utils.data import Sampler

def prepare_grouped_datasets(dataset, sensitive_attr):
    groups = {}
    for idx, item in enumerate(dataset):
        gid = item['group'].item() if torch.is_tensor(item['group']) else item['group']
        groups.setdefault(gid, []).append(idx)
    
    print(f'Found {len(groups)} demographic groups')
    print(f'Total samples: {len(dataset)}')
    for gid, indices in groups.items():
        print(f'Group {gid}: {len(indices)} samples')
    return groups

class GroupBalancedSampler(Sampler):
    def __init__(self, dataset, sensitive_attr, batch_size, groups=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.groups = groups if groups is not None else prepare_grouped_datasets(dataset, sensitive_attr)
        self.group_ids = list(self.groups.keys())
        self.cursors = {g: 0 for g in self.group_ids}
        self.max_len = max(len(idxs) for idxs in self.groups.values())

    def __len__(self):
        return self.max_len * len(self.group_ids)

    def __iter__(self):
        batch = []
        for i in range(self.__len__()):
            gid = self.group_ids[i % len(self.group_ids)]
            idx_list = self.groups[gid]
            cursor = self.cursors[gid] % len(idx_list)
            batch.append(idx_list[cursor])
            self.cursors[gid] += 1
            
            if len(batch) == self.batch_size:
                yield from batch
                batch = []
        if batch:
            yield from batch
