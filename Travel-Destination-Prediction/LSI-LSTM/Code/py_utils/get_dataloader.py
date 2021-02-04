from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import numpy as np


def get_dataloader(data, batch_size):
    dataset = SuperDataset(data=data)
    batch_sampler = BatchSampler(dataset, batch_size)
    data_loader = DataLoader(dataset=dataset,
                             collate_fn=lambda x: collate_fn(x),
                             batch_sampler=batch_sampler,
                             pin_memory=False)
    return data_loader


class SuperDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.lengths = list(map(lambda x: len(x["lngs"]), self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        super(Sampler, self).__init__()
        self.lengths = dataset.lengths
        self.data_len = len(dataset)
        self.batch_size = batch_size
        self.indices = list(range(self.data_len))

        # turn the fn to generator
        """
            divide the data into chunks in size of batch_size*10
            every chunk has sorted by data_length to make training stability
            data: all the traj_dict(include raw traj_dict and cut_traj_dict)
        """

    def __iter__(self):
        # shuffle the data
        np.random.shuffle(self.indices)
        chunk_size = self.batch_size * 10
        chunks = (self.data_len + chunk_size - 1) // chunk_size
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size:(i + 1) * chunk_size]
            partial_indices.sort(key=lambda x: self.lengths[x], reverse=True)
            self.indices[i * chunk_size:(i + 1) * chunk_size] = partial_indices
        self.batches = (self.data_len + self.batch_size - 1) // self.batch_size
        for i in range(self.batches):
            yield self.indices[i * self.batch_size:(i + 1) * self.batch_size]

    def __len__(self):
        return self.batches


def collate_fn(batch_data):

    traj_key = ["lngs", "lats", "travel_dis", "spd", "azimuth", "sem_pt"]
    info_key = ["weekday", "start_time"]
    attr_key = ["cur_pt", "destination", "dis_total", "norm_dict", "sem_O"]
    attr = {}
    traj = {}

    for key in attr_key:
        attr[key] = torch.Tensor([item[key] for item in batch_data])

    for key in info_key:
        attr[key] = torch.LongTensor([item[key] for item in batch_data])

    for key in traj_key:
        seqs = np.asarray([item[key] for item in batch_data])
        padded_seqs = numpy_fillna(seqs)
        if key == "sem_pt":
            traj[key] = torch.from_numpy(padded_seqs).long()
        else:
            traj[key] = torch.from_numpy(padded_seqs).float()

    lens = [len(item["lngs"]) for item in batch_data]
    traj["lens"] = torch.LongTensor(lens)

    return attr, traj


# normalize data by fill 0
def numpy_fillna(data):
    lens = np.asarray([len(item) for item in data])
    mask = np.arange(lens.max()) < lens[:, None]
    out = np.zeros(mask.shape, dtype=np.float32)
    out[mask] = np.concatenate(data)
    return out