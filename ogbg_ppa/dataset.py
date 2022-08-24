from typing import Callable, Optional

from torch_geometric.data import InMemoryDataset
from hfai.datasets import OGB


class OGBGDataset(InMemoryDataset):

    def __init__(self, data_name, transform: Optional[Callable] = None) -> None:
        super(OGBGDataset, self).__init__(transform=transform)

        dataset = OGB(data_name=data_name)

        self.data, self.slices = dataset.get_data()
        self.split = dataset.get_split()
