from abc import ABC, abstractmethod
from .base_dataset import BaseADDataset
from .base_net import BaseNet


class BaseTrainer(ABC):
    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: list, batch_size: int,
                 weight_decay: float, device: str, n_jobs_dataloader: int):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

    @abstractmethod
    def train(self, dataset: BaseADDataset, net: BaseNet) -> BaseNet:
        pass

    @abstractmethod
    def test(self, dataset: BaseADDataset, net: BaseNet):
        pass
