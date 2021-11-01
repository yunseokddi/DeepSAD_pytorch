import torch


class DeepSAD(object):
    def __init__(self, eta: float = 1.0) -> None:
        self.eta = eta
        self.c = None
        self.net_name = None
        self.net = None

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None
        self.ae_trainer= None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }

    def set_network(self, net_name) -> None:
        self.net_name = net_name
