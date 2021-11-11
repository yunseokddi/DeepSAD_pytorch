from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import logging
import time
import torch
import torch.optim as optim
import numpy as np


class DeepSADTrainer(BaseTrainer):
    def __init__(self, c, eta: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        self.eps = 1e-6

        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        net = net.to(self.device)

        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            print('Center c initialized.')

        print('Starting training...')
        start_time = time.time()
        net.train()

        for epoch in range(self.n_epochs):
            tq = tqdm(train_loader, total=len(train_loader))
            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0

            for data in tq:
                inputs, _, semi_targets, _ = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                optimizer.zero_grad()

                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                errors = {
                    'epoch': epoch,
                    'train loss': epoch_loss / n_batches
                }

                tq.set_postfix(errors)

                total_avg_loss = epoch_loss / n_batches

        self.train_time = time.time() - start_time
        print('Total pretraining avg loss: {:.5f}'.format(total_avg_loss))
        print('Training Time: {:.3f}s'.format(self.train_time))
        print('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        net = net.to(self.device)
        net.eval()

        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []

        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets==0, dist, self.eta *((dist + self.eps)** semi_targets.float()))
                loss = torch.mean(losses)
                scores = dist

                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        print('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        print('Test Time: {:.3f}s'.format(self.test_time))
        print('Finished testing.')



    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()

        with torch.no_grad():
            for data in train_loader:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
