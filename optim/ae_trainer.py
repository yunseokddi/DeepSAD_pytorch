from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class AETrainer(BaseTrainer):
    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: list = [30, 80],
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        self.train_time = None
        self.test_auc = None
        self.test_time = None

    def train(self, dataset: BaseADDataset, ae_net: BaseNet, writer):
        global total_avg_loss
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        criterion = nn.MSELoss(reduction='none')

        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        print('Starting pretraining...')

        start_time = time.time()
        ae_net.train()

        for epoch in range(self.n_epochs):
            scheduler.step()

            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is {}'.format(float(scheduler.get_lr()[0])))

            epoch_loss = 0.0
            n_batches = 0

            tq = tqdm(train_loader, total=len(train_loader))

            for data in tq:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)

                optimizer.zero_grad()

                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                loss = torch.mean(rec_loss)
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

            writer.add_scalar("AE/Loss", total_avg_loss, epoch)

        self.train_time = time.time() - start_time

        print('Total pretraining avg loss: {:.5f}'.format(total_avg_loss))
        print('Pretraining Time: {:.3f}s'.format(self.train_time))
        print('Finished pretraining')

        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet):
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        criterion = nn.MSELoss(reduction='none')

        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()

        with torch.no_grad():
            tq = tqdm(test_loader, total=len(test_loader))
            for data in tq:
                inputs, labels, _, idx = data
                inputs, labels, idx = inputs.to(self.device), labels.to(self.device), idx.to(self.device)

                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                scores = torch.mean(rec_loss, dim=tuple(range(1, rec.dim())))

                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss = torch.mean(rec_loss)
                epoch_loss += loss.item()
                n_batches += 1

                errors = {
                    'test loss': epoch_loss / n_batches
                }

            tq.set_postfix(errors)

        self.test_time = time.time() - start_time

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        print('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        print('Test Time: {:.3f}s'.format(self.test_time))
        print('Finished testing autoencoder.')
