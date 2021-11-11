from dataset.main import load_dataset
from DeepSAD import DeepSAD
from utils.plot_images_grid import plot_images_grid
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
import torch
import random

dataset_name = 'cifar10'
parser = argparse.ArgumentParser(description='Train Deep SAD model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

################################################################################
# net category
# 'mnist_LeNet', 'fmnist_LeNet', 'cifar10_LeNet', 'arrhythmia_mlp',
# 'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
# 'thyroid_mlp'
################################################################################
parser.add_argument('--net_name', '-net', type=str, default='cifar10_LeNet', help='Enter backbone networks')
parser.add_argument('--xp_path', '-xp', type=str, default='./experiment/', help='xp log path')
parser.add_argument('--data_path', '-data', type=str, default='./data/', help='Dataset path')
parser.add_argument('--load_model', '-load', type=bool, default=None, help='load pretrained weight')
parser.add_argument('--device', '-device', type=str, default='cuda',
                    help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
parser.add_argument('--seed', '-seed', type=int, default=1, help='Set seed. If -1, use randomization.')
parser.add_argument('--pretrain', '-pretrain', type=bool, default=True,
                    help='Pretrain neural networks parameters via autoencoder.')

################################################################################
# Dataset settings
################################################################################
parser.add_argument('--num_threads', '-num_threads', type=int, default=4,
                    help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
parser.add_argument('--n_jobs_dataloader', '-n_jobs_dataloader', type=int, default=4,
                    help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
parser.add_argument('--normal_class', '-normal_class', type=int, default=0,
                    help='Specify the normal class of the dataset (all other classes are considered anomalous).')
parser.add_argument('--known_outlier_class', '-known_outlier_class', type=int, default=1,
                    help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
parser.add_argument('--n_known_outlier_classes', '-n_known_outlier_classes', type=int, default=1,
                    help='Number of known outlier classes.'
                         'If 0, no anomalies are known.'
                         'If 1, outlier class as specified in --known_outlier_class option.'
                         'If > 1, the specified number of outlier classes will be sampled at random.')

################################################################################
# DeepSAD settings
################################################################################
parser.add_argument('--n_epochs', '-e', type=int, default=100, help='Num of epochs to train')
parser.add_argument('--eta', '-eta', type=float, default=1.0, help='Deep SAD hyperparameter eta (must be 0 < eta).')
parser.add_argument('--ratio_known_normal', '-normal_ratio', type=float, default=0.0,
                    help='Ratio of known (labeled) normal training examples.')
parser.add_argument('--ratio_known_outlier', '-outlier_ratio', type=float, default=0.01,
                    help='Ratio of known (labeled) anomalous training examples.')
parser.add_argument('--ratio_pollution', '-ratio_pollution', type=float, default=0.1,
                    help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
parser.add_argument('--lr', '-lr', type=float, default=1e-3,
                    help='Initial learning rate for Deep SAD networks training. Default=0.001')
parser.add_argument('--lr_milestone', '-lr_milestone', type=int, default=50,
                    help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='Batch size for mini-batch training.')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-6,
                    help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')
parser.add_argument('--optimizer_name', '-optimizer', type=str, default='adam',
                    help='Name of the optimizer to use for Deep SAD networks training.')

################################################################################
# AUTOENCODER settings
################################################################################
parser.add_argument('--ae_optimizer_name', '-ae_optimizer', type=str, default='adam',
                    help='Name of the optimizer to use for autoencoder pretraining.')
parser.add_argument('--ae_lr', '-ae_lr', type=float, default=1e-3,
                    help='Initial learning rate for autoencoder pretraining. Default=0.001')
parser.add_argument('--ae_n_epochs', '-ae_n_epochs', type=int, default=100,
                    help='Number of epochs to train autoencoder.')
parser.add_argument('--ae_lr_milestone', '-ae_lr_milestone', type=list, default=0,
                    help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
parser.add_argument('--ae_batch_size', '-ae_bs', type=int, default=128,
                    help='Batch size for mini-batch autoencoder training.')
parser.add_argument('--ae_weight_decay', '-ae_wd', type=float, default=1e-6,
                    help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')

args = parser.parse_args()


def main(net_name, xp_path, data_path, load_model, eta,
         ratio_known_normal, ratio_known_outlier, ratio_pollution, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay,
         pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay,
         num_threads, n_jobs_dataloader, normal_class, known_outlier_class, n_known_outlier_classes):
    print("xp path: {}".format(args.xp_path))
    print("Data path: {}".format(args.data_path))
    print("Dataset: {}".format(dataset_name))
    print("Normal class: {}".format(normal_class))
    print('Ratio of labeled normal train samples: {:.2f}'.format(ratio_known_normal))
    print('Ratio of labeled anomalous samples: {:.2f}'.format(ratio_known_outlier))
    print('Pollution ratio of unlabeled train data: {:.2f}'.format(ratio_pollution))
    print('Number of known anomaly classes: {}'.format(n_known_outlier_classes))
    print('Network: {}'.format(net_name))

    print('Eta-parameter: {:.2f}'.format(eta))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print('Set seed to {}'.format(seed))

    if num_threads > 0:
        torch.set_num_threads(num_threads)

    print('Computation device: {}'.format(device))
    print('Number of threads: {}'.format(num_threads))
    print('Number of dataloader workders: {}'.format(n_jobs_dataloader))

    dataset = load_dataset(data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution)

    if n_known_outlier_classes > 1:
        print('Known anomaly classes: {}'.format(dataset.known_outlier_classes))

    deepSAD = DeepSAD(args.eta)
    deepSAD.set_network(net_name)

    if load_model:
        deepSAD.load_model(model_path=load_model, load_ae=True, map_location=device)
        print('Loading model from {}'.format(load_model))

    print('Pretraining: {}'.format(pretrain))

    if pretrain:
        print('Pretraining optimizer: {}'.format(args.ae_optimizer_name))
        print('Pretraining learning rate: {}'.format(args.ae_lr))
        print('Pretraining epochs: {}'.format(args.ae_n_epochs))
        print('Pretraining learning rate scheduler milestones: {}'.format(args.ae_lr_milestone))
        print('Pretraining batch size: {}'.format(args.ae_batch_size))
        print('Pretraining weight decay: {}'.format(args.ae_weight_decay))

    ae_lr_milestone = [30, 80]
    deepSAD.pretrain(dataset,
                     optimizer_name=args.ae_optimizer_name,
                     lr=args.ae_lr,
                     n_epochs=args.ae_n_epochs,
                     lr_milestones=ae_lr_milestone,
                     batch_size=args.ae_batch_size,
                     weight_decay=args.ae_weight_decay,
                     device=args.device,
                     n_jobs_dataloader=args.n_jobs_dataloader)

    deepSAD.save_ae_result(export_json=xp_path + '/ae_result.json')

    print('Training optimizer: {}'.format(args.optimizer_name))
    print('Training learning rate: {}'.format(args.lr))
    print('Training epochs: {}'.format(args.n_epochs))
    print('Training learning rate scheduler milestones: {}'.format(args.lr_milestone))
    print('Training batch size: {}'.format(args.batch_size))
    print('Training weight decay: {}'.format(args.weight_decay))

    lr_milestone = [30, 80]

    deepSAD.train(dataset,
                  optimizer_name=args.optimizer_name,
                  lr=args.lr,
                  n_epochs=args.n_epochs,
                  lr_milestones=lr_milestone,
                  batch_size=args.batch_size,
                  weight_decay=args.weight_decay,
                  device=device,
                  n_jobs_dataloader=n_jobs_dataloader)

    deepSAD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    deepSAD.save_results(export_json=xp_path + '/results.json')
    deepSAD.save_model(export_model=xp_path + '/model.tar')

    # Plot most anomalous and most normal test samples
    indices, labels, scores = zip(*deepSAD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_all_sorted = indices[np.argsort(scores)]  # from lowest to highest score
    idx_normal_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # from lowest to highest score

    if dataset_name in ('mnist', 'fmnist', 'cifar10'):

        if dataset_name in ('mnist', 'fmnist'):
            X_all_low = dataset.test_set.data[idx_all_sorted[:32], ...].unsqueeze(1)
            X_all_high = dataset.test_set.data[idx_all_sorted[-32:], ...].unsqueeze(1)
            X_normal_low = dataset.test_set.data[idx_normal_sorted[:32], ...].unsqueeze(1)
            X_normal_high = dataset.test_set.data[idx_normal_sorted[-32:], ...].unsqueeze(1)

        if dataset_name == 'cifar10':
            X_all_low = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[:32], ...], (0, 3, 1, 2)))
            X_all_high = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[-32:], ...], (0, 3, 1, 2)))
            X_normal_low = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[:32], ...], (0, 3, 1, 2)))
            X_normal_high = torch.tensor(
                np.transpose(dataset.test_set.data[idx_normal_sorted[-32:], ...], (0, 3, 1, 2)))

        plot_images_grid(X_all_low, export_img=xp_path + '/all_low', padding=2)
        plot_images_grid(X_all_high, export_img=xp_path + '/all_high', padding=2)
        plot_images_grid(X_normal_low, export_img=xp_path + '/normals_low', padding=2)
        plot_images_grid(X_normal_high, export_img=xp_path + '/normals_high', padding=2)


if __name__ == "__main__":
    main(args.net_name, args.xp_path, args.data_path, args.load_model, args.eta,
         args.ratio_known_normal, args.ratio_known_outlier, args.ratio_pollution, args.device, args.seed,
         args.optimizer_name, args.lr, args.n_epochs, args.lr_milestone, args.batch_size, args.weight_decay,
         args.pretrain, args.ae_optimizer_name, args.ae_lr, args.ae_n_epochs, args.ae_lr_milestone, args.ae_batch_size,
         args.ae_weight_decay,
         args.num_threads, args.n_jobs_dataloader, args.normal_class, args.known_outlier_class,
         args.n_known_outlier_classes)
