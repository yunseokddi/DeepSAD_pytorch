from dataset.main import load_dataset

import argparse

parser = argparse.ArgumentParser(description='Train Deep SAD model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

################################################################################
# net category
# 'mnist_LeNet', 'fmnist_LeNet', 'cifar10_LeNet', 'arrhythmia_mlp',
# 'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
# 'thyroid_mlp'
################################################################################
parser.add_argument('--net_name', '-net', type=str, default='cifar10_LeNet', help='Enter backbone network')
parser.add_argument('--xp_path', '-xp', type=str, default='./experiment/', help='Tensorboard log path')
parser.add_argument('--data_path', '-data', type=str, default='./data/', help='Dataset path')
parser.add_argument('--load_model', '-load', type=bool, default=None, help='load pretrained weight')
parser.add_argument('--device', '-device', type=str, default='cuda',
                    help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
parser.add_argument('--seed', '-seed', type=int, default=1, help='Set seed. If -1, use randomization.')
parser.add_argument('--pretrain', '-pretrain', type=bool, default=True,
                    help='Pretrain neural network parameters via autoencoder.')

################################################################################
# Dataset settings
################################################################################
parser.add_argument('--num_threads', '-num_threads', type=int, default=0,
                    help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
parser.add_argument('--n_jobs_dataloader', '-n_jobs_dataloader', type=int, default=0,
                    help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
parser.add_argument('--normal_class', '-normal_class', type=int, default=0,
                    help='Specify the normal class of the dataset (all other classes are considered anomalous).')
parser.add_argument('--known_outlier_class', '-known_outlier_class', type=int, default=1,
                    help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
parser.add_argument('--n_known_outlier_classes', '-n_known_outlier_classes', type=int, default=0,
                    help='Number of known outlier classes.'
                         'If 0, no anomalies are known.'
                         'If 1, outlier class as specified in --known_outlier_class option.'
                         'If > 1, the specified number of outlier classes will be sampled at random.')

################################################################################
# DeepSAD settings
################################################################################
parser.add_argument('--num_epochs', '-e', type=int, default=50, help='Num of epochs to train')
parser.add_argument('--eta', '-eta', type=float, default=1.0, help='Deep SAD hyperparameter eta (must be 0 < eta).')
parser.add_argument('--ratio_known_normal', '-normal_ratio', type=float, default=0.0,
                    help='Ratio of known (labeled) normal training examples.')
parser.add_argument('--ratio_known_outlier', '-outlier_ratio', type=float, default=0.0,
                    help='Ratio of known (labeled) anomalous training examples.')
parser.add_argument('--ratio_pollution', '-ratio_pollution', type=float, default=0.0,
                    help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
parser.add_argument('--lr', '-lr', type=float, default=1e-3,
                    help='Initial learning rate for Deep SAD network training. Default=0.001')
parser.add_argument('--lr_milestone', '-lr_milestone', type=int, default=0,
                    help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='Batch size for mini-batch training.')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-6,
                    help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')
parser.add_argument('--optimizer_name', '-optimizer', type=str, default='adam',
                    help='Name of the optimizer to use for Deep SAD network training.')

################################################################################
# AUTOENCODER settings
################################################################################
parser.add_argument('--ae_optimizer_name', '-ae_optimizer', type=str, default='adam',
                    help='Name of the optimizer to use for autoencoder pretraining.')
parser.add_argument('--ae_lr', '-ae_lr', type=float, default=1e-3,
                    help='Initial learning rate for autoencoder pretraining. Default=0.001')
parser.add_argument('--ae_n_epochs', '-ae_n_epochs', type=int, default=100,
                    help='Number of epochs to train autoencoder.')
parser.add_argument('--ae_lr_milestone', '-ae_lr_milestone', type=int, default=0,
                    help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
parser.add_argument('--ae_batch_size', '-ae_bs', type=int, default=128,
                    help='Batch size for mini-batch autoencoder training.')
parser.add_argument('--ae_weight_decay', '-ae_wd', type=float, default=1e-6,
                    help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')


print("asd")
# def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, eta,
#          ratio_known_normal, ratio_known_outlier, ratio_pollution, device, seed,
#          optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay,
#          pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay,
#          num_threads, n_jobs_dataloader, normal_class, known_outlier_class, n_known_outlier_classes):
#
