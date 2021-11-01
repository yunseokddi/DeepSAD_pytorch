from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder


def build_network(net_name, ae_net=None):
    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    else:
        net = None

    return net


def build_autoencoder(net_name):
    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    else:
        ae_net = None

    return ae_net
