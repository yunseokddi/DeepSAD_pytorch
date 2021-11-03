from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder


def build_network(net_name, ae_net=None):
    net = CIFAR10_LeNet()

    return net


def build_autoencoder(net_name):
    ae_net = CIFAR10_LeNet_Autoencoder()

    return ae_net
