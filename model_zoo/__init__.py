import torch.optim as optim

from .COVIDNet import CovidNet, CNN
from .DenseVoxelNet import DenseVoxelNet
from .Densenet3D import DualPathDenseNet, DualSingleDenseNet, SinglePathDenseNet
from .HighResNet3D import HighResNet3D
from .HyperDensenet import HyperDenseNet, HyperDenseNet_2Mod
from .ResNet3DMedNet import generate_resnet3d
from .ResNet3D_VAE import ResNet3dVAE
from .SkipDenseNet3D import SkipDenseNet3D
from .Unet2D import Unet
from .Unet3D import UNet3D
from .Vnet import VNet, VNetLight
from .ResUNet_3D import ResUNet_3D
from .UNET3D_ours import UNet_ours
from .WingsNet import WingsNet
from .WingsNet_new import WingsNet_new
from .UnetPlusPlus.res_unet_plus import ResUnetPlusPlus
model_list = ['UNET3D', 'DENSENET1', "UNET2D", 'DENSENET2', 'DENSENET3', 'HYPERDENSENET', "SKIPDENSENET3D",
              "DENSEVOXELNET", 'VNET', 'VNET2', "RESNET3DVAE", "RESNETMED3D", "COVIDNET1", "COVIDNET2", "CNN",
              "HIGHRESNET", "ResUNet_2stage", "ResUNet_3D", "UNET3D_ours", "WingsNet", "WingsNet_new", "ResUnetPlusPlus"]


def create_model(args):
    model_name = args.model
    assert model_name in model_list
    optimizer_name = args.opt
    lr = args.lr
    in_channels = args.inChannels
    num_classes = args.classes
    weight_decay = 0.0000000001
    print("Building Model . . . . . . . ." + model_name)
    global model
    if model_name == 'VNET2':
        model = VNetLight(in_channels=in_channels, elu=False, classes=num_classes)
    elif model_name == 'VNET':
        model = VNet(in_channels=in_channels, elu=False, classes=num_classes)
    elif model_name == 'UNET3D':
        model = UNet3D(in_channels=in_channels, n_classes=num_classes, base_n_filter=16)
    elif model_name == 'DENSENET1':
        model = SinglePathDenseNet(in_channels=in_channels, classes=num_classes)
    elif model_name == 'DENSENET2':
        model = DualPathDenseNet(in_channels=in_channels, classes=num_classes)
    elif model_name == 'DENSENET3':
        model = DualSingleDenseNet(in_channels=in_channels, drop_rate=0.1, classes=num_classes)
    elif model_name == "UNET2D":
        model = Unet(in_channels, num_classes)
    elif model_name == "RESNET3DVAE":
        model = ResNet3dVAE(in_channels=in_channels, classes=num_classes, dim=args.patch_size)
    elif model_name == "SKIPDENSENET3D":
        model = SkipDenseNet3D(growth_rate=16, num_init_features=32, drop_rate=0.1, classes=num_classes)
    elif model_name == "COVIDNET1":
        model = CovidNet('small', num_classes)
    elif model_name == "COVIDNET2":
        model = CovidNet('large', num_classes)
    elif model_name == "CNN":
        model = CNN(num_classes, 'resnet18')
    elif model_name == "HYPERDENSENET":
        if in_channels == 2:
            model = HyperDenseNet_2Mod(classes=num_classes)
        elif in_channels == 3:
            model = HyperDenseNet(classes=num_classes)
        else:
            raise NotImplementedError
    elif model_name == "DENSEVOXELNET":
        model = DenseVoxelNet(in_channels=in_channels, classes=num_classes)
    elif model_name == "HIGHRESNET":
        model = HighResNet3D(in_channels=in_channels, classes=num_classes)
    elif model_name == "RESNETMED3D":
        depth = 18
        model = generate_resnet3d(in_channels=in_channels, classes=num_classes, model_depth=depth)
    elif model_name == 'ResUNet_3D':
        model = ResUNet_3D(in_channels=in_channels, out_channels=num_classes)
    elif model_name == 'UNET3D_ours':
        model = UNet_ours(n_channels=in_channels, n_classes=num_classes)

    elif model_name == 'WingsNet':
        model = WingsNet(in_channels, num_classes)
    elif model_name == 'ResUnetPlusPlus':
        model = ResUnetPlusPlus(in_channels, num_classes)
    print(model_name, 'Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    global optimizer
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
    elif optimizer_name == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSPROP':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer
