import argparse
from easydict import EasyDict as edict
import torch
import model_zoo
from cfg import cfg
from torch.utils.data import DataLoader
import dataset
from trainer import Trainer
from losses3D import BCEDiceLoss
from losses3D.losses_1 import TverskyLoss
def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', type=bool, default=False, dest='resume')
    parser.add_argument('-p', '--module_path', type=str, default=None, dest='module_path')
    args = vars(parser.parse_args())
    cfg.update(args)
    return edict(cfg)
    
def main(cfg):
    model, opt = model_zoo.create_model(cfg)
    model.to(cfg.device)
    train_dataset = dataset.My_dataset(cfg.train_data_path, data_augment=True)
    valid_dataset = dataset.My_dataset(cfg.val_data_path)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: (1 - epoch / cfg.train_epochs)**0.9)

    if cfg.resume:
        cfg.start_epoch = model.restore_checkpoint(cfg.module_path, optimizer=opt, lr_scheduler=lr_scheduler)
    criterion = BCEDiceLoss(classes=cfg.classes)
    # criterion = TverskyLoss(0.2, 0.8)
    trainer = Trainer(cfg, model, criterion, opt, lr_scheduler=lr_scheduler, train_data_loader=train_loader, valid_data_loader=valid_loader)
    trainer.training()
if __name__ == '__main__':
    cfg_update = get_args(**cfg)
    main(cfg_update)