import os
import logging
from datetime import datetime
import random
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch
from torch.autograd import Variable
from torch.backends import cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import albumentations as A
import cv2

import sys
sys.path.append('data/home/LAHNet/')  # Package path

from data import mulDataset  # Dataset processing
from models import LAHNet  # Import model
from loss import Make_Criterion
from evaluation import evaluate_circle, evaluate_iris
from utils import check_mkdir, get_circle_edge

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 120
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.benchmark = False
cudnn.deterministic = True
# assert dataset_name in ['MICHE', 'CASIA-Iris-M1', 'CASIA-iris-distance', 'UBIRIS.v2']
dataset_name = 'MICHE'
model_name = 'lahnet'
pretrain_model_path = None


def get_args():
    parser = argparse.ArgumentParser(description='Train paprmeters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', type=int, default=600, dest='epoch_num')
    parser.add_argument('-b', '--batch-size', type=int, nargs='?', default=8, dest='batch_size')
    parser.add_argument('-b1', '--val-batch-size', type=int, nargs='?', default=1, dest='val_batch_size')
    parser.add_argument('-lr', '--learning-rate', type=float, nargs='?', default=0.002, dest='lr')
    parser.add_argument('-lg', '--log-name', type=str, default='logging.log', dest='log_name')
    parser.add_argument('-ckp', '--checkpoints', type=str, default=pretrain_model_path, help='Load model from a .pth file', dest='snapshot')
    parser.add_argument('-d', '--deep-supervision', type=int, default=1, help='train aux', dest='DS')

    return parser.parse_args()


def main(train_args):
    ########################################### logging and writer #############################################
    writer = SummaryWriter(log_dir=os.path.join(log_path, 'summarywriter_'+train_args['log_name'].split('.')[0]), comment=train_args['log_name'])

    logging.info('------------------------------------------------train configs------------------------------------------------')
    logging.info(train_args)

    ############################################# define a CNN #################################################

    logging.info(seed)
    net = LAHNet(4, 0.1, 32).cuda()
    logging.info(net)

    total_params = sum([param.nelement() for param in net.parameters()])
    logging.info(f'net Params is {total_params/1e6}M')

    ########################################### dataset(MICHE & UBIRIS.v2) #######################################
    train_augment = A.Compose([
        A.PadIfNeeded(p=1, min_height=384, min_width=384, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),

        A.Compose([
            A.CLAHE(p=0.5),
            A.RandomContrast(p=0.2),
            A.RGBShift(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.ChannelShuffle(p=0.2),
            A.InvertImg(p=0.2)
        ], p=0.5)
    ])
    val_augment = A.Compose([
        A.PadIfNeeded(p=1, min_height=384, min_width=400, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.CenterCrop(384, 384),
    ])


    ########################################### dataset(CASIA-Iris-M1 & CASIA-iris-distance) #############################################
    # train_augment = A.Compose([
    #    A.CenterCrop(384, 384),
    #    A.HorizontalFlip(p=0.4),
    #    A.VerticalFlip(p=0.4),
    #    A.Equalize(p=0.3),
    # ])
    # val_augment = A.Compose([
    #     A.CenterCrop(384, 384),
    # ])

    train_dataset = mulDataset(dataset_name, mode='train', transform=train_augment)
    val_dataset = mulDataset(dataset_name, mode='val', transform=val_augment)
    train_loader = DataLoader(train_dataset, batch_size=train_args['batch_size'], num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_args['val_batch_size'], num_workers=8, drop_last=True)

    train_data_size = train_dataset.__len__()
    valid_data_size = val_dataset.__len__()
    logging.info(f'data augment: \n{train_augment}\n, \n{val_augment}\n')
    logging.info(f'The dataset {dataset_name} is ready!')
    logging.info(f'Training num:{train_data_size}')
    logging.info(f'Validation num:{valid_data_size}')

    ########################################### criterion #############################################
    criterion = Make_Criterion(train_args)  #bceloss
    logging.info(f'''criterion is ready! \n{criterion}\n''')

    ########################################### optimizer #############################################
    # Adam
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * train_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr'], 'weight_decay': 1e-8}
    ], betas=(0.95, 0.999))
    logging.info(f'optimizer is ready! \n{optimizer}\n')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3,
                                                                     T_mult=2, eta_min=1e-5, last_epoch=-1)
    logging.info(f'scheduler is ready! \n{scheduler}\n')

    ######################################### train and val ############################################
    net.train()
    try:
        curr_epoch = 1
        train_args['best_record'] = {'epoch': 0, 'val_loss': 999, 'E1': 999, 'IoU': 0, 'Dice': 0, 'F1': 0, 'recall': 0}
        train_args['best_record_inner'] = {'epoch': 0, 'val_loss': 999, 'E1': 999, 'IoU': 0, 'Dice': 0, 'Hsdf':999}
        train_args['best_record_outer'] = {'epoch': 0, 'val_loss': 999, 'E1': 999, 'IoU': 0, 'Dice': 0, 'Hsdf':999}

        for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
            train(writer, train_loader, net, criterion, optimizer, epoch, train_args)
            val_loss = validate(writer, val_loader, net, criterion, optimizer, epoch, train_args)
            scheduler.step(val_loss)
        writer.close()

        print('best  record   epoch:{:2d}   E1:{:.5f} '.format(
                train_args['best_record']['epoch'], train_args['best_record']['E1']))

        logging.info('-------------------------------------------------best record------------------------------------------------')
        logging.info('mask   epoch:{}   val loss {:.5f}  E1:{:.5f}   IoU:{:.5f}   Dice:{:.5f}  F1:{:.5f}   recall:{:.5f}'.format(
            train_args['best_record']['epoch'], train_args['best_record']['val_loss'], train_args['best_record']['E1'],
            train_args['best_record']['IoU'], train_args['best_record']['Dice'], train_args['best_record']['F1'],
            train_args['best_record']['recall']
            ))
        logging.info('outer   epoch:{}  val loss {:.5f}  E1:{:.5f}   IoU:{:.5f}   Dice:{:.5f}  Hsdf:{:.5f}'.format(
            train_args['best_record_outer']['epoch'], train_args['best_record_outer']['val_loss'], train_args['best_record_outer']['E1'],
            train_args['best_record_outer']['IoU'], train_args['best_record_outer']['Dice'], train_args['best_record_outer']['Hsdf']
            ))
        logging.info('inner   epoch:{}  val loss {:.5f}  E1:{:.5f}   IoU:{:.5f}   Dice:{:.5f}  Hsdf:{:.5f}'.format(
            train_args['best_record_inner']['epoch'], train_args['best_record_inner']['val_loss'], train_args['best_record_inner']['E1'],
            train_args['best_record_inner']['IoU'], train_args['best_record_inner']['Dice'], train_args['best_record_inner']['Hsdf']
            ))

    except KeyboardInterrupt:
        if isinstance(net, torch.nn.DataParallel):
            torch.save(net.module.state_dict(), log_path+'/INTERRUPTED.pth')
        else:
            torch.save(net.state_dict(), log_path+'/INTERRUPTED.pth')
        logging.info('Saved interrupt!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


def train(writer, train_loader, net, criterion, optimizer, epoch, train_args):
    logging.info('--------------------------------------------------training...------------------------------------------------')
    iters = len(train_loader)
    curr_iter = (epoch - 1) * iters  #tatol iters/batch number

    for i, data in enumerate(train_loader):
        image, mask, iris_mask, pupil_mask, loc = \
            data['image'], data['mask'], data['iris_edge_mask'], data['pupil_edge_mask'], data['heatmap'] #BCHW
        # image, mask, iris_mask, pupil_mask = \
        #     data['image'], data['mask'], data['iris_edge_mask'], data['pupil_edge_mask']
        assert image.size()[2:] == mask.size()[2:]

        image = image.cuda()
        mask = mask.cuda()
        iris_mask = iris_mask.cuda()
        pupil_mask = pupil_mask.cuda()
        loc = loc.cuda()

        _input = torch.cat([image, loc], dim=1)
        # _input = image

        optimizer.zero_grad()  # reset gradient
        _output = net(_input)
        pred_masks = _output['pred_masks']
        pred_mask, pred_iris_mask, pred_pupil_mask = \
            pred_masks[0][:,0:1,:,:], pred_masks[0][:,1:2,:,:], pred_masks[0][:,2:3,:,:]

        loss_mask = criterion(pred_mask, mask)
        loss_iris = criterion(pred_iris_mask, iris_mask)
        loss_pupil = criterion(pred_pupil_mask, pupil_mask)

        coarse_masks = _output['coarse_masks']
        loss_aux = sum([criterion(coarse_mask, transforms.Resize((coarse_mask.size()[2:]))(mask)) for coarse_mask in coarse_masks]) / train_args['batch_size']

        beta = 0.3
        loss = loss_mask + loss_iris + loss_pupil + beta * loss_aux
        # loss = loss_mask + loss_iris + loss_pupil

        loss.backward()
        optimizer.step()

        writer.add_scalar('train_loss/iter', loss.item(), curr_iter)
        writer.add_scalar('train_loss_mask/iter', loss_mask.item(), curr_iter)
        writer.add_scalar('train_loss_iris/iter', loss_iris.item(), curr_iter)
        writer.add_scalar('train_loss_pupil/iter', loss_pupil.item(), curr_iter)
        writer.add_scalar('train/loss', loss.item(), epoch)
        writer.add_scalar('train/loss_mask', loss_mask.item(), epoch)
        writer.add_scalar('train/loss_iris', loss_iris.item(), epoch)
        writer.add_scalar('train/loss_pupil', loss_pupil.item(), epoch)


        if (i + 1) % train_args['print_freq'] == 0:
            print('epoch:{:2d}  iter/iters:{:3d}/{:3d}  train_loss:{:.9f}  loss_mask:{:.9f}  loss_iris:{:.9}  loss_pupil:{:.9}'.format(
               epoch, i+1, iters, loss, loss_mask, loss_iris, loss_pupil))
            logging.info('epoch:{:2d}  iter/iters:{:3d}/{:3d}  train_loss:{:.9f}  loss_mask:{:.9f}  loss_iris:{:.9}  loss_pupil:{:.9}'.format(
                epoch, i+1, iters, loss, loss_mask, loss_iris, loss_pupil))

        curr_iter += 1

def validate(writer, val_loader, net, criterion, optimizer, epoch, train_args):
    net.eval()

    e1, iou, dice, f1, recall, precision = 0, 0, 0, 0, 0, 0
    iris_e1, iris_dice, iris_iou, iris_hsdf = 0, 0, 0, 0
    pupil_e1, pupil_dice, pupil_iou, pupil_hsdf = 0, 0, 0, 0
    iris_e1_raw, iris_dice_raw, iris_iou_raw = 0, 0, 0
    pupil_e1_raw, pupil_dice_raw, pupil_iou_raw = 0, 0, 0

    L = len(val_loader)

    for i, data in enumerate(val_loader):
        image, mask, iris_edge, iris_mask, pupil_edge, pupil_mask, loc = \
            data['image'], data['mask'], data['iris_edge'],\
            data['iris_edge_mask'], data['pupil_edge'], data['pupil_edge_mask'], data['heatmap']
        # image, mask, iris_edge, iris_mask, pupil_edge, pupil_mask = \
        #     data['image'], data['mask'], data['iris_edge'], \
         #    data['iris_edge_mask'], data['pupil_edge'], data['pupil_edge_mask']
       
        image = Variable(image).cuda()
        mask = Variable(mask).cuda()
        iris_edge = Variable(iris_edge).cuda()
        iris_mask = Variable(iris_mask).cuda()
        pupil_edge = Variable(pupil_edge).cuda()
        pupil_mask = Variable(pupil_mask).cuda()
        loc = Variable(loc).cuda()

        _input = torch.cat([image, loc], dim=1)
        # _input = image
        
        with torch.no_grad():
            _output = net(_input)

        pred_masks = _output['pred_masks']
        pred_mask, pred_iris_mask, pred_pupil_mask = \
            pred_masks[0][:,0:1,:,:], pred_masks[0][:,1:2,:,:], pred_masks[0][:,2:3,:,:]

        loss_mask = criterion(pred_mask, mask)
        loss_iris = criterion(pred_iris_mask, iris_mask)
        loss_pupil = criterion(pred_pupil_mask, pupil_mask)
        val_loss = loss_mask + loss_iris + loss_pupil        

        ################### post process #####################
        pred_iris_circle_mask, pred_iris_edge, _ = get_circle_edge(pred_iris_mask)
        pred_pupil_circle_mask, pred_pupil_egde, _ = get_circle_edge(pred_pupil_mask) # draw a circle on pupil_edge for display

        ################### val for iris mask ##################
        val_results = evaluate_iris(pred_mask, mask, dataset_name)
        e1 += torch.true_divide(val_results['E1'], L)
        iou += torch.true_divide(val_results['IoU'], L)
        dice += torch.true_divide(val_results['Dice'], L)
        f1 += torch.true_divide(val_results['F1'], L)
        recall += torch.true_divide(val_results['recall'], L)
        # precision += torch.true_divide(val_results['precision'], L)

        ################### val for iris edge ##################
        iris_val_results = evaluate_circle(pred_iris_circle_mask, iris_mask, pred_iris_edge, iris_edge, dataset_name)
        iris_e1 += torch.true_divide(iris_val_results['E1'], L)
        iris_dice += torch.true_divide(iris_val_results['Dice'], L)
        iris_iou += torch.true_divide(iris_val_results['IoU'], L)
        iris_hsdf += torch.true_divide(iris_val_results['Hsdf'], L)

        iris_val_results_raw = evaluate_iris(pred_iris_mask, iris_mask, dataset_name)    
        iris_e1_raw += torch.true_divide(iris_val_results_raw['E1'], L)
        iris_iou_raw += torch.true_divide(iris_val_results_raw['IoU'], L)
        iris_dice_raw +=torch.true_divide(iris_val_results_raw['Dice'], L)

        #################### val for pupil edge ##################
        pupil_val_results = evaluate_circle(pred_pupil_circle_mask, pupil_mask, pred_pupil_egde, pupil_edge, dataset_name)
        pupil_e1 += torch.true_divide(pupil_val_results['E1'], L)
        pupil_dice += torch.true_divide(pupil_val_results['Dice'], L)
        pupil_iou += torch.true_divide(pupil_val_results['IoU'], L)
        pupil_hsdf += torch.true_divide(pupil_val_results['Hsdf'], L)

        pupil_val_results_raw = evaluate_iris(pred_pupil_mask, pupil_mask, dataset_name)    
        pupil_e1_raw += torch.true_divide(pupil_val_results_raw['E1'], L)
        pupil_iou_raw += torch.true_divide(pupil_val_results_raw['IoU'], L)
        pupil_dice_raw += torch.true_divide(pupil_val_results_raw['Dice'], L)
        

    logging.info('------------------------------------------------current val result-----------------------------------------------')    
    logging.info('>maks      epoch:{:2d}   val loss:{:.7f}   learning rate:{:.12f}   E1:{:.7f}   Dice:{:.7f}   IOU:{:.7f}   F1:{:.7f}   recall:{:.7f}'. \
        format(epoch, loss_mask, optimizer.param_groups[1]['lr'], e1, dice, iou, f1, recall))
    logging.info('>iris_raw  epoch:{:2d}   val loss:{:.7f}   learning rate:{:.12f}   E1:{:.7}    Dice:{:.7f}   IOU:{:.7f}   Hsdf:{:.7f}'. \
        format(epoch, loss_iris, optimizer.param_groups[1]['lr'], iris_e1_raw, iris_dice_raw, iris_iou_raw, iris_hsdf))
    logging.info('>iris      epoch:{:2d}   val loss:{:.7f}   learning rate:{:.12f}   E1:{:.7}    Dice:{:.7f}   IOU:{:.7f}   Hsdf:{:.7f}'. \
        format(epoch, loss_iris, optimizer.param_groups[1]['lr'], iris_e1, iris_dice, iris_iou, iris_hsdf))
    logging.info('>pupil_raw epoch:{:2d}   val loss:{:.7f}   learning rate:{:.12f}   E1:{:.7}    Dice:{:.7f}   IOU:{:.7f}   Hsdf:{:.7f}'. \
        format(epoch, loss_pupil, optimizer.param_groups[1]['lr'], pupil_e1_raw, pupil_dice_raw, pupil_iou_raw, pupil_hsdf))
    logging.info('>pupil     epoch:{:2d}   val loss:{:.7f}   learning rate:{:.12f}   E1:{:.7}    Dice:{:.7f}   IOU:{:.7f}   Hsdf:{:.7f}'. \
        format(epoch, loss_pupil, optimizer.param_groups[1]['lr'], pupil_e1, pupil_dice, pupil_iou, pupil_hsdf))
    
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/e1', e1, epoch)
    writer.add_scalar('val/iou', iou, epoch)
    writer.add_scalar('val/dice', dice, epoch)
    writer.add_scalar('val/f1', f1, epoch)
    writer.add_scalar('val/recall', recall, epoch)
    writer.add_scalar('lr', optimizer.param_groups[1]['lr'], epoch)

    writer.add_images('image', image, epoch)
    writer.add_images('mask', mask, epoch)
    writer.add_images('pred_mask', torch.sigmoid(pred_mask)>0.5, epoch)
    writer.add_images('iris_mask', iris_mask, epoch)
    writer.add_images('pred_iris_mask', torch.sigmoid(pred_iris_mask)>0.5, epoch)
    writer.add_images('pupil_mask', pupil_mask, epoch)
    writer.add_images('pred_pupil_mask', torch.sigmoid(pred_pupil_mask)>0.5, epoch)

    if e1 < train_args['best_record']['E1']:
        if isinstance(net, torch.nn.DataParallel):
            torch.save(net.module.state_dict(),
                       os.path.join(checkpoint_path, train_args['log_name'].split('.')[0] + '.pth'))
        else:
            torch.save(net.state_dict(), os.path.join(checkpoint_path, train_args['log_name'].split('.')[0] + '.pth'))

    if e1 < train_args['best_record']['E1']:
        train_args['best_record']['val_loss'] = val_loss
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['E1'] = e1
        train_args['best_record']['IoU'] = iou
        train_args['best_record']['Dice'] = dice
        train_args['best_record']['F1'] = f1
    
    if iris_e1 < train_args['best_record_outer']['E1']:
        train_args['best_record_outer']['val_loss'] = val_loss
        train_args['best_record_outer']['epoch'] = epoch
        train_args['best_record_outer']['E1'] = iris_e1
        train_args['best_record_outer']['IoU'] = iris_iou
        train_args['best_record_outer']['Dice'] = iris_dice
        train_args['best_record_outer']['Hsdf'] = iris_hsdf

    if pupil_e1 < train_args['best_record_inner']['E1']:
        train_args['best_record_inner']['val_loss'] = val_loss
        train_args['best_record_inner']['epoch'] = epoch
        train_args['best_record_inner']['E1'] = pupil_e1
        train_args['best_record_inner']['IoU'] = pupil_iou
        train_args['best_record_inner']['Dice'] = pupil_dice
        train_args['best_record_inner']['Hsdf'] = pupil_hsdf

    net.train()
    return val_loss


if __name__ == '__main__':
    args = get_args()
    train_args = {
        'epoch_num': args.epoch_num,
        'batch_size': args.batch_size,
        'val_batch_size': args.val_batch_size,
        'lr': args.lr,
        'snapshot': args.snapshot,  # empty string denotes learning from scratch
        'log_name': args.log_name,
        'print_freq': 50,
        'deep_supervise': args.DS
    }

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    check_mkdir(os.path.join('./experiments/', dataset_name))
    check_mkdir(os.path.join('./experiments/', dataset_name, model_name))
    save_dir = os.path.join('./experiments/', dataset_name, model_name)

    log_path = os.path.join(save_dir, start_time + '_' + train_args['log_name'].split('.')[0])
    checkpoint_path = os.path.join(log_path, 'checkpoints')
    check_mkdir(log_path)
    check_mkdir(checkpoint_path)
    logging.basicConfig(
        filename=os.path.join(log_path, train_args['log_name']),
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    main(train_args)
