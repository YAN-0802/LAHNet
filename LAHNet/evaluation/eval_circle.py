import cv2
import numpy as np
import torch
import torch.nn as nn

from hausdorff import hausdorff_distance


def to_inputsize(pred_masks, true_masks, pred_edges, true_edges, dataset_name):
    assert dataset_name in ['MICHE', 'CASIA-Iris-M1', 'CASIA-iris-distance', 'UBIRIS.v2']
    if dataset_name == 'CASIA-Iris-M1':
        tH, tW = 400, 400
    else:
        tH, tW = 480, 640

    h, w = true_masks.shape[2], true_masks.shape[3]
    if h>tH and w>tW:
        crop_to = torchvision.transforms.CenterCrop((tH, tW))
        true_masks = crop_to(true_masks)
        pred_masks = crop_to(pred_masks)
        pred_edges = crop_to(pred_edges)
        true_edges = crop_to(true_edges)
    if h<tH and w<tW:
        pad_to = nn.ZeroPad2d(((tH-h)//2, (tH-h)//2, (tW-w)//2, (tW-w)//2))
        true_masks = pad_to(true_masks)
        pred_masks = pad_to(pred_masks)
        pred_edges = pad_to(pred_edges)
        true_edges = pad_to(true_edges)
    return pred_masks, true_masks, pred_edges, true_edges
  
def compute_tfpn(pred_mask,true_mask):
    '''
    return: dict {tp, fp, tn, fn}
    input shape: cwh = 1x400x400
    '''
    c, r = true_mask.shape[1], pred_mask.shape[2]
    num_pixel = c*r

    # to BoolTensor
    true_mask = true_mask>0
    pred_mask = pred_mask>0
    
    tp = (true_mask & pred_mask).sum()
    fp = (~true_mask & pred_mask).sum()
    tn = (~(true_mask | pred_mask)).sum()
    fn = (true_mask & (~pred_mask)).sum()

    return {
        'tp': torch.true_divide(tp , num_pixel),
        'fp': torch.true_divide(fp , num_pixel),
        'tn': torch.true_divide(tn , num_pixel),
        'fn': torch.true_divide(fn , num_pixel)
    }

def compute_e1(n_batch, pred_masks, true_masks):
    sum_e1 = 0
    for i in range(n_batch):
        tpfn = compute_tfpn(pred_masks[i],true_masks[i])
        fp, fn = tpfn['fp'], tpfn['fn']
        sum_e1 += fp+fn
    return sum_e1/n_batch


def compute_miou(n_batch, true_masks, pred_masks):
    sum_iou = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(pred_masks[i], true_masks[i])
        tp, fp, fn = tfpn['tp'], tfpn['fp'], tfpn['fn']
        if tp+fn+fp == 0:
            iou=1
        else:
            iou=tp/(tp+fn+fp)
        sum_iou += iou
    return sum_iou/n_batch

def compute_dice(n_batch, pred_masks, true_masks):
    sum_dice = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(pred_masks[i], true_masks[i])
        tp, fp, fn = tfpn['tp'], tfpn['fp'], tfpn['fn']
        if 2*tp+fn+fp == 0:
            dice=1
        else:
            dice=2*tp/(2*tp+fn+fp)
        sum_dice += dice
    return sum_dice/n_batch

def compute_recall(n_batch, pred_masks, true_masks):
    sum_rc = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(pred_masks[i], true_masks[i])
        tp, fn = tfpn['tp'], tfpn['fn']
        sum_rc += tp / (tp+fn)
    return sum_rc/n_batch

def compute_precision(n_batch, pred_masks, true_masks):
    sum_p = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(pred_masks[i], true_masks[i])
        tp, fp = tfpn['tp'], tfpn['fp']
        sum_p += tp / (tp+fp+(1e-10))
    return sum_p/n_batch

def compute_f1(n_batch, pred_masks, true_masks):
    sum_f1 = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(pred_masks[i], true_masks[i])
        tp, fp, fn = tfpn['tp'], tfpn['fp'], tfpn['fn']
        if tp+fp == 0:
            precision = tp
        else:
            precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        if precision+recall == 0:
            f1 = tp
        else:
            f1 = (2*precision*recall) / (precision+recall)
        if f1 > 999:
            f1 = 0
        sum_f1 += f1
    return sum_f1/n_batch


def get_circle(pred_mask):
    h, w = pred_mask.shape
    pred_edge = np.zeros((h, w))
    pred_mask = cv2.Canny(pred_mask, 100, 200, L2gradient=True)
    circles = cv2.HoughCircles(pred_mask, cv2.HOUGH_GRADIENT, dp=1, minDist=pred_edge.shape[0]//3, param1=200, param2=3, minRadius=3, maxRadius=300)
    try:
        circles = np.uint16(np.around(circles))
        circle = circles[0][0]
        cv2.circle(pred_edge, (circle[0], circle[1]), circle[2], (255,255,255), 1)
    except:
        pass
    return pred_edge

def get_coords(nparray):
    coords = []

    h, w = nparray.shape
    for i in range(h):
        for j in range(w):
            if nparray[i, j] > 0:
                coords.append([i, j])
    
    return np.asarray(coords)


def Hausdorff(pred_edge, true_edge):
    pred_edge = np.asarray(pred_edge>0)
    true_edge = np.asarray(true_edge>0)
    _, h, w = true_edge.shape

    pred_coords = get_coords(pred_edge[0])
    true_coords = get_coords(true_edge[0])

    if len(pred_coords) == 0 or len(true_coords)==0:
        hsdf =  float("inf")
    else:
        hsdf = hausdorff_distance(pred_coords, true_coords) / w

    return hsdf


def compute_hsdf(n_batch, pred_edges, true_edges):
    hsdf = 0
    for i in range(n_batch):
        hsdf_i = Hausdorff(pred_edges[i], true_edges[i])
        if hsdf_i == float("inf"):
            continue
        hsdf += hsdf_i
    return hsdf/n_batch


def evaluate_circle(pred_masks, true_masks, pred_edges, true_edges, dataset_name):
    """Evaluation without the densecrf with the dice coefficient"""
    n_batch = true_masks.size()[0]

    e1 = compute_e1(n_batch, pred_masks, true_masks)
    dice = compute_dice(n_batch, pred_masks, true_masks)
    iou = compute_miou(n_batch, pred_masks, true_masks)
    hsdf = compute_hsdf(n_batch, pred_edges.cpu(), true_edges.cpu())


    return {
        'E1': e1*100,
        'IoU': iou*100,
        'Dice': dice,
        'Hsdf': hsdf*100
        
    }