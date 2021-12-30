import math

import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, resnet34, mobilenet_v2, mnasnet1_0, shufflenet_v2_x1_0
import torch.nn.functional as F

import matplotlib.pyplot as plt
import scipy.misc
import imageio

from nms import nms
from lib.lane import Lane
from lib.focal_loss import FocalLoss
from lib.ghm_loss import GHMC
from .transformer import TransConvEncoderModule, build_position_encoding
from .transformer_loftr import LocalFeatureTransformer
from .muxnet import muxnet_m
from .vit import Transformer

from .resnet import resnet122 as resnet122_cifar
from .matching import match_proposals_with_targets

def show_feature_map(img_origin, feature_map, img_name):
    feature_map = feature_map.squeeze(0)
    # print(feature_map.shape)
    fm = torch.abs(feature_map)
    # fmm = F.normalize(torch.sum(fm.mul(fm), dim=0))
    fmm = F.normalize(torch.sum(fm, dim=0))
    print(fmm)
    up = nn.Upsample(scale_factor=32, mode='bilinear')
    origin_fmm = up(fmm.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    # print(origin_fmm.shape)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    # for index in range(1, feature_map_num+1):
    for index in range(1, 2):
        # plt.subplot(row_num, row_num, index)
        # plt.imshow(feature_map[index-1], cmap='hsv')
        # plt.axis('off')
        img = cv2.resize(img_origin.permute(1,2,0).cpu().numpy(), (640, 384))
        plt.imshow(img)
        plt.imshow(origin_fmm.cpu().numpy(), alpha=0.6, cmap='jet')  #alpha设置透明度, cmap可以选择颜色
        # plt.imshow()
        plt.savefig(img_name)
        # imageio.imsave("./feature_map/"+str(index)+".png", feature_map[index-1])
        # plt.imsave(img_name, origin_fmm.cpu().numpy(), cmap='jet')
    plt.show()

class RESA(nn.Module):
    def __init__(self):
        super(RESA, self).__init__()
        self.iter = 5 #cfg.resa.iter
        chan = 64 #cfg.resa.input_channel
        fea_stride = 32 #cfg.backbone.fea_stride
        self.height = 360 // fea_stride + 1 #cfg.img_height // fea_stride
        self.width = 640 // fea_stride #cfg.img_width // fea_stride
        self.alpha = 2.0 #cfg.resa.alpha
        conv_stride = 9 #cfg.resa.conv_stride

        for i in range(self.iter):
            conv_vert1 = nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride//2), groups=1, bias=False)
            conv_vert2 = nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride//2), groups=1, bias=False)

            setattr(self, 'conv_d'+str(i), conv_vert1)
            setattr(self, 'conv_u'+str(i), conv_vert2)

            conv_hori1 = nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride//2, 0), groups=1, bias=False)
            conv_hori2 = nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride//2, 0), groups=1, bias=False)

            setattr(self, 'conv_r'+str(i), conv_hori1)
            setattr(self, 'conv_l'+str(i), conv_hori2)

            idx_d = (torch.arange(self.height) + self.height //
                     2**(self.iter - i)) % self.height
            # print(idx_d)
            setattr(self, 'idx_d'+str(i), idx_d)

            idx_u = (torch.arange(self.height) - self.height //
                     2**(self.iter - i)) % self.height
            setattr(self, 'idx_u'+str(i), idx_u)

            idx_r = (torch.arange(self.width) + self.width //
                     2**(self.iter - i)) % self.width
            setattr(self, 'idx_r'+str(i), idx_r)

            idx_l = (torch.arange(self.width) - self.width //
                     2**(self.iter - i)) % self.width
            setattr(self, 'idx_l'+str(i), idx_l)

    def forward(self, x):
        x = x.clone() #2*128*46*80
        # print(x.shape)
        # print(x.shape)

        for direction in ['d', 'u']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                # print(idx)
                # print(x[..., idx, :].shape)
                x.add_(self.alpha * F.relu(conv(x[..., idx, :])))

        for direction in ['r', 'l']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx])))

        return x

class LaneATT(nn.Module):
    def __init__(self,
                 cfg = None,
                 backbone='resnet34',
                 pretrained_backbone=True,
                 S=72,
                 img_w=640,
                 img_h=360,
                 anchors_freq_path=None,
                 topk_anchors=None,
                 anchor_feat_channels=64,
                 trans_dims=128):
        super(LaneATT, self).__init__()
        # Some definitions
        self.cfg = cfg
        self.feature_extractor, backbone_nb_channels, self.stride = get_backbone(backbone, pretrained_backbone)
        self.img_w = img_w
        self.n_strips = S - 1
        self.n_offsets = S
        self.fmap_h = img_h // self.stride
        fmap_w = img_w // self.stride
        self.fmap_w = fmap_w
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels
        self.trans_dims = trans_dims
        self.flag = 0

        # Anchor angles, same ones used in Line-CNN
        self.left_angles = [72., 60., 49., 39., 30., 22.]
        self.right_angles = [108., 120., 131., 141., 150., 158.]
        self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]

        # Generate anchors
        self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)

        # Filter masks if `anchors_freq_path` is provided
        if anchors_freq_path is not None:
            anchors_mask = torch.load(anchors_freq_path).cpu()
            assert topk_anchors is not None
            ind = torch.argsort(anchors_mask, descending=True)[:topk_anchors]
            self.anchors = self.anchors[ind]
            self.anchors_cut = self.anchors_cut[ind]

        # Pre compute indices for the anchor pooling
        self.cut_zs, self.cut_ys, self.cut_xs, self.invalid_mask = self.compute_anchor_cut_indices(
            self.anchor_feat_channels, fmap_w, self.fmap_h)

        # Setup and initialize layers
        self.resa = RESA()
        self.vit = Transformer(dim=1280, depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1)
        # self.pos_embedding = nn.Parameter(torch.randn(1, 12*20, 1280))
        self.pos_embedding = build_position_encoding(1280, shape=(self.cfg['batch_size'], self.cfg['pos_shape_h'], self.cfg['pos_shape_w'])).cuda()
        self.trans_loftr = LocalFeatureTransformer(self.cfg)
        self.trans = TransConvEncoderModule(attn_in_dims=[backbone_nb_channels, self.trans_dims], attn_out_dims=[self.trans_dims, self.anchor_feat_channels], pos_shape=(self.cfg['batch_size'], self.cfg['pos_shape_h'], self.cfg['pos_shape_w']))
        self.trans_new = TransConvEncoderModule(attn_in_dims=[anchor_feat_channels, self.trans_dims], attn_out_dims=[self.trans_dims, self.anchor_feat_channels], pos_shape=(self.cfg['batch_size'], self.cfg['pos_shape_h'], self.cfg['pos_shape_w']))
        self.conv1 = nn.Conv2d(backbone_nb_channels, self.anchor_feat_channels, kernel_size=1)
        self.cls_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, 2)
        self.reg_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, self.n_offsets + 1)
        self.cls_layer_add = nn.Linear(3 * self.anchor_feat_channels * self.fmap_h, 2)
        self.reg_layer_add = nn.Linear(3 * self.anchor_feat_channels * self.fmap_h, self.n_offsets + 1)
        self.attention_layer = nn.Linear(self.anchor_feat_channels * self.fmap_h, len(self.anchors) - 1)
        self.initialize_layer(self.attention_layer)
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)

    def forward(self, x, conf_threshold=None, nms_thres=0, nms_topk=3000):
        # print(x.shape)
        if self.cfg['batch_size'] == 1:
            img_origin = x.squeeze(0)
        batch_features = self.feature_extractor(x)
        b, d, _, _= x.shape
        # if self.flag == 0:
        #     show_feature_map(img_origin, batch_features, "./feature_map/featrue_origin_cat_muxnet.png")
        # self.flag += 1
        # print(batch_features.shape)
        if self.cfg['trans_new_new']:
            trans_new_batch_features = self.trans(batch_features)
        if self.cfg['trans']:
            batch_features = self.trans(batch_features)
            # if self.flag == 0:
            #     show_feature_map(img_origin, batch_features, "./feature_map/featrue_trans_cat.png")
            # self.flag += 1
        elif self.cfg['trans_loftr']:
            batch_features = self.conv1(batch_features) 
            batch_features = self.trans_loftr(batch_features)
        elif self.cfg['vit']:
            batch_features = batch_features + self.pos_embedding[:b, :, :, :]
            batch_features = batch_features.view(-1, 1280, 12*20).permute(0,2,1)
            batch_features = self.vit(batch_features)
            batch_features = batch_features.permute(0,2,1).reshape(-1, 1280, 12, 20)
            batch_features = self.conv1(batch_features) 
        else:
            batch_features = self.conv1(batch_features) #减小特征维数
            # if self.flag == 0:
            #     show_feature_map(img_origin, batch_features, "./feature_map/featrue_origin_cat.png")
            # self.flag += 1
        # print(batch_features.shape)
        batch_anchor_features = self.cut_anchor_features(batch_features) # 4*1000*64*11*1
        # print(batch_anchor_features.shape)
        if self.cfg['resa']:
            # Generate RESA features
            resa_batch_features  = self.resa(batch_features)
            # if self.flag == 0:
            #     show_feature_map(resa_batch_features, "./feature_map/featrue_resa.png")
            # self.flag += 1
            resa_anchor_featrues = self.cut_anchor_features(resa_batch_features)

            # Join proposals from all images into a single proposals features batch
            batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h) #4000*704
            # print(batch_anchor_features.shape)
            resa_anchor_featrues = resa_anchor_featrues.view(-1, self.anchor_feat_channels * self.fmap_h)
            batch_anchor_features = torch.cat((resa_anchor_featrues, batch_anchor_features), dim=1)
        
        elif self.cfg['trans_new']:
            # Generate RESA features
            trans_batch_features  = self.trans_new(batch_features)
            # if self.flag == 0:
            #     show_feature_map(resa_batch_features, "./feature_map/featrue_resa.png")
            # self.flag += 1
            trans_anchor_featrues = self.cut_anchor_features(trans_batch_features)

            # Join proposals from all images into a single proposals features batch
            batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h) #4000*704
            # print(batch_anchor_features.shape)
            trans_anchor_featrues = trans_anchor_featrues.view(-1, self.anchor_feat_channels * self.fmap_h)
            batch_anchor_features = torch.cat((trans_anchor_featrues, batch_anchor_features), dim=1)
        
        if self.cfg['trans_new_new']:
            trans_new_anchor_featrues = self.cut_anchor_features(trans_new_batch_features)

            # Join proposals from all images into a single proposals features batch
            batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h) #4000*704
            # print(batch_anchor_features.shape)
            trans_new_anchor_featrues = trans_new_anchor_featrues.view(-1, self.anchor_feat_channels * self.fmap_h)
            batch_anchor_features = torch.cat((trans_new_anchor_featrues, batch_anchor_features), dim=1)            
            

        elif self.cfg['add_resa']:
            # Generate RESA features
            resa_batch_features  = self.resa(batch_features)
            resa_anchor_featrues = self.cut_anchor_features(resa_batch_features)

            # Join proposals from all images into a single proposals features batch
            batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h) #4000*704
            # print(batch_anchor_features.shape)
            resa_anchor_featrues = resa_anchor_featrues.view(-1, self.anchor_feat_channels * self.fmap_h)

            # Add attention features
            softmax = nn.Softmax(dim=1)
            scores = self.attention_layer(batch_anchor_features) #4000*999
            attention = softmax(scores).reshape(x.shape[0], len(self.anchors), -1) #4*1000*999
            attention_matrix = torch.eye(attention.shape[1], device=x.device).repeat(x.shape[0], 1, 1) #4*1000*1000
            non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False) #3996000*3
            attention_matrix[:] = 0
            attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten() #3996000
            batch_anchor_features = batch_anchor_features.reshape(x.shape[0], len(self.anchors), -1) #4*1000#704(64*11)
            attention_features = torch.bmm(torch.transpose(batch_anchor_features, 1, 2),
                                           torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)
            attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
            batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
            batch_anchor_features = torch.cat((resa_anchor_featrues, attention_features, batch_anchor_features), dim=1)


        else:
            # Join proposals from all images into a single proposals features batch
            batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h) #4000*704

            # Add attention features
            softmax = nn.Softmax(dim=1)
            scores = self.attention_layer(batch_anchor_features) #4000*999
            attention = softmax(scores).reshape(x.shape[0], len(self.anchors), -1) #4*1000*999
            attention_matrix = torch.eye(attention.shape[1], device=x.device).repeat(x.shape[0], 1, 1) #4*1000*1000
            non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False) #3996000*3
            attention_matrix[:] = 0
            attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten() #3996000
            batch_anchor_features = batch_anchor_features.reshape(x.shape[0], len(self.anchors), -1) #4*1000#704(64*11)
            attention_features = torch.bmm(torch.transpose(batch_anchor_features, 1, 2),
                                           torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)
            attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
            batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
            batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)

        # Predict
        if self.cfg['add_resa']:
            cls_logits = self.cls_layer_add(batch_anchor_features)
            reg = self.reg_layer_add(batch_anchor_features) 
        else:
            cls_logits = self.cls_layer(batch_anchor_features)
            reg = self.reg_layer(batch_anchor_features)

        # Undo joining
        cls_logits = cls_logits.reshape(x.shape[0], -1, cls_logits.shape[1])
        reg = reg.reshape(x.shape[0], -1, reg.shape[1])

        # Add offsets to anchors
        reg_proposals = torch.zeros((*cls_logits.shape[:2], 5 + self.n_offsets), device=x.device)
        reg_proposals += self.anchors
        reg_proposals[:, :, :2] = cls_logits
        reg_proposals[:, :, 4:] += reg

        # Apply nms
        proposals_list = self.nms(reg_proposals, nms_thres, nms_topk, conf_threshold)

        return proposals_list

    def nms_old(self, batch_proposals, batch_attention_matrix, nms_thres, nms_topk, conf_threshold):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals, attention_matrix in zip(batch_proposals, batch_attention_matrix):
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            with torch.no_grad():
                scores = softmax(proposals[:, :2])[:, 1]
                if conf_threshold is not None:
                    # apply confidence threshold
                    above_threshold = scores > conf_threshold
                    proposals = proposals[above_threshold]
                    scores = scores[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append((proposals[[]], self.anchors[[]], attention_matrix[[]], None))
                    continue
                keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
                keep = keep[:num_to_keep]
            proposals = proposals[keep]
            anchor_inds = anchor_inds[keep]
            attention_matrix = attention_matrix[anchor_inds]
            proposals_list.append((proposals, self.anchors[keep], attention_matrix, anchor_inds))

        return proposals_list

    def nms(self, batch_proposals, nms_thres, nms_topk, conf_threshold):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals in batch_proposals:
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            with torch.no_grad():
                scores = softmax(proposals[:, :2])[:, 1]
                if conf_threshold is not None:
                    # apply confidence threshold
                    above_threshold = scores > conf_threshold
                    proposals = proposals[above_threshold]
                    scores = scores[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append((proposals[[]], self.anchors[[]], 1, None))
                    continue
                keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
                keep = keep[:num_to_keep]
            proposals = proposals[keep]
            anchor_inds = anchor_inds[keep]
            # attention_matrix = attention_matrix[anchor_inds]
            proposals_list.append((proposals, self.anchors[keep], 1, anchor_inds))

        return proposals_list

    def loss(self, proposals_list, targets, cls_loss_weight=10):
        if self.cfg['ghm']:
            ghm_loss = GHMC()
        else:
            focal_loss = FocalLoss(alpha=0.25, gamma=2.)
        smooth_l1_loss = nn.SmoothL1Loss()
        cls_loss = 0
        reg_loss = 0
        valid_imgs = len(targets)
        total_positives = 0
        for (proposals, anchors, _, _), target in zip(proposals_list, targets):
            # Filter lanes that do not exist (confidence == 0)
            target = target[target[:, 1] == 1]
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                if self.cfg['ghm']:
                    cls_label_weight = torch.ones_like(cls_target)
                    cls_loss += ghm_loss(cls_pred, cls_target, cls_label_weight).sum()
                else:
                    cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue
            # Gradients are also not necessary for the positive & negative matching
            with torch.no_grad():
                positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices = match_proposals_with_targets(
                    self, anchors, target)

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)

            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                if self.cfg['ghm']:
                    cls_label_weight = torch.ones_like(cls_target)
                    cls_loss += ghm_loss(cls_pred, cls_target, cls_label_weight).sum()
                else:
                    cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue

            # Get classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = 1.
            cls_pred = all_proposals[:, :2]

            # Regression targets
            reg_pred = positives[:, 4:]
            with torch.no_grad():
                target = target[target_positives_indices] #num_pos * 77
                positive_starts = (positives[:, 2] * self.n_strips).round().long()
                target_starts = (target[:, 2] * self.n_strips).round().long()
                target[:, 4] -= positive_starts - target_starts  #？？why
                all_indices = torch.arange(num_positives, dtype=torch.long)
                ends = (positive_starts + target[:, 4] - 1).round().long()
                invalid_offsets_mask = torch.zeros((num_positives, 1 + self.n_offsets + 1), #pos*73
                                                   dtype=torch.int)  # length + S + pad
                invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
                invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
                invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                invalid_offsets_mask[:, 0] = False
                reg_target = target[:, 4:]
                reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask] #将invalid的部分置为一样

            # Loss calc
            reg_loss += smooth_l1_loss(reg_pred, reg_target)
            if self.cfg['ghm']:
                cls_label_weight = torch.ones_like(cls_target)
                cls_loss += ghm_loss(cls_pred, cls_target, cls_label_weight).sum() / num_positives
            else:
                cls_loss += focal_loss(cls_pred, cls_target).sum() / num_positives
            # cls_loss += focal_loss(cls_pred, cls_target).sum() / num_positives

        # Batch mean
        cls_loss /= valid_imgs
        reg_loss /= valid_imgs

        loss = cls_loss_weight * cls_loss + reg_loss
        return loss, {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'batch_positives': total_positives}

    def compute_anchor_cut_indices(self, n_fmaps, fmaps_w, fmaps_h):
        # definitions
        n_proposals = len(self.anchors_cut) #1000

        # indexing
        unclamped_xs = torch.flip((self.anchors_cut[:, 5:] / self.stride).round().long(), dims=(1,))
        unclamped_xs = unclamped_xs.unsqueeze(2)
        unclamped_xs = torch.repeat_interleave(unclamped_xs, n_fmaps, dim=0).reshape(-1, 1)
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
        unclamped_xs = unclamped_xs.reshape(n_proposals, n_fmaps, fmaps_h, 1)
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)
        cut_ys = torch.arange(0, fmaps_h)
        cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(n_proposals, n_fmaps, fmaps_h)
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = torch.arange(n_fmaps).repeat_interleave(fmaps_h).repeat(n_proposals)[:, None] #1000*64*11

        return cut_zs, cut_ys, cut_xs, invalid_mask

    def cut_anchor_features(self, features):
        # definitions
        batch_size = features.shape[0]
        n_proposals = len(self.anchors)
        n_fmaps = features.shape[1]
        batch_anchor_features = torch.zeros((batch_size, n_proposals, n_fmaps, self.fmap_h, 1), device=features.device)
        # actual cutting
        for batch_idx, img_features in enumerate(features):
            rois = img_features[self.cut_zs, self.cut_ys, self.cut_xs].view(n_proposals, n_fmaps, self.fmap_h, 1)
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut = self.generate_side_anchors(self.left_angles, x=0., nb_origins=lateral_n)
        right_anchors, right_cut = self.generate_side_anchors(self.right_angles, x=1., nb_origins=lateral_n)
        bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)

        return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat([left_cut, bottom_cut, right_cut])

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = nb_origins * len(angles)

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 lenght, S coordinates, score[0] = negative prob, score[1] = positive prob
        anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut

    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[2] = 1 - start_y
        anchor[3] = start_x
        anchor[5:] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w

        return anchor

    def draw_anchors(self, img_w, img_h, k=None):
        base_ys = self.anchor_ys.numpy()
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        i = -1
        for anchor in self.anchors:
            i += 1
            if k is not None and i != k:
                continue
            anchor = anchor.numpy()
            xs = anchor[5:]
            ys = base_ys * img_h
            points = np.vstack((xs, ys)).T.round().astype(int)
            for p_curr, p_next in zip(points[:-1], points[1:]):
                img = cv2.line(img, tuple(p_curr), tuple(p_next), color=(0, 255, 0), thickness=5)

        return img

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def proposals_to_pred(self, proposals):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for lane in proposals:
            lane_xs = lane[5:] / self.img_w
            start = int(round(lane[2].item() * self.n_strips))
            length = int(round(lane[4].item()))
            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) &
                       (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def decode(self, proposals_list, as_lanes=False):
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals, _, _, _ in proposals_list:
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            if as_lanes:
                pred = self.proposals_to_pred(proposals)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded

    def cuda(self, device=None):
        cuda_self = super().cuda(device)
        cuda_self.anchors = cuda_self.anchors.cuda(device)
        cuda_self.anchor_ys = cuda_self.anchor_ys.cuda(device)
        cuda_self.cut_zs = cuda_self.cut_zs.cuda(device)
        cuda_self.cut_ys = cuda_self.cut_ys.cuda(device)
        cuda_self.cut_xs = cuda_self.cut_xs.cuda(device)
        cuda_self.invalid_mask = cuda_self.invalid_mask.cuda(device)
        return cuda_self

    def to(self, *args, **kwargs):
        device_self = super().to(*args, **kwargs)
        device_self.anchors = device_self.anchors.to(*args, **kwargs)
        device_self.anchor_ys = device_self.anchor_ys.to(*args, **kwargs)
        device_self.cut_zs = device_self.cut_zs.to(*args, **kwargs)
        device_self.cut_ys = device_self.cut_ys.to(*args, **kwargs)
        device_self.cut_xs = device_self.cut_xs.to(*args, **kwargs)
        device_self.invalid_mask = device_self.invalid_mask.to(*args, **kwargs)
        return device_self


def get_backbone(backbone, pretrained=False):
    if backbone == 'resnet122':
        backbone = resnet122_cifar()
        fmap_c = 64
        stride = 4
    elif backbone == 'resnet34':
        backbone = torch.nn.Sequential(*list(resnet34(pretrained=pretrained).children())[:-2])
        fmap_c = 512
        stride = 32
    elif backbone == 'resnet18':
        backbone = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-2])
        fmap_c = 512
        stride = 32
    elif backbone == 'mobilenetv2':
        backbone = torch.nn.Sequential(*list(mobilenet_v2(pretrained=pretrained).children())[:-1])
        fmap_c = 1280
        stride = 32
    elif backbone == 'mnasnet1_0':
        backbone = torch.nn.Sequential(*list(mnasnet1_0(pretrained=pretrained).children())[:-1])
        fmap_c = 1280
        stride = 32
    elif backbone == 'muxnet':
        backbone = torch.nn.Sequential(*list(muxnet_m(pretrained=pretrained).children())[:-2])
        fmap_c = 1280
        stride = 32
    elif backbone == 'shufflenet_v2_x1_0':
        backbone = torch.nn.Sequential(*list(shufflenet_v2_x1_0(pretrained=pretrained).children())[:-1])
        fmap_c = 1024
        stride = 32       
    else:
        raise NotImplementedError('Backbone not implemented: `{}`'.format(backbone))

    return backbone, fmap_c, stride
