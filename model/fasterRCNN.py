import torch
import torchvision

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

import math

from torchvision.transforms import transforms

from config.config import config

def get_iou(boxes1,boxes2):
    # boxes1 : (number of boxes1, 4)
    # boxes2 : (number of boxes2, 4)
    # return : (number of boxes1, number of boxes2)
    
    # area of boxes (x2-x1)*(y2-y1)
    area1 = (boxes1[:,2]-boxes1[:,0])*(boxes1[:,3]-boxes1[:,1])
    area2 = (boxes2[:,2]-boxes2[:,0])*(boxes2[:,3]-boxes2[:,1])

    # get the coordinates of the intersection rectangle
    x1 = torch.max(boxes1[:,None,0],boxes2[:,0])
    y1 = torch.max(boxes1[:,None,1],boxes2[:,1])
    x2 = torch.min(boxes1[:,None,2],boxes2[:,2])
    y2 = torch.min(boxes1[:,None,3],boxes2[:,3])
    
    # compute the area of intersection rectangle
    intersection = (x2-x1).clamp(min=0)*(y2-y1).clamp(min=0)

    # compute the area of union
    union = (area1[:,None]+area2)-intersection

    # compute the IOU
    iou = intersection/union

    return iou

def sample_pos_neg(labels, positive_count, total_count):
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]

    num_pos = min(positive_count, positive.numel())
    num_neg = min(total_count - num_pos, negative.numel())

    perm_pos_idx = torch.randperm(positive.numel(),device=positive.device)[:num_pos]
    perm_neg_idx = torch.randperm(negative.numel(),device=negative.device)[:num_neg]
    pos_idx = positive[perm_pos_idx]
    neg_idx = negative[perm_neg_idx]
    sampled_pos_idx_mask = torch.zeros_like(labels, dtype = torch.bool)
    sampled_neg_idx_mask = torch.zeros_like(labels, dtype = torch.bool)
    sampled_pos_idx_mask[pos_idx] = True
    sampled_neg_idx_mask[neg_idx] = True

    return sampled_pos_idx_mask, sampled_neg_idx_mask

def apply_regression(reg_pred, anchors_or_proposals):
    # reg_pred : (number of anchors = batch_size * (9 * 38 * 50 = 17100), 1, 4)
    # anchors_or_proposals : # (number_of_anchors = 38 * 50 * 9 = 17100, 4)

    reg_pred = reg_pred.reshape(reg_pred.size(0), -1, 4) # (batch_size * (9 * 38 * 50 = 17100), 1, 4)

    w = anchors_or_proposals[:,2] - anchors_or_proposals[:,0] # (number of anchors = 38 * 50 * 9 = 17100,)
    h = anchors_or_proposals[:,3] - anchors_or_proposals[:,1] # (number of anchors = 38 * 50 * 9 = 17100,)
    ctr_x = anchors_or_proposals[:,0] + 0.5 * w # (number of anchors = 38 * 50 * 9 = 17100,)
    ctr_y = anchors_or_proposals[:,1] + 0.5 * h # (number of anchors = 38 * 50 * 9 = 17100,)


    dx = reg_pred[...,0] # takes the 0th element of the last dimension,  
                            # (number of anchors = batch_size * (9 * 38 * 50 = 17100), 1,)
    dy = reg_pred[...,1] # (number of anchors = batch_size * (9 * 38 * 50 = 17100), 1,)
    dw = reg_pred[...,2] # (number of anchors = batch_size * (9 * 38 * 50 = 17100), 1,)
    dh = reg_pred[...,3] # (number of anchors = batch_size * (9 * 38 * 50 = 17100), 1,)

    # beacause of this ??
    dw = torch.clamp(dw, max=math.log(1000.0 / 16))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16))

    pred_ctr_x = dx * w[:, None] + ctr_x[:, None] # result = [[(dx[0] * w[0]) + ctr_x[0]],
                                                    #           [(dx[1] * w[1]) + ctr_x[1]],
                                                    #           [(dx[2] * w[2]) + ctr_x[2]],
                                                    #           ...,
                                                    #           [(dx[16999] * w[16999]) + ctr_x[16999]]]
                                                    # (number of anchors = 38 * 50 * 9 = 17100, 1,)
    pred_ctr_y = dy * h[:, None] + ctr_y[:, None] # (number of anchors = 38 * 50 * 9 = 17100, 1,)
    pred_w = torch.exp(dw) * w[:, None] # (number of anchors = 38 * 50 * 9 = 17100, 1,)
    pred_h = torch.exp(dh) * h[:, None] # (number of anchors = 38 * 50 * 9 = 17100, 1,)

    pred_x1 = pred_ctr_x - 0.5 * pred_w # (number of anchors = 38 * 50 * 9 = 17100, 1,)
    pred_y1 = pred_ctr_y - 0.5 * pred_h # (number of anchors = 38 * 50 * 9 = 17100, 1,)
    pred_x2 = pred_ctr_x + 0.5 * pred_w # (number of anchors = 38 * 50 * 9 = 17100, 1,)
    pred_y2 = pred_ctr_y + 0.5 * pred_h # (number of anchors = 38 * 50 * 9 = 17100, 1,)

    pred_boxes = torch.stack((pred_x1, pred_y1, pred_x2, pred_y2), dim = 2) # (number of anchors = 38 * 50 * 9 = 17100, 1, 4)

    return pred_boxes

def clamp_boxes(boxes, img_shape):
    boxes_x1 = boxes[..., 0].clamp(min = 0, max = img_shape[-1]) # (10000,)
    boxes_y1 = boxes[..., 1].clamp(min = 0, max = img_shape[-2]) # (10000,)
    boxes_x2 = boxes[..., 2].clamp(min = 0, max = img_shape[-1]) # (10000,)
    boxes_y2 = boxes[..., 3].clamp(min = 0, max = img_shape[-2]) # (10000,)
    boxes = torch.cat((boxes_x1[...,None], boxes_y1[...,None], boxes_x2[...,None], boxes_y2[...,None]), dim = -1) # (10000, 4)
    return boxes

def get_regression(gt_boxes, anchors_or_proposals):
    # get ctr_x, ctr_y, w, h from x1, y1, x2, y2 for anchors
    w = anchors_or_proposals[:,2] - anchors_or_proposals[:,0]
    h = anchors_or_proposals[:,3] - anchors_or_proposals[:,1]
    ctr_x = anchors_or_proposals[:,0] + 0.5 * w
    ctr_y = anchors_or_proposals[:,1] + 0.5 * h

    # get ctr_x, ctr_y, w, h from x1, y1, x2, y2 for gt_boxes
    gt_w = gt_boxes[:,2] - gt_boxes[:,0]
    gt_h = gt_boxes[:,3] - gt_boxes[:,1]
    gt_ctr_x = gt_boxes[:,0] + 0.5 * gt_w
    gt_ctr_y = gt_boxes[:,1] + 0.5 * gt_h

    targets_dx = (gt_ctr_x - ctr_x) / w
    targets_dy = (gt_ctr_y - ctr_y) / h
    targets_dw = torch.log(gt_w / w)
    targets_dh = torch.log(gt_h / h)

    regression_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim = 1)

    return regression_targets
    
class RPN(torch.nn.Module):
    def __init__(self, device):
        super(RPN, self).__init__()

        # anchor parameters
        self.anchor_scale = config.RPN_ANCHOR_SCALES
        self.anchor_ratio = config.RPN_ANCHOR_RATIOS

        self.anchor_number = len(self.anchor_scale) * len(self.anchor_ratio)

        # 3*3 conv layer with 512 filters
        self.conv_layer = torch.nn.Conv2d(in_channels = config.RPN_CONV_CHANNELS, 
                                    out_channels = config.RPN_CONV_CHANNELS, 
                                    kernel_size = 3, 
                                    stride = 1, 
                                    padding = 1) 
        
        # 1*1 conv layer with 18 filters, 18 = 2 (forground / background) * 9 (number of anchors)
        self.cls_layer = torch.nn.Conv2d(in_channels = config.RPN_CONV_CHANNELS, 
                                    out_channels = self.anchor_number * 2, 
                                    kernel_size = 1, 
                                    stride = 1, 
                                    padding = 0)
        
        # 1*1 conv layer with 36 filters, 36 = 4 (coordinates) * 9 (number of anchors)
        self.reg_layer = torch.nn.Conv2d(in_channels = config.RPN_CONV_CHANNELS, 
                                    out_channels = self.anchor_number * 4, 
                                    kernel_size = 1, 
                                    stride = 1, 
                                    padding = 0)
        
        for layer in [self.conv_layer, self.cls_layer, self.reg_layer]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

        self.device = device
        
    def generate_anchors(self, feature_map):
        # anchor for a single pixel
        scale = torch.as_tensor(self.anchor_scale)
        ratio = torch.as_tensor(self.anchor_ratio)
        ratio_w = torch.sqrt(ratio)
        ratio_h = 1 / ratio_w
        widths = (ratio_w[:, None] * scale[None, :]).view(-1)
        heights = (ratio_h[:, None] * scale[None, :]).view(-1)
        shapes = torch.stack([-widths, -heights, widths, heights], dim = 1) / 2 # /2 for center to corner
        shapes = shapes.round()
        # (9, 4)

        # anchors for all pixels in the feature map, yet with its size adjusted to the image size
        shift_w = torch.arange(0, feature_map.shape[-1]) * config.FESTURE_MAP_STRIDE + config.FESTURE_MAP_STRIDE/2*config.RPN_CENTRALIZE_ANCHORS
        shift_h = torch.arange(0, feature_map.shape[-2]) * config.FESTURE_MAP_STRIDE + config.FESTURE_MAP_STRIDE/2*config.RPN_CENTRALIZE_ANCHORS
        shift_h, shift_w = torch.meshgrid(shift_h, shift_w,indexing='ij')

        shift_w = shift_w.reshape(-1)
        shift_h = shift_h.reshape(-1)
        shifts = torch.stack([shift_w, shift_h, shift_w, shift_h], dim = 1)
        # (feature_map.height * feature_map.width, 4)

        # combine single pixel anchors and shifts
        anchors = (shifts.view(-1,1,4) + shapes.view(1,-1,4)).reshape(-1,4) # (feature_map.height * feature_map.width * 9, * 4)

        # # draw anchors on the image and save the image
        # fig, ax = plt.subplots(1)
        # ax.imshow(torch.zeros((feature_map.shape[-2]*16, feature_map.shape[-1]*16)).numpy(), cmap = 'gray')
        # for box in anchors[:9]:
        #     rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth = 1, edgecolor = 'r', facecolor = 'none')
        #     ax.add_patch(rect)
        # plt.savefig('anchors.png')
        # plt.close(fig)

        return anchors
        ### the anchors are not at the center !! (cf 13:54)

    def filter_proposals(self, proposals, cls_pred, img_shape):

        # cls_pred = cls_pred.reshape(-1,2) # ((batch_size = 1) * feature_map.height * feature_map.width * 9 * 2)
        # cls_pred = cls_pred[:,1]
        cls_pred = cls_pred.reshape(-1)
        # print("\n in filter_proposals, before sigmoid, cls_pred", cls_pred[:20])
        # cls_pred = torch.sigmoid(cls_pred) # ((batch_size = 1) * feature_map.height * feature_map.width * 9 * 2)
                                           # sigmoid is used to convert the output to a probability
        # print("\n in filter_proposals, after sigmoid, cls_pred", cls_pred[:20])

        # if self.training:
        # choose only the proposals that doesn't cross the image boundary
        # img_height, img_width = img_shape
    
        # Create a mask for valid proposals
        # valid_mask = (
        #     (proposals[:, 0] >= 0) &           # x1 >= 0
        #     (proposals[:, 1] >= 0) &           # y1 >= 0
        #     (proposals[:, 2] <= img_width) &   # x2 <= image width
        #     (proposals[:, 3] <= img_height)    # y2 <= image height
        # )
        
        # Apply the mask to filter proposals and class predictions
        # proposals = proposals[valid_mask]
        # cls_pred = cls_pred[valid_mask]

        # if not self.training:
        #     # clamp the proposals
        #     proposals = clamp_boxes(proposals, img_shape[-2:]) # (10000, 4)
        proposals = clamp_boxes(proposals, img_shape[-2:]) # (10000, 4)


        # remove the super small boxes, cf 知乎专栏
        min_size = config.RPN_MIN_PROPOSAL_SIZE
        ws, hs = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        proposals = proposals[keep]
        cls_pred = cls_pred[keep]


        if self.training:
            prenms_topk = config.RPN_PRE_NMS_TOP_N_TRAIN
            topk = config.RPN_NMS_TOP_N_TRAIN
        else:
            prenms_topk = config.RPN_PRE_NMS_TOP_N_TEST
            topk = config.RPN_NMS_TOP_N_TEST

        _, top_n_idx = torch.topk(cls_pred, min(prenms_topk,len(cls_pred))) # (10000,)
                                                   # topk returns (values, indices)
        cls_pred = cls_pred[top_n_idx] # (10000,)
        proposals = proposals[top_n_idx] # (10000, 4)


        # NMS based on objectness score
        # Non-Maximum Suppression, 非极大值抑制, 删除重叠过多的候选框
        keep_mask = torch.zeros_like(cls_pred, dtype = torch.bool) # (10000,) A boolean mask initialized to False values
        keep_indices = torchvision.ops.nms(proposals, cls_pred, config.RPN_NMS_THRESHOLD) # The indices of the proposals that survived non-maximum suppression
        
        # because of this ??
        keep_mask[keep_indices] = True
        keep_indices = torch.where(keep_mask)[0]


        keep_indices_after_nms = keep_indices[cls_pred[keep_indices].sort(descending = True)[1]]
        
        
        proposals = proposals[keep_indices_after_nms[:topk]]
        cls_pred = cls_pred[keep_indices_after_nms[:topk]]

        return proposals, cls_pred

    def assign_targets(self, anchors, gt_boxes):
        iou_matrix = get_iou(gt_boxes.to(self.device), anchors.to(self.device)) # (number of gt_boxes, number of anchors)

        # for each anchor get best gt box index
        best_gt_for_anchor_score, best_gt_for_anchor_idx = iou_matrix.max(dim = 0)

        # this will be used to get anchor that has an IOU > 0.7 with a gt box
        # we make a copy because later we need to add also the anchors with the best IOU with a gt box
        best_match_gt_idx_before_threshold = best_gt_for_anchor_idx.clone()

        below_low_threshold = best_gt_for_anchor_score < config.RPN_LOW_THRESHOLD # a list of boolean values
        between_thresholds = (best_gt_for_anchor_score >= config.RPN_LOW_THRESHOLD) & (best_gt_for_anchor_score < config.RPN_HIGH_THRESHOLD)
        best_gt_for_anchor_idx[below_low_threshold] = -1
        best_gt_for_anchor_idx[between_thresholds] = -2

        # low quality anchor boxes
        best_anchor_for_gt_score, _ = iou_matrix.max(dim = 1)
        gt_anchor_pair_with_best_iou = torch.where(iou_matrix == best_anchor_for_gt_score[:,None])

        # get the anchor indeces to update
        idx_to_update = gt_anchor_pair_with_best_iou[1]
        best_gt_for_anchor_idx[idx_to_update] = best_match_gt_idx_before_threshold[idx_to_update]

        # by now the best best_gt_for_anchor_idx is : valid / -1(bg) / -2(to ignore)
        matched_gt_boxes = gt_boxes[best_gt_for_anchor_idx.clamp(min=0)]

        # set all foreground anchor labels to 1
        labels = best_gt_for_anchor_idx >= 0
        labels = labels.to(torch.float32)

        # set all background anchor labels to 0
        bg_anchors = best_gt_for_anchor_idx == -1
        labels[bg_anchors] = 0.0

        # set all ignore anchor labels to -1
        ignore_anchors = best_gt_for_anchor_idx == -2
        labels[ignore_anchors] = -1.0


        # print number of positive, negative, ignore anchors
        # num_pos = torch.sum(labels == 1)
        # num_neg = torch.sum(labels == 0)
        # num_ignore = torch.sum(labels == -1)
        # print("\n [in RPN] num_pos, num_neg, num_ignore", num_pos, num_neg, num_ignore)

        # later for classification we pick labels which have >= 0
        return labels, matched_gt_boxes
    

    def forward(self, img, feature_map, targets):

        # img : (batch_size = 1, channel_number = 3, height = img.height, width = img.width)
        # feature_map : (batch_size = 1, channel_number = 512, height = img.height / 16, width = img.width / 16)
        # targets['bboxes'] : (batch_size = 1, number of gt_boxes, 4)
        # targets['labels'] : (batch_size = 1, number of gt_boxes)

        feature_map_ReLU = torch.nn.ReLU()(self.conv_layer(feature_map))
        cls_pred = self.cls_layer(feature_map_ReLU) # (batch_size = 1, 9*2, height = img.height / 16, width = img.width / 16)
        reg_pred = self.reg_layer(feature_map_ReLU) # (batch_size = 1, 9*4, height = img.height / 16, width = img.width / 16)

        # generate anchors
        anchors = self.generate_anchors(feature_map) # (feature_map.height * feature_map.width * 9, 4)

        # reshape cls_pred from (batch_size = 1, 9*2, feature_map.height, feature_map.width) 
        # to ((batch_size = 1) * feature_map.height * feature_map.width * 9 * 2, 1)
        cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(-1, 1)

        # reshape reg_pred from (batch_size = 1, 9*4, feature_map.height, feature_map.width) 
        # to ((batch_size = 1) * 9 * feature_map.height * feature_map.width, 4)
        reg_pred = reg_pred.view(reg_pred.size(0),self.anchor_number,4,feature_map.shape[-2],feature_map.shape[-1]).permute(0, 3, 4, 1, 2).reshape(-1, 4)

        # print("\n [in RPN] cls_pred.shape, reg_pred.shape", cls_pred.shape, reg_pred.shape)
        # apply regression to anchors, what we get is called proposals
        proposals = apply_regression(reg_pred.detach().reshape(-1,1,4).to(self.device), anchors.to(self.device)) # (feature_map.height * feature_map.width * 9, 1, 4)
        proposals = proposals.reshape(proposals.size(0),4) # (number of anchors, 4)
                                                        # number of anchors  = number of proposals at this step
                                                        # proposals are just anchors applied regressions here

        # filter proposals (handle cross)
        # print("in forward, before filter_proposals, cls_pred", cls_pred[:20])

        # 传入filter proposal的是经过softmax，然后选出foreground的分数
        hh, ww = feature_map.shape[-2], feature_map.shape[-1]
        cls_pred_softmax = torch.nn.functional.softmax(cls_pred.view(1,hh,ww,self.anchor_number,2), dim = 4)
        cls_pred_fg = cls_pred_softmax[...,1].view(1,-1)
        cls_pred = cls_pred.view(-1,2)
        proposals, scores = self.filter_proposals(proposals, cls_pred_fg, img.shape[-2:])
        # print("in forward, after filter_proposals, cls_pred(used later in binary cross entropy)", cls_pred[:20])

        rpn_output = {'proposals': proposals, 'scores': scores}

        if not self.training or targets is None:
            return rpn_output
        else:
            # in training
            # assign ground truth boxe and label to each anchor
            labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets(anchors, targets['bboxes'][0])

            # based on gt assignment above, get the regression targets for each anchor
            # matched_gt_boxes_for_anchors : (number of anchors, 4)
            # anchors : (number of anchors, 4)
            regression_targets = get_regression(matched_gt_boxes_for_anchors.to(self.device), anchors.to(self.device))

            # for training, we don't use the entire set of anchors, but only some samples
            sampled_pos_idx_mask, sampled_neg_idx_mask = sample_pos_neg(labels_for_anchors, positive_count=config.RPN_SAMPLE_POSITIVE_NUM, total_count=config.RPN_SAMPLE_TOTAL_NUM)
            # print the number of true values in the mask
            # print("\n [in RPN] torch.sum(sampled_pos_idx_mask), torch.sum(sampled_neg_idx_mask)",torch.sum(sampled_pos_idx_mask), torch.sum(sampled_neg_idx_mask))
            # return 2 tensors, each of size = size of labels_for_anchors, with 128 True in each that marks the positive / negative anchors to be used for training
            sampled_idx = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]

            # cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            #     cls_pred[sampled_idx].flatten(),
            #     labels_for_anchors[sampled_idx].flatten(),
            # ) #/ sampled_idx.numel()
            # # log loss

            # print("cls_pred.shape", cls_pred.shape, "content", cls_pred[:20])
            # print("labels_for_anchors.shape", labels_for_anchors.shape, "content", labels_for_anchors[:20])
            # print("labels type", labels_for_anchors.dtype)

            cls_pred = cls_pred[sampled_idx].view(-1,2)
            labels_for_anchors = labels_for_anchors[sampled_idx].view(-1)
            cls_loss = torch.nn.functional.cross_entropy(cls_pred, labels_for_anchors.long(), ignore_index = -1)

            # print("\n sampled_idx.numel()", sampled_idx.numel())

            localization_loss = (
                torch.nn.functional.smooth_l1_loss(
                    reg_pred[sampled_pos_idx_mask],
                    regression_targets[sampled_pos_idx_mask],
                    beta = 1/9,
                    reduction='sum',
                ) / (sampled_idx.numel())
                # ) / feature_map.shape[-1] / feature_map.shape[-2] * config.RPN_LAMBDA
            )

            # print("\n feature_map.shape[-1] * feature_map.shape[-2]",feature_map.shape[-1] * feature_map.shape[-2])

            # print("\n [in RPN] cls_loss, localization_loss", cls_loss, localization_loss)

            rpn_output['rpn_classification_loss'] = cls_loss
            rpn_output['rpn_localization_loss'] = localization_loss

            return rpn_output



class ROIhead(torch.nn.Module):
    def __init__(self, device):
        super(ROIhead, self).__init__()
        self.in_channels = config.ROI_IN_CHANNELS
        self.num_classes = config.ROI_CLASS_NUM
        self.pool_size = config.ROI_POOL_SIZE
        # self.fc_dim = 1024  # fc stands for fully connected
        self.fc_dim = config.ROI_FC_CHANNELS  # to match the dimensions of VGG fc6 and fc7 layers
    
        # # # [TODO] cf. Fast RCNN 3.1. Truncated SVD for faster detection
        # self.fc6 = torch.nn.Linear(self.in_channels * self.pool_size * self.pool_size, self.fc_dim)
        # self.fc7 = torch.nn.Linear(self.fc_dim, self.fc_dim)

        # Load pretrained VGG16 and extract fc6 and fc7
        # vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        vgg16 = torchvision.models.vgg16(weights=None)  # No weights loaded initially
        state_dict = torch.load(config.BACKBONE_PATH)  # Your converted .pth file
        vgg16.load_state_dict(state_dict, strict=False)  # Load the weights to the model  
        self.fc6 = vgg16.classifier[0]  # Pretrained fc6 from VGG
        self.fc7 = vgg16.classifier[3]  # Pretrained fc7 from VGG

        self.cls_layer = torch.nn.Linear(self.fc_dim, self.num_classes)
        self.reg_layer = torch.nn.Linear(self.fc_dim, self.num_classes * 4)


        torch.nn.init.normal_(self.cls_layer.weight, std=0.01)
        torch.nn.init.constant_(self.cls_layer.bias, 0)

        torch.nn.init.normal_(self.reg_layer.weight, std=0.001)
        torch.nn.init.constant_(self.reg_layer.bias, 0)

        self.device = device
 
    def assign_targets(self, proposals, gt_boxes, gt_labels):

        iou_matrix = get_iou(gt_boxes.to(self.device), proposals.to(self.device))
        best_gt_for_proposal_score, best_gt_for_proposal_idx = iou_matrix.max(dim = 0)
        # gives the most possible class, for each proposal
        # note that the index is the index of gt list, but not the complete 21 classes
        # thus it goes from 0 to number of gt_boxes - 1, with the latter up to 20

        # As in [9], we take ... that have IoU overlap with a groundtruth bounding box of at least 0.5
        # fg_idx = best_gt_for_proposal_score >= 0.5
        # The lower threshold of 0.1 appears to act as a heuristic for hard example mining
        bg_idx = (best_gt_for_proposal_score < config.ROI_HIGH_THRESHOLD) & (best_gt_for_proposal_score >= config.ROI_LOW_THRESHOLD)
        neither_idx = best_gt_for_proposal_score < config.ROI_LOW_THRESHOLD

        # we keep these negative, in order to use best_gt_for_proposal_idx 
        # which goes from 0 to maybe 20, to label the proposals
        best_gt_for_proposal_idx[bg_idx] = -1
        best_gt_for_proposal_idx[neither_idx] = -2

        # same as in "labels = gt_labels[best_gt_for_proposal_idx.clamp(min=0)]"
        # we assign gt_boxes even to bg and neither
        # but we don't use them, we ignore them by referring to labels
        matched_gt_boxes = gt_boxes[best_gt_for_proposal_idx.clamp(min=0)]

        # by using clamp, the bg and neither are labeled as the first fg in gt_labels
        # but now everything goes from 1 to upper
        labels = gt_labels[best_gt_for_proposal_idx.clamp(min=0)]
        labels = labels.to(torch.int64)
        # we set bg to 0
        # bg_proposals = best_gt_for_proposal_idx == -1
        labels[bg_idx] = 0
        # and neither to -1
        # neither_proposals = best_gt_for_proposal_idx == -2
        labels[neither_idx] = -1

        # the returned tensors are of the same length
        return labels, matched_gt_boxes
    
    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        # remove low score boxes
        keep = torch.where(pred_scores > config.ROI_SCORE_THRESHOLD)[0]
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        # remove small boxes
        min_size = config.ROI_MIN_PROPOSAL_SIZE
        w = pred_boxes[:,2] - pred_boxes[:,0]
        h = pred_boxes[:,3] - pred_boxes[:,1]
        keep = torch.where((w >= min_size) & (h >= min_size))[0]
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        # class wise NMS
        keep_mask = torch.zeros_like(pred_scores, dtype = torch.bool)
        for cls_id in torch.unique(pred_labels):
            curr_idx = torch.where(pred_labels == cls_id)[0]
            curr_keep = torchvision.ops.nms(pred_boxes[curr_idx], pred_scores[curr_idx], config.ROI_NMS_THRESHOLD)
            keep_mask[curr_idx[curr_keep]] = True
        keep_idx = torch.where(keep_mask)[0]
        keep_idx_after_nms = keep_idx[pred_scores[keep_idx].sort(descending = True)[1]]
        keep = keep_idx_after_nms[:config.ROI_NMS_TOP_N]
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        return pred_boxes, pred_labels, pred_scores

    def forward(self, feature_map, proposals, image_shape, targets):

        loc_normalize_mean = torch.tensor(config.REG_NOMALIZE_MEAN, dtype=torch.float32, device=self.device)
        loc_normalize_std = torch.tensor(config.REG_NOMALIZE_STD, dtype=torch.float32, device=self.device)
    

        if self.training and targets is not None:
            
            proposals = torch.cat([proposals, targets['bboxes'][0]], dim=0)


            gt_boxes = targets['bboxes'][0]
            gt_labels = targets['labels'][0]

            # assign labels and gt boxes for each proposal
            labels, matched_gt_boxes = self.assign_targets(proposals, gt_boxes, gt_labels)
            # We use mini-batches of size R = 128, 
            # sampling 64 RoIs from each image. 
            # As in [9], we take 25% of the RoIs from object proposals 
            # that have IoU overlap with a groundtruth bounding box of at least 0.5.
            sampled_pos_idx_mask, sampled_neg_idx_mask = sample_pos_neg(labels, positive_count=config.ROI_SAMPLE_POSITIVE_NUM, total_count=config.ROI_SAMPLE_TOTAL_NUM)

            # print sample length
            # print("\n [in ROI head] torch.sum(sampled_pos_idx_mask), torch.sum(sampled_neg_idx_mask)",torch.sum(sampled_pos_idx_mask), torch.sum(sampled_neg_idx_mask))


            sampled_idx = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
            proposals = proposals[sampled_idx]
            labels = labels[sampled_idx]
            matched_gt_boxes = matched_gt_boxes[sampled_idx]
            regression_targets = get_regression(matched_gt_boxes.to(self.device), proposals.to(self.device)) # (sampled_training_proposals, 4)


            regression_targets = ((regression_targets - loc_normalize_mean) / loc_normalize_std)
            # by now we get the "reference" for trainging
            # labels and regression_targets
        
        # ROI pooling (during inference for all proposals, during training for sampled 128 proposals)
        # spatial scale is needed, because the ROIs are defined in terms of original iamge coordinates
        proposal_roi_pool_features = torchvision.ops.roi_pool(feature_map, [proposals], output_size = self.pool_size, spatial_scale = 1/config.FESTURE_MAP_STRIDE).flatten(start_dim = 1)
        # (sampled_training_proposals = 128, 512 * 7 * 7 = 25088)
        ROI_pooling_output = torch.nn.functional.relu(self.fc6(proposal_roi_pool_features))
        ROI_pooling_output = torch.nn.functional.relu(self.fc7(ROI_pooling_output))
        cls_pred = self.cls_layer(ROI_pooling_output) # (sampled_training_proposals = 128, 21)
        reg_pred = self.reg_layer(ROI_pooling_output) # (sampled_training_proposals = 128, 21 * 4 = 84)

        num_boxes, num_classes = cls_pred.shape
        reg_pred = reg_pred.reshape(num_boxes, num_classes, 4) # (sampled_training_proposals = 128, 21, 4)

        frcnn_output = {}
        
        if self.training and targets is not None:
            classification_loss = torch.nn.functional.cross_entropy(cls_pred, labels.long()) # / labels.numel()
            # print("\n labels.numel()", labels.numel())

            # compute localization only for non background 
            fg_proposals_idx = torch.where(labels > 0)[0]
            # get class labels for them
            fg_cls_labels = labels[fg_proposals_idx].long()
            localization_loss = torch.nn.functional.smooth_l1_loss(
                reg_pred[fg_proposals_idx, fg_cls_labels],
                regression_targets[fg_proposals_idx],
                # beta = 1/9,
                beta = 1,
                reduction = 'sum'
            # ) / labels.numel() 
            ) / torch.sum(labels >= 0)

            frcnn_output['frcnn_classification_loss'] = classification_loss
            frcnn_output['frcnn_localization_loss'] = localization_loss
        else:
            # print("\n [in ROIhead] cls_pred.shape, reg_pred.shape", cls_pred.shape, reg_pred.shape)
            # apply regression predictions to proposals
            reg_pred = reg_pred * loc_normalize_std + loc_normalize_mean

            pred_boxes = apply_regression(reg_pred.to(self.device), proposals.to(self.device))

            pred_scores = torch.nn.functional.softmax(cls_pred, dim = -1)

            # clamp boxes to image boundary
            pred_boxes = clamp_boxes(pred_boxes, image_shape[-2:])

            # create labels for each prediction
            pred_labels = torch.arange(num_classes, device = cls_pred.device).view(1, -1).expand_as(pred_scores)

            # remove background class predictions
            pred_boxes = pred_boxes[:,1:] # (number_proposals = 128, 20, 4)
            pred_scores = pred_scores[:,1:]
            pred_labels = pred_labels[:,1:]

            # batch everything by making every class prediction a separate instance
            pred_boxes = pred_boxes.reshape(-1, 4)
            pred_scores = pred_scores.reshape(-1)
            pred_labels = pred_labels.reshape(-1)

            pred_boxes, pred_labels, pred_scores = self.filter_predictions(pred_boxes, pred_labels, pred_scores)

            frcnn_output['bboxes'] = pred_boxes
            frcnn_output['labels'] = pred_labels
            frcnn_output['scores'] = pred_scores

        return frcnn_output



class FasterRCNN(torch.nn.Module):
    def __init__(self, device):
        super(FasterRCNN, self).__init__()

        self.device = device

        # load models
        # vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        vgg16 = torchvision.models.vgg16(weights=None)  # No weights loaded initially
        state_dict = torch.load(config.BACKBONE_PATH)  # Your converted .pth file
        vgg16.load_state_dict(state_dict, strict=False)  # Load the weights to the model
        vgg16 = vgg16.to(device)        

        self.backbone = vgg16.features[:-1]
        self.rpn = RPN(device=self.device)
        self.roi_head = ROIhead(device=self.device)

        # freeze the first 10 layers of the backbone
        for layer in self.backbone[:10]:
            for param in layer.parameters():
                param.requires_grad = False

        # image normalisation parameters
        # self.image_mean = torch.tensor([0.485, 0.456, 0.406])
        # self.image_std = torch.tensor([0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x[[2, 1, 0], ...]),  # Convert RGB to BGR
            transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1, 1, 1]),  # Normalize for Caffe
        ])

        # image resizing parameters
        self.min_size = config.FRCNN_IMG_MIN_SIZE
        self.max_size = config.FRCNN_IMG_MAX_SIZE
    
    def normalise_resize_img_and_boxes(self, img, bboxes):
        # normalise image
        # mean = torch.as_tensor(self.image_mean, dtype = img.dtype, device = img.device)
        # std = torch.as_tensor(self.image_std, dtype = img.dtype, device = img.device)
        # img = (img - mean[:, None, None]) / std[:, None, None]
        # img = transforms.ToTensor()(img) * 255
        img = img * 255
        img = self.transform(img)
        # img = img.unsqueeze(0)

        # resize image
        h, w = img.shape[-2:]
        img_shape = torch.tensor(img.shape[-2:])
        min_size = torch.min(img_shape).to(dtype=torch.float32)
        max_size = torch.max(img_shape).to(dtype=torch.float32)
        scale = torch.min(float(self.min_size) / min_size, float(self.max_size) / max_size)
        scale_factor = scale.item()

        img = torch.nn.functional.interpolate(img, 
                                              size=None, 
                                              scale_factor = scale_factor, 
                                              mode = 'bilinear', 
                                              recompute_scale_factor=True, 
                                              align_corners = False)

        # resize boxes
        if bboxes is not None:
            ratio = [torch.tensor(s,dtype=torch.float32,device=bboxes.device)
                     /torch.tensor(s_ori,dtype=torch.float32,device=img.device) 
                     for s, s_ori in zip(img.shape[-2:], (h,w))]
            ratio_height, ratio_width = ratio
            x_min, y_min, x_max, y_max = bboxes.unbind(2)
            x_min = x_min * ratio_width
            x_max = x_max * ratio_width
            y_min = y_min * ratio_height
            y_max = y_max * ratio_height
            bboxes = torch.stack((x_min, y_min, x_max, y_max), dim = 2)
        return img, bboxes
    
    def transform_boxes_to_original_image_size(self, boxes, new_size, ori_size):
        ratio = [torch.tensor(s_ori,dtype=torch.float32,device=boxes.device)
                 /torch.tensor(s,dtype=torch.float32,device=boxes.device) 
                 for s, s_ori in zip(new_size, ori_size)]
        ratio_height, ratio_width = ratio
        x_min, y_min, x_max, y_max = boxes.unbind(1)
        x_min = x_min * ratio_width
        x_max = x_max * ratio_width
        y_min = y_min * ratio_height
        y_max = y_max * ratio_height
        boxes = torch.stack((x_min, y_min, x_max, y_max), dim = 1)
        return boxes

    '''
    def draw_proposals_on_image(self, img, proposals, target_boxes, string, id_img):
        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        classes = sorted(classes)
        classes = ['background'] + classes
        idx2label = {idx: classes[idx] for idx in range(len(classes))}

        bboxes = proposals['bboxes'].cpu().detach().numpy()
        scores = proposals['scores'].cpu().detach().numpy()
        labels = proposals['labels'].cpu().detach().numpy()

        # Convert the tensor image to numpy (if it's normalized, you may want to denormalize it first)
        img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()

        # Plot the image
        fig, ax = plt.subplots(1)
        ax.imshow(img_np)

    # Check if proposals are empty
        if proposals['bboxes'].numel() > 0:
            # Loop through each proposal and add it to the plot
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = bboxes[i]
                width = x2 - x1
                height = y2 - y1
                rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                # Add the score and label to the plot
                ax.text(x1, y1, f'{idx2label[labels[i]]} {scores[i]:.2f}', color='r')

        else:
            print("No proposals to display")

        # Check if target_boxes are empty
        # print("target_boxes",target_boxes)
        if target_boxes.numel() > 0:
            target_boxes = target_boxes.squeeze(dim=0)
            for box in target_boxes:
                x1, y1, x2, y2 = box.cpu().numpy()
                width = x2 - x1
                height = y2 - y1
                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
        else:
            print("No target boxes to display")

        plt.savefig(string+str(id_img[0])+".png")
        plt.close(fig)
    '''

    def forward(self, img, targets = None, img_id = 0):
        old_shape = img.shape[-2:]
        if self.training:
            img, targets['bboxes'] = self.normalise_resize_img_and_boxes(img, targets['bboxes'])
        else:
            img, _ = self.normalise_resize_img_and_boxes(img, None) 
        # img : (batch_size = 1, channel_number = 3, height = img_after_resize.height, width = img_after_resize.width)

        # call backbone
        feature_map = self.backbone(img) # (batch_size = 1, channel_number = 512, height = img_after_resize.height / 16, width = img_after_resize.width / 16)

        # call RPN and get proposals
        rpn_output = self.rpn(img, feature_map, targets)
        proposals = rpn_output['proposals']

        # call ROI head and convert proposals to boxes
        frcnn_output = self.roi_head(feature_map, proposals, img.shape[-2:], targets)

        # draw proposals on the img and save the img with proposals
        # self.draw_proposals_on_image(img, frcnn_output, targets['bboxes'], "frcnn_output_", img_id)

        if not self.training:
            # transform boxes back to original image size
            frcnn_output['bboxes'] = self.transform_boxes_to_original_image_size(frcnn_output['bboxes'], img.shape[-2:], old_shape)
            
        return rpn_output, frcnn_output




        
    



