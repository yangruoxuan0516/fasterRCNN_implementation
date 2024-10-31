import torch
import numpy as np
import cv2
import argparse
import random
import os
import yaml
from tqdm import tqdm
from model.fasterRCNN import FasterRCNN
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt
    
    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area'):
    # det_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2, score], ...],
    #       'car' : [[x1, y1, x2, y2, score], ...]
    #   }
    #   {det_boxes_img_2},
    #   ...
    #   {det_boxes_img_N},
    # ]
    #
    # gt_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2], ...],
    #       'car' : [[x1, y1, x2, y2], ...]
    #   },
    #   {gt_boxes_img_2},
    #   ...
    #   {gt_boxes_img_N},
    # ]
    
    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)
    all_aps = {}
    # average precisions for ALL classes
    aps = []
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]
        
        # cls_dets = [
        #   (0, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (0, [x1_M, y1_M, x2_M, y2_M, score_M]),
        #   (1, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (1, [x1_N, y1_N, x2_N, y2_N, score_N]),
        #   ...
        # ]
        
        # Sort them by confidence score
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])
        
        # For tracking which gt boxes of this class have already been matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Number of gt boxes for this class for recall calculation
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)
        
        # For each prediction
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            # Get gt boxes for this image and this label
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1
            
            # Get best matching gt box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            # TP only if iou >= threshold and this gt has not yet been matched
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                # If tp then we set this gt box as matched
                gt_matched[im_idx][max_iou_gt_idx] = True
        # Cumulative tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            
            # Replace precision values with recall r with maximum precision value
            # of any recall value >= r
            # This computes the precision envelope
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # For computing area, get points where recall changes value
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Add the rectangular areas to get ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Get precision values for recall values >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]
                
                # Get max of those precision values
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
    # compute mAP at provided iou threshold
    mean_ap = sum(aps) / len(aps)
    return mean_ap, all_aps

'''




# import numpy as np

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area + 1e-6)
    return iou

def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area'):
    all_aps = []

    # Extract unique class labels from ground truth
    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)

    # Compute AP for each class
    for label in gt_labels:
        cls_dets = [
            (im_idx, det) 
            for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for det in im_dets[label]
        ]
        
        # Sort detections by confidence score in descending order
        cls_dets = sorted(cls_dets, key=lambda x: -x[1][-1])

        # Ground truth and matching tracking
        gt_matched = [[False] * len(im_gt.get(label, [])) for im_gt in gt_boxes]
        num_gts = sum(len(im_gt.get(label, [])) for im_gt in gt_boxes)
        tp = np.zeros(len(cls_dets))
        fp = np.zeros(len(cls_dets))

        for i, (im_idx, det) in enumerate(cls_dets):
            max_iou = 0
            max_iou_idx = -1
            for j, gt_box in enumerate(gt_boxes[im_idx].get(label, [])):
                iou = calculate_iou(det[:4], gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = j
            
            if max_iou >= iou_threshold and not gt_matched[im_idx][max_iou_idx]:
                tp[i] = 1
                gt_matched[im_idx][max_iou_idx] = True
            else:
                fp[i] = 1
        
        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / num_gts if num_gts > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float32).eps)

        # Calculate AP based on method
        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            for j in range(len(precisions) - 1, 0, -1):
                precisions[j - 1] = np.maximum(precisions[j - 1], precisions[j])
            ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1.1, 0.1):
                prec_interp = precisions[recalls >= interp_pt]
                ap += prec_interp.max() if prec_interp.size > 0 else 0.0
            ap /= 11.0
        else:
            raise ValueError("Method can only be 'area' or 'interp'")
        
        all_aps.append(ap)

    # Calculate mean AP over all classes
    mean_ap = np.mean(all_aps) if all_aps else 0
    return mean_ap, {label: ap for label, ap in zip(gt_labels, all_aps)}









def load_model_and_dataset():
    # # Read the config file #
    # with open(args.config_path, 'r') as file:
    #     try:
    #         config = yaml.safe_load(file)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    # print(config)
    ########################
    
    
    seed = 1111
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    voc = VOCDataset(split='test')
    test_dataset = DataLoader(voc, batch_size=1, shuffle=False)
    
    faster_rcnn_model = FasterRCNN(device)
    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    # faster_rcnn_model.load_state_dict(torch.load('/home/infres/ryang-23/fasterRCNN_implementation/result/model.pth',
    faster_rcnn_model.load_state_dict(torch.load('result/model.pth',
                                                 map_location=device))
    return faster_rcnn_model, voc, test_dataset


def infer():
    if not os.path.exists('samples'):
        os.mkdir('samples')
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset()
    
    # Hard coding the low score threshold for inference on images for now
    # Should come from config
    faster_rcnn_model.roi_head.threshold = 0.7
    
    for sample_count in tqdm(range(30)):
        # random_idx = random.randint(0, len(voc))
        random_idx = sample_count
        img_id, im, target = voc[random_idx]
        # print(img_id)
        im = im.unsqueeze(0).float().to(device)

        fname = voc.imgpath % str(img_id)
        # print(fname)
        gt_im = cv2.imread(fname)
        gt_im_copy = gt_im.copy()


        # Saving images with ground truth boxes
        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            text = voc.idx2label[target['labels'][idx].detach().cpu().item()]
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(gt_im_copy , (x1, y1), (x1 + 10+text_w, y1 + 10+text_h), [255, 255, 255], -1)
            cv2.putText(gt_im, text=voc.idx2label[target['labels'][idx].detach().cpu().item()],
                        org=(x1+5, y1+15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(gt_im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        cv2.imwrite('samples/output_frcnn_gt_{}.png'.format(sample_count), gt_im)
        
        # Getting predictions from trained model
        rpn_output, frcnn_output = faster_rcnn_model(im, None)
        boxes = frcnn_output['bboxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        im = cv2.imread(fname)
        im_copy = im.copy()
        
        # Saving images with predicted boxes
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(im, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            text = '{} : {:.2f}'.format(voc.idx2label[labels[idx].detach().cpu().item()],
                                        scores[idx].detach().cpu().item())
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(im_copy , (x1, y1), (x1 + 10+text_w, y1 + 10+text_h), [255, 255, 255], -1)
            cv2.putText(im, text=text,
                        org=(x1+5, y1+15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(im_copy, 0.7, im, 0.3, 0, im)
        cv2.imwrite('samples/output_frcnn_{}.jpg'.format(sample_count), im)


def evaluate_map():
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset()

    faster_rcnn_model.roi_head.threshold = 0.7

    gts = []
    preds = []
    stop = 0
    for one_batch in tqdm(test_dataset):

        if stop == 1000:
            break
        stop += 1
        img_id, im, target = one_batch
        # print('img_id', img_id)
        # print('target', target)
        # im_name = fname
        im = im.float().to(device)
        target_boxes = target['bboxes'].float().to(device)[0]
        target_labels = target['labels'].long().to(device)[0]
        rpn_output, frcnn_output = faster_rcnn_model(im, None)

        boxes = frcnn_output['bboxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        
        # print('boxes', boxes)
        # print('labels', labels)
        # print('scores', scores)


        pred_boxes = {}
        gt_boxes = {}
        for label_name in voc.label2idx:
            pred_boxes[label_name] = []
            gt_boxes[label_name] = []
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = labels[idx].detach().cpu().item()
            score = scores[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            pred_boxes[label_name].append([x1, y1, x2, y2, score])
        for idx, box in enumerate(target_boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = target_labels[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            gt_boxes[label_name].append([x1, y1, x2, y2])
        
        gts.append(gt_boxes)
        preds.append(pred_boxes)

    print('preds', preds)
    print('gts', gts)
   
    mean_ap, all_aps = compute_map(preds, gts)#, method='interp')

    print('mean_ap', mean_ap)

    print('Class Wise Average Precisions')
    for idx in range(len(voc.idx2label)):
        print('AP for class {} = {:.10f}'.format(voc.idx2label[idx], all_aps[voc.idx2label[idx]]))
    print('Mean Average Precision : {:.10f}'.format(mean_ap))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Arguments for faster rcnn inference')
    # parser.add_argument('--config', dest='config_path',
    #                     default='config/voc.yaml', type=str)
    # parser.add_argument('--evaluate', dest='evaluate',
    #                     default=False, type=bool)
    # parser.add_argument('--infer_samples', dest='infer_samples',
    #                     default=True, type=bool)
    # args = parser.parse_args()
    # if args.infer_samples:
    #     infer(args)
    # else:
    #     print('Not Inferring for samples as `infer_samples` argument is False')
        
    # if args.evaluate:
    #     evaluate_map(args)
    # else:
    #     print('Not Evaluating as `evaluate` argument is False')
    evaluate_map()
    # infer()
