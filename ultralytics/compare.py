from ultralytics import YOLO
import cv2
import numpy as np
import os
from torchvision.ops import box_iou
import torch

def xywh2xyxy(boxes, img_w, img_h):
    boxes = boxes.clone() if isinstance(boxes, torch.Tensor) else np.array(boxes)
    x_c, y_c, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    return torch.stack([x1, y1, x2, y2], dim=1) if isinstance(boxes, torch.Tensor) else np.stack([x1, y1, x2, y2], axis=1)

def draw_boxes(img, boxes, classes=None, scores=None, color=(0,255,0), label_prefix=""):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = ""
        if classes is not None:
            label += f"{label_prefix}{int(classes[i])}"
        if scores is not None:
            label += f" {scores[i]:.2f}"
        if label:
            cv2.putText(img, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    img_path = r"C:\Users\cly\Desktop\datasets\images\train\c000597.jpg"
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
    model_path = r"C:\Users\cly\Desktop\ultralytics\runs\train\b+c+r0.4\weights\best.pt"
    iou_threshold = 0.5
    save_dir = r"C:\Users\cly\Desktop\ultralytics\runs\detect"
    os.makedirs(save_dir, exist_ok=True)

    # 读图
    img_ori = cv2.imread(img_path)
    h, w = img_ori.shape[:2]

    # 读GT
    if not os.path.exists(label_path):
        print(f"找不到标签文件：{label_path}")
        return
    gt_boxes = []
    gt_classes = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            cls, x, y, bw, bh = map(float, line.strip().split())
            gt_boxes.append([x, y, bw, bh])
            gt_classes.append(int(cls))
    gt_boxes = torch.tensor(gt_boxes)
    gt_boxes_xyxy = xywh2xyxy(gt_boxes, w, h)

    # 加载模型推理
    model = YOLO(model_path)
    results = model(img_path, conf=0.25)[0]

    pred_boxes = results.boxes.xyxy.cpu()
    pred_scores = results.boxes.conf.cpu()
    pred_classes = results.boxes.cls.cpu().int()

    # 计算IoU矩阵
    if len(pred_boxes) == 0 or len(gt_boxes_xyxy) == 0:
        print("没有预测框或没有真实框")
        return
    ious = box_iou(pred_boxes, gt_boxes_xyxy)

    max_iou_vals, max_iou_idx = ious.max(dim=1)
    matched_pred_idx = []
    matched_gt_idx = []

    for pred_idx, (iou_val, gt_idx) in enumerate(zip(max_iou_vals, max_iou_idx)):
        if iou_val >= iou_threshold and gt_idx.item() not in matched_gt_idx:
            matched_pred_idx.append(pred_idx)
            matched_gt_idx.append(gt_idx.item())

    ### 1. 纯预测图（红色假阳+绿色真阳）
    img_pred = img_ori.copy()
    # 预测匹配上 (绿)
    draw_boxes(img_pred, pred_boxes[matched_pred_idx], pred_classes[matched_pred_idx].tolist(),
               pred_scores[matched_pred_idx].tolist(), color=(0,255,0), label_prefix="P")
    # 预测未匹配 (红)
    unmatched_pred_idx = [i for i in range(len(pred_boxes)) if i not in matched_pred_idx]
    draw_boxes(img_pred, pred_boxes[unmatched_pred_idx], pred_classes[unmatched_pred_idx].tolist(),
               pred_scores[unmatched_pred_idx].tolist(), color=(0,0,255), label_prefix="FP")
    cv2.imwrite(os.path.join(save_dir, "pred_only.jpg"), img_pred)

    ### 2. 纯GT图（蓝色）
    img_gt = img_ori.copy()
    draw_boxes(img_gt, gt_boxes_xyxy, gt_classes, color=(255,0,0), label_prefix="GT")
    cv2.imwrite(os.path.join(save_dir, "gt_only.jpg"), img_gt)

    ### 3. 对比图（红=假阳，绿=真阳，蓝=漏检）
    img_cmp = img_ori.copy()
    # 预测匹配上 (绿)
    draw_boxes(img_cmp, pred_boxes[matched_pred_idx], pred_classes[matched_pred_idx].tolist(),
               pred_scores[matched_pred_idx].tolist(), color=(0,255,0), label_prefix="P")
    # 预测未匹配 (红)
    draw_boxes(img_cmp, pred_boxes[unmatched_pred_idx], pred_classes[unmatched_pred_idx].tolist(),
               pred_scores[unmatched_pred_idx].tolist(), color=(0,0,255), label_prefix="FP")
    # 漏检真实框 (蓝)
    unmatched_gt_idx = [i for i in range(len(gt_boxes_xyxy)) if i not in matched_gt_idx]
    draw_boxes(img_cmp, gt_boxes_xyxy[unmatched_gt_idx], [gt_classes[i] for i in unmatched_gt_idx],
               color=(255,0,0), label_prefix="FN")
    cv2.imwrite(os.path.join(save_dir, "compare.jpg"), img_cmp)

    print(f"保存三张图到 {save_dir} ：pred_only.jpg, gt_only.jpg, compare.jpg")

if __name__ == "__main__":
    main()
