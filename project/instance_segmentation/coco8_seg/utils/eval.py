import io
import gc
import copy
import torch
import torchvision
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from contextlib import redirect_stdout
import pycocotools.mask as mask_util
from typing import Tuple, List, Dict
from collections import OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss, maskrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers

def get_iou_types(model):
    iou_types = ["bbox"]
    if isinstance(model, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def coco_evaluate(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim = 1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

def prepare_for_coco_segmentation(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        scores = prediction["scores"]
        labels = prediction["labels"]
        masks = prediction["masks"]

        masks = masks > 0.5

        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype = np.uint8, order = "F"))[0] for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results

def coco_merge(img_ids, eval_imgs):
    all_img_ids = [img_ids]
    all_eval_imgs = [eval_imgs]

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index = True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs

def evaluation_forward(model, images, targets):
    original_image_sizes: List[Tuple[int, int]] = []

    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                bb_idx = torch.where(degenerate_boxes.any(dim = 1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    features = model.backbone(images.tensors)

    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    model.rpn.training = True
    model.roi_heads.training = True

    # rpn
    features_rpn = list(features.values())

    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)

    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    proposal_losses = {}

    assert targets is not None

    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)

    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )

    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    # roi_heads
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}

    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
    num_images = len(boxes)

    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )

    if model.roi_heads.has_mask():
        mask_proposals = [p["boxes"] for p in result]

        if matched_idxs is None:
            raise ValueError("if in training, matched_idxs should not be None")

        num_images = len(proposals)
        mask_proposals = []
        pos_matched_idxs = []

        for img_id in range(num_images):
            pos = torch.where(labels[img_id] > 0)[0]
            mask_proposals.append(proposals[img_id][pos])
            pos_matched_idxs.append(matched_idxs[img_id][pos])

        if model.roi_heads.mask_roi_pool is not None:
            mask_features = model.roi_heads.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features = model.roi_heads.mask_head(mask_features)
            mask_logits = model.roi_heads.mask_predictor(mask_features)
        else:
            raise Exception("Expected mask_roi_pool to be not None")

        loss_mask = {}

        if targets is None or pos_matched_idxs is None or mask_logits is None:
            raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

        gt_masks = [t["masks"] for t in targets]
        gt_labels = [t["labels"] for t in targets]
        rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
        loss_mask = {"loss_mask": rcnn_loss_mask}

        detector_losses.update(loss_mask)

    detections = result
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)

    model.rpn.training = False
    model.roi_heads.training = False

    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)

    return losses, detections

def evaluation_loss_map(model, validation_dataset, validation_dataloader,
    selected_keys, device, iou_types, cpu_device):
    model.eval()

    val_loss = 0.0
    val_loss_classifier = 0.0
    val_loss_box_reg = 0.0
    val_loss_mask = 0.0
    val_loss_objectness = 0.0
    val_loss_rpn_box_reg = 0.0

    coco = validation_dataloader.dataset.coco

    with torch.inference_mode():
        coco_gt = copy.deepcopy(coco)
        coco_eval = {}
        img_ids_list = []
        eval_imgs_list = {k: [] for k in iou_types}

        for iou_type in iou_types:
            coco_eval[iou_type] = COCOeval(coco_gt, iouType = iou_type)

        for batch_idx, (images, targets) in enumerate(validation_dataloader):
            images = list(image.to(device) for image in images)
            eval_targets = [{k: v.to(device) for k, v in t.items() if k in selected_keys} for t in targets]

            losses, detections = evaluation_forward(model, images, eval_targets)
            loss = sum(loss for loss in losses.values())

            val_loss += loss.item()
            val_loss_classifier += losses["loss_classifier"].item()
            val_loss_box_reg += losses["loss_box_reg"].item()
            val_loss_mask += losses["loss_mask"].item()
            val_loss_objectness += losses["loss_objectness"].item()
            val_loss_rpn_box_reg += losses["loss_rpn_box_reg"].item()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            outputs = model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            # coco evaluation
            res = {target["image_id"]: output for target, output in zip(targets, outputs)}

            unique_img_ids = list(np.unique(list(res.keys())))
            img_ids_list.extend(unique_img_ids)

            for iou_type in iou_types:
                if iou_type == "bbox":
                    results = prepare_for_coco_detection(res)
                if iou_type == 'segm':
                    results = prepare_for_coco_segmentation(res)
                with redirect_stdout(io.StringIO()):
                    coco_dt = COCO.loadRes(coco_gt, results) if results else COCO()

                eval = coco_eval[iou_type]

                eval.cocoDt = coco_dt
                eval.params.imgIds = list(unique_img_ids)

                img_ids, eval_imgs = coco_evaluate(eval)

                eval_imgs_list[iou_type].append(eval_imgs)

            # delete tensor and clear cache to free up memory
            del images, eval_targets, targets, losses, detections
            # clear system memory
            gc.collect()

        # clear cache after each epoch
        torch.cuda.empty_cache()
        # clear system memory
        gc.collect()

        # validation loss data
        losses_dict = {
            'total_val_loss': val_loss / len(validation_dataloader),
            'total_val_loss_classifier': val_loss_classifier / len(validation_dataloader),
            'total_val_loss_box_reg': val_loss_box_reg / len(validation_dataloader),
            'total_val_loss_mask': val_loss_mask / len(validation_dataloader),
            'total_val_loss_objectness': val_loss_objectness / len(validation_dataloader),
            'total_val_loss_rpn_box_reg': val_loss_rpn_box_reg / len(validation_dataloader)
        }

        # synchronize between processes
        for iou_type in iou_types:
            eval_imgs_list[iou_type] = np.concatenate(eval_imgs_list[iou_type], 2)

            array_img_ids = np.array(img_ids_list)
            merged_img_ids, idx = np.unique(array_img_ids, return_index = True)
            merged_eval_imgs = eval_imgs_list[iou_type][..., idx]

            img_ids = list(merged_img_ids)
            eval_imgs = list(merged_eval_imgs.flatten())

            coco_eval[iou_type].evalImgs = eval_imgs
            coco_eval[iou_type].params.imgIds = img_ids
            coco_eval[iou_type]._paramsEval = copy.deepcopy(coco_eval[iou_type].params)

        # coco accumulate
        for eval in coco_eval.values():
            eval.accumulate()

        # coco summary
        for iou_type, eval in coco_eval.items():
            print(f"IoU metric: {iou_type}")
            eval.summarize()

    return losses_dict