# Object Detection

## <ins>Introduction</ins>

## <ins>Evaluation Metric</ins>
### Intersection Over Union (IoU)
![](image/iou.png)
<br>Measurement based on Jaccard Index, is used to measure the overlap between two bounding boxes—typically a ground truth bounding box and a predicted bounding box. By calculating the IoU, we can determine whether a detection is valid (True Positive) or invalid (False Positive).

Basic Concept (Confusion Matrix)
- True Positive (TP): A correct detection. Detection with IoU ≥ threshold.
- False Positive (FP): A wrong detection. Detection with IoU < threshold.
- False Negative (FN): A ground truth not detected.
- True Negative (TN): Does not apply. It means possible bounding boxes that were corrrectly not detected or should not be detected within an image.

Threshold is usually set to 50%, 75% or 95%.

Precision, Recall, Average Precision (AP), and Mean Average Precision (mAP), IoU Threshold, Confidence Score, Precision-Recall Curve

## <ins>Reference</ins>
1. https://github.com/rafaelpadilla/Object-Detection-Metrics?tab=readme-ov-file