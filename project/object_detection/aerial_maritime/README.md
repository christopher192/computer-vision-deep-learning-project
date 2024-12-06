# Object Detection (PyTorch)

## <ins>Introduction</ins>

## <ins>Evaluation Metric</ins>
### Intersection Over Union (IoU)
![](image/iou.png)
<br>Measurement based on Jaccard Index, is used to measure the overlap between two bounding boxes—typically a ground truth bounding box and a predicted bounding box. By calculating the IoU, we can determine whether a detection is valid (True Positive) or invalid (False Positive).

Basic Concept for Precison Recall Curve (Confusion Matrix)
- True Positive (TP): A correct detection. Detection with IoU ≥ threshold.
- False Positive (FP): A wrong detection. Detection with IoU < threshold.
- False Negative (FN): A ground truth not detected.
- True Negative (TN): Does not apply. It means possible bounding boxes that were corrrectly not detected or should not be detected within an image.

Threshold is usually set to 50%, 75% or 95%.

### Precision
- The accuracy of the detected objects, indicating how many detections were correct.
- Consider important if want to minimizing false detections.

Precision = TP/ (TP + FP) = TP/ All Detection

### Recall
- The ability of the model to identify all instances of objects in the images.
- Consider important if want to detect every instances of an object.

Recall = TP/ (TP + FN) = TP/ All Ground Truth

### Precision Recall Curve
- Good object detector: Maintain high precision as recall increasing.
- Poor object detector: Rapid drop in precision as recall increasing.

### Average Precision (AP)

### Mean Average Precision (mAP)

To be continued....

### Confusion Matrix
Before the IoU and class matching operations are performed, confidence score is used to filter out detections that are not considered confident enough.
1. True Positive (TP)
    - The detection is matched with a ground truth (IoU > threshold).
    - The detection class matches the ground truth class.
2. False Positive (FP)
    - A detection either IoU < threshold or class mismatch.
3. False Negative (FN)
    - Ground truth (IoU < threshold or class mismatch).

## <ins>Result</ins>
![](result/2024-04-18_10-54-14/confusion_matrix.png)

## <ins>Reference</ins>
1. https://github.com/rafaelpadilla/Object-Detection-Metrics?tab=readme-ov-file
2. https://docs.ultralytics.com/guides/yolo-performance-metrics/#interpretation-of-results