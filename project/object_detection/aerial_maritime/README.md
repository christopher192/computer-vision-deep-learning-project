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
The `ConfusionMatrix` class has been referenced from the Ultralytics repository, an open-source implementation available on their GitHub page under the `ultralytics/utils/metrics.py` file. As the repository is a well-known and reliable source in the deep learning community, the implementation is efficient and suitable for this project. 

Link to further understand the code: https://docs.ultralytics.com/reference/utils/metrics/ 

Before the IoU and class matching operations are performed, confidence score is used to filter out detections that are not considered confident enough.
1. True Positive (TP)
    - The detection is matched with a ground truth (IoU > threshold).
    - The detection class matches the ground truth class.
2. False Positive (FP)
    - A detection either IoU < threshold or class mismatch.
3. False Negative (FN)
    - Ground truth (IoU < threshold or class mismatch).

## <ins>Result</ins>
<p float="left">
    <img src="result/2024-04-18_10-54-14/confusion_matrix.png" width="45%" />
</p>
Explanation

1. True Positives (TP)
    - Values along the diagonal of the matrix (correct predictions).
    - Example: For `car`, TP = 79, as 79 instances of `car` were correctly predicted as `car`.
2. False Positives (FP)
    - Values in the same row as a given class but outside the diagonal (instances wrongly predicted as this class).
    - Example: For `car`, FP = 2 (`movable-objects`) + 7 (`jetski`) + 70 (`background`).
3. False Negatives (FN)
    - Values in the same column as a given class but outside the diagonal (instances of the class wrongly predicted as another).
    - Example: For `car`, FN = 42 (`background`) + 2 (`jetski`).

When the ground truth is `background` (there is no object present, it is a background region), but the model detects it as `jetski` (a non-background class), this would be classified as a False Positive (FP).

If there is a `class` in the ground truth, but the model predicts it as `background`, this is a False Negative (FN).

## <ins>Reference</ins>
1. https://github.com/rafaelpadilla/Object-Detection-Metrics?tab=readme-ov-file
2. https://docs.ultralytics.com/guides/yolo-performance-metrics/#interpretation-of-results
3. https://github.com/ultralytics/ultralytics