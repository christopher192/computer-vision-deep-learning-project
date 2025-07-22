## XR Real-Time Object Detection

### Introduction
This project demonstrates a real-time object detection system in an XR environment using Unity and Barracuda, Unity's neural network inference library. Applying TinyYOLOv2 ONNX model, the system performs on-device inference to detect objects from a live camera feed or image within a Unity scene.

### Technology Used
1. `Unity Barracuda` - Neural network inference library for Unity<br>
https://github.com/Unity-Technologies/barracuda-release.git
2. `ONNX TinyYolo` - Pre-trained object detection model (Pascal VOC)<br>
https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/tiny-yolov2/model
3. `Unity WebCamTexture` - Captures live video from the user's camera<br>
https://docs.unity3d.com/6000.1/Documentation/ScriptReference/WebCamTexture.html

### Key Feature
- Real-time inference with `TinyYOLOv2 ONNX` using `Unity Barracuda`.
- Live camera input via `WebCamTexture`.
- Object visualization with bounding boxes and labels.
- Non-Maximum Suppression (NMS) to filter overlapping detections.
- Detecting `person` only (can be extended to all classes).

### Pros of Using Barracuda
1. On-Device Inference
    - Run directly on the device (CPU or GPU)
    - No need to send data to a server, low latency, no internet dependency, better privacy
    - Real-time okay at ~15–30 FPS
2. Cross-Platform Support
    - Support Android, iOS, Windows, macOS, and WebGL
    - Easily deploy a single app to multiple platform
3. Integrated with Unity
    - Seemless integration with
        - Unity UI (Canvas, RawImage, RectTransform) for drawing bounding boxes
        - WebCamTexture or ARCameraManager for live video input
    - No need for Python wrappers, Flask servers, or OpenCV
4. ONNX Model Compatibility
    - Support object detection model such as TinyYOLOv2, TinyYOLOv3, TinyYOLOv4 (simplified versions)
    - Other models such as MobileNet, ResNet, UNet, etc
    - Allows training in PyTorch/TensorFlow → export → Unity inference
5. Maintainability
    - Open-source, regularly updated by Unity Technologies

### Cons of Using Barracuda
1. No GPU CUDA/TensorRT Acceleration
    - Does not use CUDA or TensorRT, fully support hardware like NVIDIA GPUs
    - Result in slower performance for large models like YOLOv5/YOLOv8
2. Limited Model Support
    - Only support ONNX format
    - Unable to support many modern ONNX models (YOLOv5, YOLOv8, EfficientDet, etc) yet
3. No Built-In Postprocessing
    - No built-in Non-Max Suppression (NMS), softmax, or argmax
    - Must manually implement decoding function, bounding box scaling, NMS in C#
4. Lower Performance on Large Input
    - Inference on large images will be much slower without CUDA framework support
5. No Quantization Support
    - No support for INT8/FP16 quantized models
    - FP32 inference, result in higher memory usage and slower on devices

### TinyYOLOv2 ONNX Implementation in Python
- Can review how `TinyYOLOv2` works in this notebook `tinyyolov2-8-onnx.ipynb`.

### Required UI Component in Unity Hub
| Component                | Description                                                                |
| ------------------------ | -------------------------------------------------------------------------- |
| **Canvas**               | Root UI container. Set **Render Mode** to `Screen Space - Overlay`.        |
| **RawImage**             | Displays the live webcam feed (`WebCamTexture`).                           |
| **Bounding Box Prefab**  | A `UI Panel` (RectTransform + Image) with a `TextMeshPro` child. Drag it from the Hierarchy into `Assets/Prefabs/` folder so at runtime, the script will clone (Instantiate) this prefab for each detected object. |
| **BoundingBoxContainer** | An empty `RectTransform` (UI container) for dynamically adding boxes.      |

### Implementation
The Unity implementation is primarily written in `TinyYoloDetector.cs` and consists of the following major components.

YOLO Manager – 4 Required Parameters
- Model Asset (NNModel)
    - The TinyYOLOv2 ONNX model used for inference.
- WebCam Display (RawImage)
    - Shows the live camera feed.
- Bounding Box Container (RectTransform)
    - UI container to hold all detected bounding boxes.
- Bounding Box Prefab (RectTransform)
    - UI box prefab (with label) used to draw each detection.

1. Webcam Input<br>
Captures live video stream from the camera using `WebCamTexture` and displays it on a `RawImage`.
```
webcam = new WebCamTexture(INPUT_SIZE, INPUT_SIZE);
webcamDisplay.texture = webcam;
webcam.Play();
```

2. Model Inference (Barracuda)<br>
Loads the ONNX model with `ModelLoader` and executes it on each frame.
```
var model = ModelLoader.Load(modelAsset);
worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);
worker.Execute(inputTensor);
Tensor output = worker.PeekOutput();
```

3. Image Preprocessing<br>
The captured image is resized and converted into a Barracuda `Tensor`.
```
Tensor TransformInput(Color32[] pixels)
```

Purpose: Preprocess the webcam image into a Barracuda Tensor.<br>
Converts a `Color32[]` image (from `WebCamTexture`) into a normalized 3D tensor in the format `CHW` (Channels-Height-Width), suitable for model input.

| Stage                        | Shape                      | Format                       |
| ---------------------------- | -------------------------- | ---------------------------- |
| Raw webcam pixels            | `[camWidth * camHeight]`   | 1D `Color32[]`               |
| After Texture2D creation     | `[camHeight, camWidth, 4]` | RGBA texture                 |
| After resize to 416×416      | `[416, 416, 3]`            | HWC (RGB)                    |
| After float array conversion | `[3 × 416 × 416]`          | CHW (1D)                     |
| Final Tensor                 | `[1, 416, 416, 3]`         | NHWC (Barracuda TensorShape) |

4. YOLOv2 Output Decoding<br>
Parses the model output tensor to extract bounding boxes and class scores.
```
List<(Rect, float, string)> DecodeYOLO(Tensor output)
```

Purpose: Converting the raw model output tensor (from TinyYOLOv2) into a list of meaningful object detections` properties such as, Bounding Box, Class Label, Confidence Score.

- Output shape: [1, 13, 13, 125]
- For each of the 13×13 grid cells, the model predicts:
    - 5 boxes, each with:
    - tx, ty, tw, th: box coordinates
    - tc: objectness score
    - 20 class scores
- Total: (5 × (5 + 20)) = 125 values per cell

Loop through grid and anchors
- Extract box data apply sigmoid to objectness
    ```
    tx, ty, tw, th = output[...]  
    tc = Sigmoid(output[...])
    ```
- Compute class confidence
    ```
    probs = Softmax(classScores)
    confidence = tc * probs[classIndex]
    ```
- Filter low-confidence
    ```
    if (confidence < threshold) continue;
    ```
- Decode box position and size
    ```
    cx = (Sigmoid(tx) + x) / GRID_SIZE  
    cy = (Sigmoid(ty) + y) / GRID_SIZE  
    bw = exp(tw) * anchor_w / GRID_SIZE  
    bh = exp(th) * anchor_h / GRID_SIZE
    ```
- Add final box
    ```
    boxes.Add((rect, confidence, label))
    ```

Sigmoid(x) – Activation function for box center and confidence<br>
Used to normalize box center coordinates and confidence score, ensures values are between 0 and 1
```
sigmoid(x) = 1 / (1 + exp(-x))
```

Softmax(x) – Activation for class prediction<br>
Used to convert class scores into probability distribution over classes
```
softmax(xi) = exp(xi - max(x)) / sum(exp(xj - max(x)))
```

5. Non-Maximum Suppression (NMS)<br>
Filters out overlapping boxes based on confidence and IoU.
```
List<(Rect, float, string)> ApplyNMS(List<(Rect, float, string)> detections)
```

6. Bounding Box and Label visualization<br>
Draws bounding boxes and labels using Unity UI (RectTransform + TextMeshPro).
```
void DrawBoxes(List<(Rect, float, string)> boxes)
```

7. Label Filtering
Currently filters to show only `person` class, but can be extended.
```
if (labels[classIndex] == "person")
```

### Result
![alt text](<Screenshot 2025-07-21 154527.png>)

### Limitation
- Only TinyYOLOv2 and TinyYOLOv3 are reliably supported by Unity Barracuda due to limited ONNX operator support
- Retraining is required if you want to detect custom object classes or improve detection accuracy beyond the Pascal VOC dataset

### Reference
- https://github.com/thtrieu/darkflow
- https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects
- https://www.youtube.com/watch?v=PyjBd7IDYZs