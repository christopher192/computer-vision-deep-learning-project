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

### TinyYOLOv2 ONNX Implementation in Python
- Can review how `TinyYOLOv2` works in this notebook `tinyyolov2-8-onnx.ipynb`.

### Implementation
The Unity implementation is primarily written in `TinyYoloDetector.cs` and consists of the following major components.

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

4. YOLOv2 Output Decoding<br>
Parses the model output tensor to extract bounding boxes and class scores.
```
List<(Rect, float, string)> DecodeYOLO(Tensor output)
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

### Limitation
- Only TinyYOLOv2 and TinyYOLOv3 are reliably supported by Unity Barracuda due to limited ONNX operator support.
- Retraining is required if you want to detect custom object classes or improve detection accuracy beyond the Pascal VOC dataset.