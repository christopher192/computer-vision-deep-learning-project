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

### 