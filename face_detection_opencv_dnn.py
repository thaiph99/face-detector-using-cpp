import argparse
import os
import time
import math

import cv2

# Colors  >>> BGR Format(BLUE, GREEN, RED)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 9, 2)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)


def detectFaceOpenCVDnn(net, frame, framework="caffe", conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    if framework == "caffe":
        blob = cv2.dnn.blobFromImage(
            frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False,
        )
    else:
        blob = cv2.dnn.blobFromImage(
            frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False,
        )

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    # for i in range(detections.shape[2]):
    i = 0
    confidence = detections[0, 0, i, 2]
    if confidence > conf_threshold:
        x1 = int(detections[0, 0, i, 3] * frameWidth)
        y1 = int(detections[0, 0, i, 4] * frameHeight)
        x2 = int(detections[0, 0, i, 5] * frameWidth)
        y2 = int(detections[0, 0, i, 6] * frameHeight)
        bboxes.append([x1, y1, x2, y2])
        cv2.rectangle(
            frameOpencvDnn,
            (x1, y1),
            (x2, y2),
            GOLDEN,
            int(round(frameHeight / 150)),
            2,
        )

        x3, y3 = (x1+x2)//2, (y1+y2)//2

        A = (frameWidth//2, frameHeight)
        B = (frameWidth//2, y3)
        C = (x3, y3)

        cv2.circle(frameOpencvDnn, C, 2, WHITE, 3, 2)
        cv2.circle(frameOpencvDnn, B, 2, WHITE, 3, 2)
        cv2.line(frameOpencvDnn, C, A, YELLOW, 1, 1)
        cv2.line(frameOpencvDnn, B, A, YELLOW, 1, 1)

        AB = math.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2)
        BC = math.sqrt((C[0]-B[0])**2+(C[1]-B[1])**2)
        AC = math.sqrt((A[0]-C[0])**2+(A[1]-C[1])**2)
        alpha1 = math.acos((AC**2+AB**2-BC**2)/(2*AB*AC))
        alpha1 = alpha1 * 180/math.pi
        alpha1 = round(alpha1, 2)
        # alpha = math.degrees(alpha)
        text = f"alpha : {alpha1} degrees"
        cv2.putText(frameOpencvDnn, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, BLUE, 2, cv2.LINE_AA)

    return frameOpencvDnn, bboxes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument("--video", type=str, default="",
                        help="Path to video file")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="caffe",
        choices=["caffe", "tf"],
        help="Type of network to run",
    )
    args = parser.parse_args()

    framework = args.framework
    source = args.video
    device = args.device

    # OpenCV DNN supports 2 networks.
    # 1. FP16 version of the original Caffe implementation ( 5.4 MB )
    # 2. 8 bit Quantized version using TensorFlow ( 2.7 MB )

    if framework == "caffe":
        modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "models/deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = "models/opencv_face_detector_uint8.pb"
        configFile = "models/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    if device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # outputFolder = "output-dnn-videos"
    # if not os.path.exists(outputFolder):
    #     os.makedirs(outputFolder)

    if source:
        cap = cv2.VideoCapture(source)
        outputFile = os.path.basename(source)[:-4] + ".avi"
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L)
        outputFile = "grabbed_from_camera.avi"

    # hasFrame, frame = cap.read()

    # vid_writer = cv2.VideoWriter(
    #     os.path.join(outputFolder, outputFile),
    #     cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    #     15,
    #     (frame.shape[1], frame.shape[0]),
    # )

    frame_count = 0
    tt_opencvDnn = 0

    while True:
        t = time.time()
        hasFrame, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if not hasFrame:
            break

        frame_count += 1

        outOpencvDnn, bboxes = detectFaceOpenCVDnn(net, frame)
        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn

        label = "OpenCV DNN {} FPS : {:.2f}".format(
            device.upper(), fpsOpencvDnn)
        cv2.putText(outOpencvDnn, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, BLUE, 2, cv2.LINE_AA)

        cv2.line(outOpencvDnn, (w//2, 0), (w//2, h), PINK, 2)
        cv2.imshow("Face Detection Comparison", outOpencvDnn)

        # vid_writer.write(outOpencvDnn)

        if frame_count == 1:
            tt_opencvDnn = 0

        k = cv2.waitKey(5)
        if k == 27:
            break

    cv2.destroyAllWindows()
    # vid_writer.release()
