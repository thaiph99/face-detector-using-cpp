#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <boost/algorithm/string.hpp>
#include <math.h>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);

const std::string caffeConfigFile = "models/deploy.prototxt";
const std::string caffeWeightFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

const std::string tensorflowConfigFile = "models/opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "models/opencv_face_detector_uint8.pb";

cv::Scalar GREEN = (0, 255, 0),
           RED = (0, 0, 255),
           BLACK = (0, 0, 0),
           YELLOW = (0, 255, 255),
           WHITE = (255, 255, 255),
           CYAN = (255, 255, 0),
           MAGENTA = (255, 0, 242),
           GOLDEN = (32, 218, 165),
           LIGHT_BLUE = (255, 9, 2),
           PURPLE = (128, 0, 128),
           CHOCOLATE = (30, 105, 210),
           PINK = (147, 20, 255),
           ORANGE = (0, 69, 255);

void detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN, string framework)
{
    cout << YELLOW << endl;
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;

    Mat inputBlob;
    if (framework == "caffe")
        inputBlob = blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
    else
        inputBlob = blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);

    net.setInput(inputBlob, "data");
    Mat detection = net.forward("detection_out");
    int midFrame = int(frameWidth / 2);
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    int x1, x2, y1, y2;
    string s = "";
    float confidence;
    for (int i = 0; i < detectionMat.rows; i++)
    {
        confidence = detectionMat.at<float>(i, 2);
        if (confidence > confidenceThreshold)
        {
            x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
            rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(0, 225, 255), 2, 4);

            if (x1 <= midFrame && midFrame <= x2)
                continue;
            else if (midFrame < x1)
                s = format("right : %d fixels", abs(x1 - midFrame));
            else if (midFrame > x2)
                s = format("left : %d fixels", abs(midFrame - x2));

            putText(frameOpenCVDNN, s, Point(int(frameWidth / 2) + 5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
        }
    }
}

int main(int argc, const char **argv)
{
    string videoFileName;
    string device;
    string framework;

    // Take arguments from command line
    if (argc == 4)
    {
        videoFileName = argv[1];
        device = argv[2];
        framework = argv[3];
    }
    else if (argc == 3)
    {
        videoFileName = argv[1];
        device = argv[2];
        framework = "caffe";
    }
    else if (argc == 2)
    {
        videoFileName = argv[1];
        device = "cpu";
        framework = "caffe";
    }
    else
    {
        videoFileName = "";
        device = "cpu";
        framework = "caffe";
    }

    boost::to_upper(device);
    cout << "Configuration:" << endl;
    cout << "Device - " << device << endl;
    if (framework == "caffe")
        cout << "Network type - Caffe" << endl;
    else
        cout << "Network type - TensorFlow" << endl;
    if (videoFileName == "")
        cout << "No video found, using camera stream" << endl;
    else
        cout << "Video file - " << videoFileName << endl;

    Net net;

    if (framework == "caffe")
        net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
    else
        net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);

#if (CV_MAJOR_VERSION >= 4)
    if (device == "CPU")
    {
        net.setPreferableBackend(DNN_TARGET_CPU);
    }
    else
    {
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
    }
#else
    // force CPU backend for OpenCV 3.x as CUDA backend is not supported there
    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    device = "cpu";
#endif

    cv::VideoCapture source;
    if (videoFileName != "")
        source.open(videoFileName);
    else
        source.open(0, CAP_V4L);

    Mat frame_t;

    double tt_opencvDNN = 0;
    double fpsOpencvDNN = 0;
    double totalFps = 0;
    double totalFrames = 0;
    while (true)
    {
        source >> frame_t;
        Mat frame;
        flip(frame_t, frame, 1);
        if (frame.empty())
            break;

        double t = cv::getTickCount();
        detectFaceOpenCVDNN(net, frame, framework);
        tt_opencvDNN = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        fpsOpencvDNN = 1 / tt_opencvDNN;
        totalFps += fpsOpencvDNN;
        totalFrames++;

        int frameHeight = frame.rows;
        int frameWidth = frame.cols;

        Point p1(int(frameWidth / 2), 0);
        Point p2(int(frameWidth / 2), frameHeight);
        line(frame, p1, p2, Scalar(0, 0, 255), 4);

        putText(frame, format("OpenCV DNN %s FPS = %.2f", device.c_str(), fpsOpencvDNN), Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);

        imshow("OpenCV - DNN Face Detection", frame);
        int k = waitKey(5);
        if (k == 27)
        {
            destroyAllWindows();
            break;
        }
    }
    printf("Average Fps : %.2f\n", totalFps / totalFrames);
}