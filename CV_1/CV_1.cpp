//OpenCV task 1 Соломахин.С.А. 381808-3

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//The first and the second task
Mat DetAndDraw(Mat& img, CascadeClassifier& cascade, double scale) {
    vector<Rect> faces;
    Mat gray;
    Mat tmpImg;
    cvtColor(img, gray, COLOR_RGB2GRAY);
    cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    tmpImg = img(Rect(faces[0].x, faces[0].y, faces[0].width + 0.1 * faces[0].width, faces[0].height + 0.1 * faces[0].height));
    Mat croppedImg = tmpImg.clone();
    for (size_t i = 0; i < faces.size(); i++) {
        Rect r = faces[i];
        Scalar color = Scalar(255, 0, 0);
        rectangle(img, Point(cvRound(r.x * scale), cvRound(r.y * scale)), Point(cvRound((r.x +
            r.width - 1) * scale), cvRound((r.y + r.height - 1) * scale)), color, 3, 8, 0);
    }

    imshow("Face Detection", img);
    imshow("Cropped Image", croppedImg);
    return croppedImg;
}

//The third task
Mat ShowBinaryEdges(Mat& img) {
    Mat edgedImg;
    Canny(img, edgedImg, 100, 100);

    imshow("Binnary Edges Iamge", edgedImg);
    return edgedImg;
}

//The fifth task
Mat MorphOp(Mat& img) {
    Mat dilImg;
    dilate(img, dilImg, getStructuringElement(1, Size(5, 5)));
    imshow("Morph Op", dilImg);
    return dilImg;
}

//The sixth task
Mat GaussFilt(Mat& img, Mat& origImg) {
    Mat m;
    Mat tmpImg;
    GaussianBlur(img, tmpImg, Size(5, 5), 0, 0);
    imshow("Gaussian Filtered", tmpImg);
    normalize(origImg, m, 0, 1, NORM_MINMAX);
    return m;
}

//The seventh task
Mat BilFilt(Mat& img) {
    Mat f1;
    Mat tmpImg;
    bilateralFilter(img, tmpImg, 5, 100, 100);
    f1 = tmpImg.clone();
    imshow("Bulateral Filtered", f1);
    return f1;
}

//The eighth task
Mat ImpContarst(Mat& img, Mat& f1) {
    Mat f2;
    Mat tmpImg;
    addWeighted(img, 2.6, f1, -1.5, 0, tmpImg);
    f2 = tmpImg.clone();
    imshow("8", f2);
    return f2;
}

//The ninth task
Mat FinalFilter(Mat& img, Mat& f1, Mat& f2, Mat& m) {
    Mat f = img.clone();

    uint8_t* pixelF = (uint8_t*)f.data;
    Scalar_<uint8_t> bgrF;

    uint8_t* pixel = (uint8_t*)img.data;
    int cn = img.channels();
    Scalar_<uint8_t> bgrPixel;

    uint8_t* pixelM = (uint8_t*)m.data;
    Scalar_<uint8_t> bgrM;

    uint8_t* pixelF1 = (uint8_t*)f1.data;
    Scalar_<uint8_t> bgrF1;

    uint8_t* pixelF2 = (uint8_t*)f2.data;
    Scalar_<uint8_t> bgrF2;

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            for (int l = 0; l < 3; l++) {
                bgrF.val[l] = pixelF[i * f.cols * cn + j * cn + l];
                bgrPixel.val[l] = pixel[i * img.cols * cn + j * cn + l];
                bgrM.val[l] = pixelM[i * m.cols * cn + j * cn + l];
                bgrF1.val[l] = pixelF1[i * f1.cols * cn + j * cn + l];
                bgrF2.val[l] = pixelF2[i * f2.cols * cn + j * cn + l];
            }

            for (int k = 0; k < 3; k++)
                bgrF.val[k] = bgrM.val[k] * bgrF2.val[k] + (1 - bgrM.val[k] * bgrF1.val[k]);

            pixelF[i * f.cols * cn + j * cn + 0] = bgrF.val[0];
            pixelF[i * f.cols * cn + j * cn + 1] = bgrF.val[1];
            pixelF[i * f.cols * cn + j * cn + 2] = bgrF.val[2];
        }
    }

    imshow("Final Filtered", f);
    return f;
}

int main()
{
    CascadeClassifier faceCascade;
    faceCascade.load("D:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml");
    double scale = 1;
    String path = "D:\\opencv\\sources\\samples\\data\\lena.jpg";
    Mat frame = imread(path);
    Mat frame1 = DetAndDraw(frame, faceCascade, scale);
    Mat frame2 = ShowBinaryEdges(frame1);
    Mat frame3 = MorphOp(frame2);
    Mat M = GaussFilt(frame3, frame1);
    Mat F1 = BilFilt(frame1);
    Mat F2 = ImpContarst(frame1, F1);
    Mat fin = FinalFilter(frame1, F1, F2, M);

    waitKey(0);

    return 0;
}

