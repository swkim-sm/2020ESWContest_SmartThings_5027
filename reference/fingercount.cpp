#include <iostream>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <windows.h>

using namespace cv;
using namespace std;

void skinColorDetection(Mat &imgOriginal, Mat &imgThresholded);
void sharpened(Mat &imgOriginal, Mat &imgSharpened);
void findHand(Mat &imgOriginal, Mat&imgThresholded, Mat &imgLargest, Mat&imgRect);
Point getHandCenter(const Mat& mask, double& radius);
int fingerCounting(const Mat& imgThresholded, Point center, double radius, double scale);
void movingMousePointer(Mat & mask, int fingers);

int main(int argc, char** argv)
{
	VideoCapture cap(0); // 비디오 캡쳐 객체 생성

	if (!cap.isOpened())  // 정상적으로 open 되지 않으면 exit program
	{
		cout << "Cannot open the web cam!" << endl;
		return -1;
	}

	while (true)
	{
		Mat imgOriginal;

		bool bSuccess = cap.read(imgOriginal); // video를 한 frame씩 읽음

		if (!bSuccess) // 정상적으로 read하지 못하면 break
		{
			cout << "Cannot read a frame from video stream!" << endl;
			break;
		}

		Mat imgYCrCb;
		cvtColor(imgOriginal, imgYCrCb, COLOR_BGR2YCrCb);

		Mat imgSharpened;
		sharpened(imgYCrCb, imgSharpened);

		Mat imgThresholded(imgOriginal.size(), CV_8UC1);
		skinColorDetection(imgSharpened, imgThresholded); // inRange, 이진화

		Mat imgLargest(imgOriginal.size(), CV_8UC1, Scalar(0));
		Mat imgHand;

		findHand(imgOriginal, imgThresholded, imgLargest, imgHand);

		double radius;
		double scale = 2.0;

		Point center = getHandCenter(imgLargest, radius);

		int fingers;
		fingers = fingerCounting(imgLargest, center, radius, scale);
		circle(imgHand, center, 2, Scalar(0, 0, 255));
		circle(imgHand, center, radius*scale, Scalar(255, 0, 0));

		int font = FONT_HERSHEY_SCRIPT_SIMPLEX;

		if (fingers == -1 || fingers > 5) putText(imgHand, "Can Not Found", Point(20, 50), font, 2, Scalar(0, 255, 255), 3);
		else putText(imgHand, to_string(fingers), Point(20, 50), font, 2, Scalar(0, 255, 255), 3);

		movingMousePointer(imgLargest, fingers);

		namedWindow("Result");
		namedWindow("Original");
		imshow("Result", imgHand); // 결과 이미지를 보여줌
		imshow("Original", imgOriginal); // 원본 이미지를 보여줌

		// 'Esc' 키가 눌리면 break
		if (waitKey(30) == 27) {
			cout << "Esc key is pressed by user" << endl;
			break;
		}
	}

	return 0;
}

// 색으로 피부를 감지한 후 이미지를 이진화시키고 필터링하는 함수
void skinColorDetection(Mat &imgOriginal, Mat &imgThresholded)
{
	Mat YCrCb[3];
	split(imgOriginal, YCrCb);

	int maxCr = 173; int minCr = 133;
	int maxCb = 127; int minCb = 77;

	// Cr Cb 이진화
	for (int i = 0; i < imgOriginal.rows; i++)
	{
		for (int j = 0; j < imgOriginal.cols; j++)
		{
			if (YCrCb[1].at<uchar>(i, j) >= minCr && YCrCb[1].at<uchar>(i, j) <= maxCr
				&& YCrCb[2].at<uchar>(i, j) >= minCb && YCrCb[2].at<uchar>(i, j) <= maxCb)
			{
				imgThresholded.at<uchar>(i, j) = 255;
			}
			else {
				imgThresholded.at<uchar>(i, j) = 0;
			}
		}
	}

	// MORPH_ELLIPSE는 타원모양
	// 모폴로지 기법의 closing 연산 (removes small holes from the foreground 구멍 메우기, 팽창-침식)
	dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	
	medianBlur(imgThresholded, imgThresholded, 5);
}

// unsharp mask를 적용하여 이미지를 선명하게 만드는 함수
void sharpened(Mat &imgOriginal, Mat &imgSharpened)
{
	Mat blurred; double sigma = 1, threshold = 5, amount = 1;
	GaussianBlur(imgOriginal, blurred, Size(), sigma, sigma);
	Mat lowContrastMask = abs(imgOriginal - blurred) < threshold;
	imgSharpened = imgOriginal * (1 + amount) + blurred * (-amount);
	imgOriginal.copyTo(imgSharpened, lowContrastMask);
}

// 큰 영역을 인지해서 손을 찾는 함수
void findHand(Mat &imgOriginal, Mat &imgThresholded, Mat &imgLargest, Mat&imgRect)
{
	vector<vector<Point>> contours; // 외곽선 배열

	// contours는 line 중 가장 바깥쪽 line만 찾고, 모든 contours point를 저장하는 방법을 사용
	findContours(imgThresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	int largestContour = 0;

	for (int i = 0; i < contours.size(); i++) {
		if (contourArea(contours[i]) > contourArea(contours[largestContour])) {
			largestContour = i;
		}
	}
	drawContours(imgOriginal, contours, largestContour, Scalar(0, 255, 0), 2);
	drawContours(imgLargest, contours, largestContour, Scalar(255), FILLED);

	cvtColor(imgLargest, imgRect, COLOR_GRAY2BGR);
}

// 손의 중심을 찾는 함수
Point getHandCenter(const Mat& mask, double& radius) {

	Mat dst;   //거리 변환 행렬을 저장할 Mat 변수

	// distanceType(거리 측정 방식)은 유클리드(DIST_L2), 마스크의 사이즈는 5*5
	distanceTransform(mask, dst, DIST_L2, DIST_MASK_5);

	int maxIdx[2];   //좌표 값을 얻어올 배열(행, 열 순으로 저장됨)

	// 행렬 dst에서 최대값 radius를 계산하고, 최대 위치를 maxIdx 배열에 저장
	minMaxIdx(dst, NULL, &radius, NULL, maxIdx, mask);

	return Point(maxIdx[1], maxIdx[0]);
}

// 손가락의 개수를 세는 함수
int fingerCounting(const Mat& imgLargest, Point center, double radius, double scale) {

	// 손가락 개수를 세기 위한 원 그리기
	Mat imgCircle(imgLargest.size(), CV_8U, Scalar(0));
	circle(imgCircle, center, radius*scale, Scalar(255, 0, 0));

	// 원의 외곽선을 저장할 벡터
	vector<vector<Point>> contours;

	// contours는 line 중 가장 바깥쪽 line만 찾고, line을 그릴 수 있는 point만 저장하는 방식 사용
	findContours(imgCircle, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// 외곽선이 없을 때는 손 검출 X
	if (contours.size() == 0)
		return -1;

	// 외곽선을 따라 돌며 mask의 값이 0에서 1로 바뀌는 지점 확인
	int fingerCount = 0;
	for (int i = 1; i < contours[0].size(); i++) {
		Point p1 = contours[0][i - 1];
		Point p2 = contours[0][i];
		if (imgLargest.at<uchar>(p1.y, p1.x) == 0 && imgLargest.at<uchar>(p2.y, p2.x)>1)
			fingerCount++;
	}

	// 손목과 만나는 개수 1개 제외
	return (fingerCount - 1);
}

// 마우스를 동작시키는 함수
void movingMousePointer(Mat & mask, int fingers)
{
	double radius;
	Point center = getHandCenter(mask, radius);

	// 손가락이 5개 인식 될 때 마우스 커서 위치를 set해준다.
	if (fingers == 5) 
	{
		SetCursorPos((center.x - 130) * 4, (center.y - 150) * 3);
	}

	// 손가락이 0개 인식될 때 마우스 왼쪽 버튼을 클릭해준다.
	if (fingers == 0) 
	{
		mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0); // 마우스 왼쪽 버튼 누르기
		mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);  // 마우스 왼쪽 버튼 떼기
		Sleep(5);
		mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
		mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
	}
}