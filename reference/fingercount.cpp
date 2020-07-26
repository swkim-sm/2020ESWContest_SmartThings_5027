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
	VideoCapture cap(0); // ���� ĸ�� ��ü ����

	if (!cap.isOpened())  // ���������� open ���� ������ exit program
	{
		cout << "Cannot open the web cam!" << endl;
		return -1;
	}

	while (true)
	{
		Mat imgOriginal;

		bool bSuccess = cap.read(imgOriginal); // video�� �� frame�� ����

		if (!bSuccess) // ���������� read���� ���ϸ� break
		{
			cout << "Cannot read a frame from video stream!" << endl;
			break;
		}

		Mat imgYCrCb;
		cvtColor(imgOriginal, imgYCrCb, COLOR_BGR2YCrCb);

		Mat imgSharpened;
		sharpened(imgYCrCb, imgSharpened);

		Mat imgThresholded(imgOriginal.size(), CV_8UC1);
		skinColorDetection(imgSharpened, imgThresholded); // inRange, ����ȭ

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
		imshow("Result", imgHand); // ��� �̹����� ������
		imshow("Original", imgOriginal); // ���� �̹����� ������

		// 'Esc' Ű�� ������ break
		if (waitKey(30) == 27) {
			cout << "Esc key is pressed by user" << endl;
			break;
		}
	}

	return 0;
}

// ������ �Ǻθ� ������ �� �̹����� ����ȭ��Ű�� ���͸��ϴ� �Լ�
void skinColorDetection(Mat &imgOriginal, Mat &imgThresholded)
{
	Mat YCrCb[3];
	split(imgOriginal, YCrCb);

	int maxCr = 173; int minCr = 133;
	int maxCb = 127; int minCb = 77;

	// Cr Cb ����ȭ
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

	// MORPH_ELLIPSE�� Ÿ�����
	// �������� ����� closing ���� (removes small holes from the foreground ���� �޿��, ��â-ħ��)
	dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	
	medianBlur(imgThresholded, imgThresholded, 5);
}

// unsharp mask�� �����Ͽ� �̹����� �����ϰ� ����� �Լ�
void sharpened(Mat &imgOriginal, Mat &imgSharpened)
{
	Mat blurred; double sigma = 1, threshold = 5, amount = 1;
	GaussianBlur(imgOriginal, blurred, Size(), sigma, sigma);
	Mat lowContrastMask = abs(imgOriginal - blurred) < threshold;
	imgSharpened = imgOriginal * (1 + amount) + blurred * (-amount);
	imgOriginal.copyTo(imgSharpened, lowContrastMask);
}

// ū ������ �����ؼ� ���� ã�� �Լ�
void findHand(Mat &imgOriginal, Mat &imgThresholded, Mat &imgLargest, Mat&imgRect)
{
	vector<vector<Point>> contours; // �ܰ��� �迭

	// contours�� line �� ���� �ٱ��� line�� ã��, ��� contours point�� �����ϴ� ����� ���
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

// ���� �߽��� ã�� �Լ�
Point getHandCenter(const Mat& mask, double& radius) {

	Mat dst;   //�Ÿ� ��ȯ ����� ������ Mat ����

	// distanceType(�Ÿ� ���� ���)�� ��Ŭ����(DIST_L2), ����ũ�� ������� 5*5
	distanceTransform(mask, dst, DIST_L2, DIST_MASK_5);

	int maxIdx[2];   //��ǥ ���� ���� �迭(��, �� ������ �����)

	// ��� dst���� �ִ밪 radius�� ����ϰ�, �ִ� ��ġ�� maxIdx �迭�� ����
	minMaxIdx(dst, NULL, &radius, NULL, maxIdx, mask);

	return Point(maxIdx[1], maxIdx[0]);
}

// �հ����� ������ ���� �Լ�
int fingerCounting(const Mat& imgLargest, Point center, double radius, double scale) {

	// �հ��� ������ ���� ���� �� �׸���
	Mat imgCircle(imgLargest.size(), CV_8U, Scalar(0));
	circle(imgCircle, center, radius*scale, Scalar(255, 0, 0));

	// ���� �ܰ����� ������ ����
	vector<vector<Point>> contours;

	// contours�� line �� ���� �ٱ��� line�� ã��, line�� �׸� �� �ִ� point�� �����ϴ� ��� ���
	findContours(imgCircle, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// �ܰ����� ���� ���� �� ���� X
	if (contours.size() == 0)
		return -1;

	// �ܰ����� ���� ���� mask�� ���� 0���� 1�� �ٲ�� ���� Ȯ��
	int fingerCount = 0;
	for (int i = 1; i < contours[0].size(); i++) {
		Point p1 = contours[0][i - 1];
		Point p2 = contours[0][i];
		if (imgLargest.at<uchar>(p1.y, p1.x) == 0 && imgLargest.at<uchar>(p2.y, p2.x)>1)
			fingerCount++;
	}

	// �ո�� ������ ���� 1�� ����
	return (fingerCount - 1);
}

// ���콺�� ���۽�Ű�� �Լ�
void movingMousePointer(Mat & mask, int fingers)
{
	double radius;
	Point center = getHandCenter(mask, radius);

	// �հ����� 5�� �ν� �� �� ���콺 Ŀ�� ��ġ�� set���ش�.
	if (fingers == 5) 
	{
		SetCursorPos((center.x - 130) * 4, (center.y - 150) * 3);
	}

	// �հ����� 0�� �νĵ� �� ���콺 ���� ��ư�� Ŭ�����ش�.
	if (fingers == 0) 
	{
		mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0); // ���콺 ���� ��ư ������
		mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);  // ���콺 ���� ��ư ����
		Sleep(5);
		mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
		mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
	}
}