//参考《OpenCV3编程入门》第二章的代码
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void my_point_example()
{
	Mat img = imread("img_point.jpg");
	resize(img, img, Size(1000, 750));
	Mat img2 = img.clone();


	Mat img_gray;
	cvtColor(img, img_gray, COLOR_BGR2GRAY);
	imshow("img_gray", img_gray);

	vector < Point2f > corners;
	int max_corners = 200;
	double quality_level = 0.01;
	double min_distance = 3.0;
	int block_size = 3;
	bool use_harris = false;
	double k = 0.04;
	goodFeaturesToTrack(img_gray, // 输入图像（CV_8UC1 CV_32FC1） 
		corners, // 输出角点vector  
		max_corners, // 最大角点数目  
		quality_level,// 质量水平系数（小于1.0的正数，一般在0.01-0.1之间）  
		min_distance,// 最小距离，小于此距离的点忽略 
		Mat(),// mask=0的点忽略  
		block_size,// 使用的邻域数 
		use_harris,// false ='Shi Tomasi metric' true = 'harr' 
		k); // Harris角点检测时使用  

	for (int i = 0; i < corners.size(); i++)
	{
		circle(img, corners[i], 1, Scalar(0, 0, 255), 2);
	}

	TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 40, 0.01);
	cornerSubPix(img_gray, // 输入图像 
		corners, // 角点（既作为输入也作为输出）  
		Size(5, 5),  // 区域大小为 NXN; N=(winSize*2+1)  
		Size(-1, -1), // 类似于winSize，但是总具有较小的范围，Size(-1,-1)表示忽略  
		termcrit);// 停止优化的标准

	for (int i = 0; i < corners.size(); i++)
	{
		circle(img2, corners[i], 1, Scalar(0, 0, 255), 2);
	}

	imshow("img", img);
	imshow("亚像素级", img2);
}

Point2f point;
bool addRemovePt = false;
const int MAX_COUNT = 500;

//鼠标操作回调
static void onMouse(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		point = Point2f((float)x, (float)y);
		addRemovePt = true;
	}
}

int main(int argc, char** argv)
{
	cout << "当前使用的OpenCV版本为：" << CV_VERSION << endl;

	VideoCapture cap(0);

	//算法迭代终止条件
	TermCriteria termcrit(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);

	namedWindow("LK", 1);
	//鼠标操作
	setMouseCallback("LK", onMouse);

	Mat gray, prevGray, image;
	//point0为特征点的原来位置，point1为特征点的新位置
	vector<Point2f> points[2];
	while (1)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty())
			break;

		frame.copyTo(image);
		cvtColor(image, gray, COLOR_BGR2GRAY);

		if (!points[0].empty())
		{
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty())
				gray.copyTo(prevGray);
			calcOpticalFlowPyrLK(
				prevGray, //前一帧图像
				gray, //下一帧图像
				points[0], //输入坐标，单精度浮点性
				points[1], //输出坐标，单精度浮点性
				status, //跟踪特征的状态，特征的流发现为1，否则为0
				err, //输出误差矢量
				winSize,//每个金字塔层搜索窗大小
				3, //金字塔层的最大数目；如果置0，金字塔不使用(单层)；如果置1，金字塔2层，等等以此类推
				termcrit, //
				0, 0.001);

			//删除一些匹配不好的点
			int i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				//有特征的流发现 则添加该点
				if (status[i])
				{
					points[1][k++] = points[1][i];
					circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
				}
			}
			points[1].resize(k);
		}

		//鼠标添加特征点
		if (addRemovePt && points[1].size() < MAX_COUNT)
		{
			vector<Point2f> tmp;
			tmp.push_back(point);

			//获得亚像素点
			cornerSubPix(gray, tmp, winSize, Size(-1, -1), termcrit);
			points[1].push_back(tmp[0]);
			addRemovePt = false;
		}

		imshow("LK", image);

		char c = waitKey(10);
		switch (c)
		{
			case 'c':
				points[0].clear();
				points[1].clear();
				break;
			case 27: return 0; break;
		}

		std::swap(points[1], points[0]);
		cv::swap(prevGray, gray);
	}

	return 0;
}