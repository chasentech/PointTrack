//�ο���OpenCV3������š��ڶ��µĴ���
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
	goodFeaturesToTrack(img_gray, // ����ͼ��CV_8UC1 CV_32FC1�� 
		corners, // ����ǵ�vector  
		max_corners, // ���ǵ���Ŀ  
		quality_level,// ����ˮƽϵ����С��1.0��������һ����0.01-0.1֮�䣩  
		min_distance,// ��С���룬С�ڴ˾���ĵ���� 
		Mat(),// mask=0�ĵ����  
		block_size,// ʹ�õ������� 
		use_harris,// false ='Shi Tomasi metric' true = 'harr' 
		k); // Harris�ǵ���ʱʹ��  

	for (int i = 0; i < corners.size(); i++)
	{
		circle(img, corners[i], 1, Scalar(0, 0, 255), 2);
	}

	TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 40, 0.01);
	cornerSubPix(img_gray, // ����ͼ�� 
		corners, // �ǵ㣨����Ϊ����Ҳ��Ϊ�����  
		Size(5, 5),  // �����СΪ NXN; N=(winSize*2+1)  
		Size(-1, -1), // ������winSize�������ܾ��н�С�ķ�Χ��Size(-1,-1)��ʾ����  
		termcrit);// ֹͣ�Ż��ı�׼

	for (int i = 0; i < corners.size(); i++)
	{
		circle(img2, corners[i], 1, Scalar(0, 0, 255), 2);
	}

	imshow("img", img);
	imshow("�����ؼ�", img2);
}

Point2f point;
bool addRemovePt = false;
const int MAX_COUNT = 500;

//�������ص�
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
	cout << "��ǰʹ�õ�OpenCV�汾Ϊ��" << CV_VERSION << endl;

	VideoCapture cap(0);

	//�㷨������ֹ����
	TermCriteria termcrit(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);

	namedWindow("LK", 1);
	//������
	setMouseCallback("LK", onMouse);

	Mat gray, prevGray, image;
	//point0Ϊ�������ԭ��λ�ã�point1Ϊ���������λ��
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
				prevGray, //ǰһ֡ͼ��
				gray, //��һ֡ͼ��
				points[0], //�������꣬�����ȸ�����
				points[1], //������꣬�����ȸ�����
				status, //����������״̬��������������Ϊ1������Ϊ0
				err, //������ʸ��
				winSize,//ÿ������������������С
				3, //��������������Ŀ�������0����������ʹ��(����)�������1��������2�㣬�ȵ��Դ�����
				termcrit, //
				0, 0.001);

			//ɾ��һЩƥ�䲻�õĵ�
			int i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				//�������������� ����Ӹõ�
				if (status[i])
				{
					points[1][k++] = points[1][i];
					circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
				}
			}
			points[1].resize(k);
		}

		//������������
		if (addRemovePt && points[1].size() < MAX_COUNT)
		{
			vector<Point2f> tmp;
			tmp.push_back(point);

			//��������ص�
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