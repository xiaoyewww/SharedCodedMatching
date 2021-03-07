#include <iostream>
#include <pcl\io\pcd_io.h>
#include <opencv2\opencv.hpp>
#include <detect_markers.h>
#include "SharedCodedMatching.h"
#include <pcl/common/transforms.h>
//#include "SharedCodedMatching\include\SharedCodedMatching.h"
using namespace std;

int main(int argv, char** argc)
{

	/*测试代码*/
	string path = "./DataFile";

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_all(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr codedCloud(new pcl::PointCloud<pcl::PointXYZ>);

	string path_cloudAll = path + "/uncodeMarks.txt";

	if (!xyNameSpace::SharedCodedMatching::readTxtCloudFile(path_cloudAll, cloud_all))
	{
		PCL_ERROR("This dir doesnot exit  pcd file.\n");
		return(-1);
	}
	string path_coded = path + "/codeMarks.txt";
	if (!xyNameSpace::SharedCodedMatching::readTxtCloudFile(path_coded, codedCloud))
	{
		PCL_ERROR("This dir doesnot exit pcd file.\n");
		return(-1);
	}

	//读取编码点编号信息
	string path_codednumber = path + "/coded.txt";
	vector<unsigned int> coded_number;

	ifstream in(path_codednumber);
	if (!in.is_open())
		return -1;

	string s;
	while (getline(in, s))
	{
		coded_number.emplace_back(atoi(s.c_str()));
		cout << s << endl;
	}

	//xyNameSpace::SharedCodedMatching myMatching("./DataFile/cameraParas/StereoPara_lv700.xml");
	xyNameSpace::SharedCodedMatching myMatching("./DataFile/StereoPara.xml");

	myMatching.set_kdtreeNum(15);
	myMatching.set_cloud_all(*cloud_all);
	myMatching.set_codedPointXYZ(*codedCloud, coded_number);


	//cv::Mat imgL = cv::imread("./DataFile/lv700/imageL1.bmp", 0);
	//cv::Mat imgR = cv::imread("./DataFile/lv700/imageR1.bmp", 0);
	cv::Mat imgL = cv::imread("./DataFile/L/l01_0202.bmp", 0);
	cv::Mat imgR = cv::imread("./DataFile/R/r01_0202.bmp", 0);

	std::vector<cv::Point3f> result_stereo;
	Eigen::Matrix4f RT_, RT_2;
	myMatching.set_reproError(5);

	cv::Mat img = imgL.clone();
	//if (!myMatching.process_pnp(img, RT_, xyNameSpace::cam_list::CAM_LEFT, cv::SOLVEPNP_ITERATIVE))
	//{
	//	return -1;
	//}

	//用立体视觉创建点云计算RT_
	if (!myMatching.process(imgL, imgR, result_stereo, RT_2, true))
	{
		return -1;
	}
	cout << "RT:" << endl << RT_2 << endl;

	cv::Mat imgL2 = cv::imread("./DataFile/yyl/L01.bmp", 0);
	cv::Mat imgR2 = cv::imread("./DataFile/yyl/R01.bmp", 0);

	result_stereo.clear();
	if (!myMatching.process(imgL2, imgR2, result_stereo, RT_2, false))
	{
		return -1;
	}
	cout << "RT:" << endl << RT_2 << endl;

	cv::Mat imgL3 = cv::imread("./DataFile/yyl/Position_1_imageL_3.bmp", 0);
	cv::Mat imgR3 = cv::imread("./DataFile/yyl/Position_1_imageR_3.bmp", 0);

	result_stereo.clear();
	if (!myMatching.process(imgL3, imgR3, result_stereo, RT_2, true))
	{
		return -1;
	}
	cout << "RT:" << endl << RT_2 << endl;

	cv::Mat imgL4 = cv::imread("./DataFile/yyl/Position_1_imageL_4.bmp", 0);
	cv::Mat imgR4 = cv::imread("./DataFile/yyl/Position_1_imageR_4.bmp", 0);

	result_stereo.clear();
	if (!myMatching.process(imgL4, imgR4, result_stereo, RT_2, true))
	{
		return -1;
	}
	cout << "RT:" << endl << RT_2 << endl;

	Eigen::Matrix4d a2;
	a2 << RT_2(0, 0), RT_2(0, 1), RT_2(0, 2), RT_2(0, 3),
		RT_2(1, 0), RT_2(1, 1), RT_2(1, 2), RT_2(1, 3),
		RT_2(2, 0), RT_2(2, 1), RT_2(2, 2), RT_2(2, 3),
		RT_2(3, 0), RT_2(3, 1), RT_2(3, 2), RT_2(3, 3);

	return 0;
}