/*****************************************************************************
*                                                                            *
*  @file     SharedCodedMatchings.h                                          *
*  @brief    匹配两片点云之间的同名点，其中必须至少要有一个确定的同名点		 *
*  Details.  使用前需要配置好PCL和openCV库，该程序中还需要额外的标志点		 *
*			 识别程序，findCircularMarker15，不方便开源。此外程序中还涉		 *
*			 及到openCV中的solvePnp的内容，由于新旧内容不一样，特此将可		 *
&			 用的源代码单独拷贝出来使用										 *
*                                                                            *
*  @author   Zeping Wu														 *
*  @date     2019/12/18													     *
*                                                                            *
*****************************************************************************/
#pragma once

#include <iostream>
#include <map>
#include <pcl\point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2\opencv.hpp>
#include <detect_markers.h>

namespace xyNameSpace {
	enum cam_list {
		CAM_LEFT = 0,
		CAM_RIGHT = 1
	};
	struct SharedCodedMatching {
		using uint = unsigned int;
	public:
		SharedCodedMatching() = default;
		SharedCodedMatching(const cv::String &camPath_);
		~SharedCodedMatching() {}

		bool set_camParas(const cv::String &camPath_);
		void set_cloud_all(const pcl::PointCloud<pcl::PointXYZ> &cloud_all_);
		bool set_codedPointXYZ(const pcl::PointCloud<pcl::PointXYZ> &coded_, const std::vector<uint> &number_);

		void set_reproError(const double &reproError_) { reproError = reproError_; }
		void set_thresMatch(const double &thresMatch_) { thresMatch = thresMatch_; }
		void set_kdtreeNum(int kdNum_) { kd_num = kdNum_; }

		//用立体视觉创建点云来求位姿变换关系
		bool process(const cv::Mat &left_, const cv::Mat &right_, std::vector<cv::Point3f> &result_, Eigen::Matrix4f &RT_, bool showImg_ = false);

		//用单相机pnp来求位姿变换关系
		bool process_pnp(const cv::Mat &img_, Eigen::Matrix4f &RT_, int flag_cam_=CAM_LEFT, int flag_pnp_=cv::SOLVEPNP_EPNP);

		static bool readTxtCloudFile(const std::string filename, pcl::PointCloud<pcl::PointXYZ>::Ptr &pnts);
		static bool saveTxtCloudFile(const std::string filename, pcl::PointCloud<pcl::PointXYZ>::Ptr &pnts);

		const cv::Mat &getCamL() const { return camLeft; }
		const cv::Mat &getCamR() const { return camRight; }
		const cv::Mat &getdistorL() const { return distortionL; }
		const cv::Mat &getdistorR() const { return distortionR; }




	private:
		std::map<uint, pcl::PointXYZ> codedPointXYZ;
		pcl::PointCloud<pcl::PointXYZ> cloud_all;
		std::vector<cv::Point3f> cv_CloudAll;

		double reproError = 5.0;	//重投影误差
		double thresMatch = 0.7;	//匹配阈值
		int kd_num = 10;			//搜索临近的kd值

		cv::Mat camLeft, camRight;
		cv::Mat distortionL, distortionR;
		cv::Mat R_extrinsic, T_extrinsic;

		int matchPoints(const std::vector<cv::Point2f> &proPoints, const std::vector<cv::Point2f> &imgPoints, std::vector<std::pair<int, int>> & matches, const double &reproError_);

		//定位近中心的编码标志点
		void locatingNearCenter(const std::vector<std::pair<unsigned int, cv::Point2f>>& results_center_, const cv::Mat &K_, pcl::PointXYZ &point_);

		//有输入图片，检查是否匹配正确
		bool posit(const cv::Mat &img, const cv::Mat &K, const cv::Mat &distortion,
			const std::vector<std::pair<unsigned int, cv::Point2f>> &results_center,
			const std::vector<cv::Point2f> &circles_temp, std::vector<std::pair<int, int>> &matches);

		bool posit(const cv::Mat &K, const cv::Mat &distortion,
			const std::vector<std::pair<unsigned int, cv::Point2f>> &results_center,
			const std::vector<cv::Point2f> &circles_temp, std::vector<std::pair<int, int>> &matches);

		//匹配左右图片对应性
		bool matchCorrespondence(const std::vector<std::pair<int, int>> &matches_left_, const std::vector<std::pair<int, int>> &matches_right_,
			std::vector<std::pair<int, int>> &matches_);

		bool matchCorrespondence(const std::vector<std::pair<int, int>> &matches_left_, const std::vector<std::pair<int, int>> &matches_right_, std::map<int, std::pair<int, int>> &matches_);

		//计算标志点三维点,并输出变换矩阵
		bool stereoReconstruction(const std::vector<cv::Point2f> &circles_L, const std::vector<cv::Point2f> &circles_R, const std::map<int, std::pair<int, int>> &matches, std::vector<cv::Point3f> &result_, Eigen::Matrix4f &RT_);

		//利用pnp算位姿
		void solvePnP_rt(const std::vector<cv::Point2f> &circles_, const std::vector<std::pair<int, int>> &matches_, Eigen::Matrix4f &RT_, int flag_cam_ = CAM_LEFT, int flag_pnp_ = cv::SOLVEPNP_EPNP);
	};
}
