/*****************************************************************************
*                                                                            *
*  @file     SharedCodedMatchings.h                                          *
*  @brief    ƥ����Ƭ����֮���ͬ���㣬���б�������Ҫ��һ��ȷ����ͬ����		 *
*  Details.  ʹ��ǰ��Ҫ���ú�PCL��openCV�⣬�ó����л���Ҫ����ı�־��		 *
*			 ʶ�����findCircularMarker15�������㿪Դ����������л���		 *
*			 ����openCV�е�solvePnp�����ݣ������¾����ݲ�һ�����ش˽���		 *
&			 �õ�Դ���뵥����������ʹ��										 *
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

		//�������Ӿ�������������λ�˱任��ϵ
		bool process(const cv::Mat &left_, const cv::Mat &right_, std::vector<cv::Point3f> &result_, Eigen::Matrix4f &RT_, bool showImg_ = false);

		//�õ����pnp����λ�˱任��ϵ
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

		double reproError = 5.0;	//��ͶӰ���
		double thresMatch = 0.7;	//ƥ����ֵ
		int kd_num = 10;			//�����ٽ���kdֵ

		cv::Mat camLeft, camRight;
		cv::Mat distortionL, distortionR;
		cv::Mat R_extrinsic, T_extrinsic;

		int matchPoints(const std::vector<cv::Point2f> &proPoints, const std::vector<cv::Point2f> &imgPoints, std::vector<std::pair<int, int>> & matches, const double &reproError_);

		//��λ�����ĵı����־��
		void locatingNearCenter(const std::vector<std::pair<unsigned int, cv::Point2f>>& results_center_, const cv::Mat &K_, pcl::PointXYZ &point_);

		//������ͼƬ������Ƿ�ƥ����ȷ
		bool posit(const cv::Mat &img, const cv::Mat &K, const cv::Mat &distortion,
			const std::vector<std::pair<unsigned int, cv::Point2f>> &results_center,
			const std::vector<cv::Point2f> &circles_temp, std::vector<std::pair<int, int>> &matches);

		bool posit(const cv::Mat &K, const cv::Mat &distortion,
			const std::vector<std::pair<unsigned int, cv::Point2f>> &results_center,
			const std::vector<cv::Point2f> &circles_temp, std::vector<std::pair<int, int>> &matches);

		//ƥ������ͼƬ��Ӧ��
		bool matchCorrespondence(const std::vector<std::pair<int, int>> &matches_left_, const std::vector<std::pair<int, int>> &matches_right_,
			std::vector<std::pair<int, int>> &matches_);

		bool matchCorrespondence(const std::vector<std::pair<int, int>> &matches_left_, const std::vector<std::pair<int, int>> &matches_right_, std::map<int, std::pair<int, int>> &matches_);

		//�����־����ά��,������任����
		bool stereoReconstruction(const std::vector<cv::Point2f> &circles_L, const std::vector<cv::Point2f> &circles_R, const std::map<int, std::pair<int, int>> &matches, std::vector<cv::Point3f> &result_, Eigen::Matrix4f &RT_);

		//����pnp��λ��
		void solvePnP_rt(const std::vector<cv::Point2f> &circles_, const std::vector<std::pair<int, int>> &matches_, Eigen::Matrix4f &RT_, int flag_cam_ = CAM_LEFT, int flag_pnp_ = cv::SOLVEPNP_EPNP);
	};
}
