#include "SharedCodedMatching.h"
#include <pcl\visualization\cloud_viewer.h>
#include <pcl\kdtree\kdtree_flann.h>
#include <boost\make_shared.hpp>
#include <pcl\registration\transformation_estimation_svd.h>
#include <pcl\registration\icp.h>

#include <cmath>
#include <complex>
#if defined (_MSC_VER) && (_MSC_VER <= 1700)
static inline double cbrt(double x) { return (double)cv::cubeRoot((float)x); };
#endif

using namespace std;

namespace {
	void solveQuartic(const double *factors, double *realRoots) {
		const double &a4 = factors[0];
		const double &a3 = factors[1];
		const double &a2 = factors[2];
		const double &a1 = factors[3];
		const double &a0 = factors[4];

		double a4_2 = a4 * a4;
		double a3_2 = a3 * a3;
		double a4_3 = a4_2 * a4;
		double a2a4 = a2 * a4;

		double p4 = (8 * a2a4 - 3 * a3_2) / (8 * a4_2);
		double q4 = (a3_2 * a3 - 4 * a2a4 * a3 + 8 * a1 * a4_2) / (8 * a4_3);
		double r4 = (256 * a0 * a4_3 - 3 * (a3_2 * a3_2) - 64 * a1 * a3 * a4_2 + 16 * a2a4 * a3_2) / (256 * (a4_3 * a4));

		double p3 = ((p4 * p4) / 12 + r4) / 3; // /=-3
		double q3 = (72 * r4 * p4 - 2 * p4 * p4 * p4 - 27 * q4 * q4) / 432; // /=2

		double t; // *=2
		complex<double> w;
		if (q3 >= 0)
			w = -sqrt(static_cast<complex<double>>(q3 * q3 - p3 * p3 * p3)) - q3;
		else
			w = sqrt(static_cast<complex<double>>(q3 * q3 - p3 * p3 * p3)) - q3;
		if (w.imag() == 0.0) {
			w.real(cbrt(w.real()));
			t = 2.0 * (w.real() + p3 / w.real());
		}
		else {
			w = pow(w, 1.0 / 3);
			t = 4.0 * w.real();
		}

		complex<double> sqrt_2m = sqrt(static_cast<complex<double>>(-2 * p4 / 3 + t));
		double B_4A = -a3 / (4 * a4);
		double complex1 = 4 * p4 / 3 + t;
#if defined(__clang__) && defined(__arm__) && (__clang_major__ == 3 || __clang_minor__ == 4) && !defined(__ANDROID__)
		// details: https://github.com/opencv/opencv/issues/11135
		// details: https://github.com/opencv/opencv/issues/11056
		complex<double> complex2 = 2 * q4;
		complex2 = complex<double>(complex2.real() / sqrt_2m.real(), 0);
#else
		complex<double> complex2 = 2 * q4 / sqrt_2m;
#endif
		double sqrt_2m_rh = sqrt_2m.real() / 2;
		double sqrt1 = sqrt(-(complex1 + complex2)).real() / 2;
		realRoots[0] = B_4A + sqrt_2m_rh + sqrt1;
		realRoots[1] = B_4A + sqrt_2m_rh - sqrt1;
		double sqrt2 = sqrt(-(complex1 - complex2)).real() / 2;
		realRoots[2] = B_4A - sqrt_2m_rh + sqrt2;
		realRoots[3] = B_4A - sqrt_2m_rh - sqrt2;
	}

	void polishQuarticRoots(const double *coeffs, double *roots) {
		const int iterations = 2;
		for (int i = 0; i < iterations; ++i) {
			for (int j = 0; j < 4; ++j) {
				double error =
					(((coeffs[0] * roots[j] + coeffs[1]) * roots[j] + coeffs[2]) * roots[j] + coeffs[3]) * roots[j] +
					coeffs[4];
				double
					derivative =
					((4 * coeffs[0] * roots[j] + 3 * coeffs[1]) * roots[j] + 2 * coeffs[2]) * roots[j] + coeffs[3];
				roots[j] -= error / derivative;
			}
		}
	}

	inline void vect_cross(const double *a, const double *b, double *result) {
		result[0] = a[1] * b[2] - a[2] * b[1];
		result[1] = -(a[0] * b[2] - a[2] * b[0]);
		result[2] = a[0] * b[1] - a[1] * b[0];
	}

	inline double vect_dot(const double *a, const double *b) {
		return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	}

	inline double vect_norm(const double *a) {
		return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
	}

	inline void vect_scale(const double s, const double *a, double *result) {
		result[0] = a[0] * s;
		result[1] = a[1] * s;
		result[2] = a[2] * s;
	}

	inline void vect_sub(const double *a, const double *b, double *result) {
		result[0] = a[0] - b[0];
		result[1] = a[1] - b[1];
		result[2] = a[2] - b[2];
	}

	inline void vect_divide(const double *a, const double d, double *result) {
		result[0] = a[0] / d;
		result[1] = a[1] / d;
		result[2] = a[2] / d;
	}

	inline void mat_mult(const double a[3][3], const double b[3][3], double result[3][3]) {
		result[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
		result[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
		result[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];

		result[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
		result[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
		result[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];

		result[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
		result[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
		result[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];
	}
}

namespace xyNameSpace {

	class ap3p {
	private:
		template<typename T>
		void init_camera_parameters(const cv::Mat &cameraMatrix) {
			cx = cameraMatrix.at<T>(0, 2);
			cy = cameraMatrix.at<T>(1, 2);
			fx = cameraMatrix.at<T>(0, 0);
			fy = cameraMatrix.at<T>(1, 1);
		}

		template<typename OpointType, typename IpointType>
		void extract_points(const cv::Mat &opoints, const cv::Mat &ipoints, std::vector<double> &points) {
			points.clear();
			int npoints = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));
			points.resize(5 * npoints);
			for (int i = 0; i < npoints; i++) {
				points[i * 5] = ipoints.at<IpointType>(i).x * fx + cx;
				points[i * 5 + 1] = ipoints.at<IpointType>(i).y * fy + cy;
				points[i * 5 + 2] = opoints.at<OpointType>(i).x;
				points[i * 5 + 3] = opoints.at<OpointType>(i).y;
				points[i * 5 + 4] = opoints.at<OpointType>(i).z;
			}
		}

		void init_inverse_parameters() {
			inv_fx = 1. / fx;
			inv_fy = 1. / fy;
			cx_fx = cx / fx;
			cy_fy = cy / fy;
		}

		double fx, fy, cx, cy;
		double inv_fx, inv_fy, cx_fx, cy_fy;
	public:
		ap3p() : fx(0), fy(0), cx(0), cy(0), inv_fx(0), inv_fy(0), cx_fx(0), cy_fy(0) {}

		ap3p(cv::Mat cameraMatrix) {
			if (cameraMatrix.depth() == CV_32F)
				init_camera_parameters<float>(cameraMatrix);
			else
				init_camera_parameters<double>(cameraMatrix);
			init_inverse_parameters();
		}

		bool solve(cv::Mat &R, cv::Mat &tvec, const cv::Mat &opoints, const cv::Mat &ipoints) {
			//CV_INSTRUMENT_REGION();

			double rotation_matrix[3][3], translation[3];
			std::vector<double> points;
			if (opoints.depth() == ipoints.depth()) {
				if (opoints.depth() == CV_32F)
					extract_points<cv::Point3f, cv::Point2f>(opoints, ipoints, points);
				else
					extract_points<cv::Point3d, cv::Point2d>(opoints, ipoints, points);
			}
			else if (opoints.depth() == CV_32F)
				extract_points<cv::Point3f, cv::Point2d>(opoints, ipoints, points);
			else
				extract_points<cv::Point3d, cv::Point2f>(opoints, ipoints, points);

			bool result = solve(rotation_matrix, translation, points[0], points[1], points[2], points[3], points[4], points[5],
				points[6], points[7], points[8], points[9], points[10], points[11], points[12], points[13],
				points[14],
				points[15], points[16], points[17], points[18], points[19]);
			cv::Mat(3, 1, CV_64F, translation).copyTo(tvec);
			cv::Mat(3, 3, CV_64F, rotation_matrix).copyTo(R);
			return result;
		}

		int solve(std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &tvecs, const cv::Mat &opoints, const cv::Mat &ipoints) {
			//CV_INSTRUMENT_REGION();

			double rotation_matrix[4][3][3], translation[4][3];
			std::vector<double> points;
			if (opoints.depth() == ipoints.depth()) {
				if (opoints.depth() == CV_32F)
					extract_points<cv::Point3f, cv::Point2f>(opoints, ipoints, points);
				else
					extract_points<cv::Point3d, cv::Point2d>(opoints, ipoints, points);
			}
			else if (opoints.depth() == CV_32F)
				extract_points<cv::Point3f, cv::Point2d>(opoints, ipoints, points);
			else
				extract_points<cv::Point3d, cv::Point2f>(opoints, ipoints, points);

			int solutions = solve(rotation_matrix, translation,
				points[0], points[1], points[2], points[3], points[4],
				points[5], points[6], points[7], points[8], points[9],
				points[10], points[11], points[12], points[13], points[14]);

			for (int i = 0; i < solutions; i++) {
				cv::Mat R, tvec;
				cv::Mat(3, 1, CV_64F, translation[i]).copyTo(tvec);
				cv::Mat(3, 3, CV_64F, rotation_matrix[i]).copyTo(R);

				Rs.push_back(R);
				tvecs.push_back(tvec);
			}

			return solutions;
		}

		int solve(double R[4][3][3], double t[4][3],
			double mu0, double mv0, double X0, double Y0, double Z0,
			double mu1, double mv1, double X1, double Y1, double Z1,
			double mu2, double mv2, double X2, double Y2, double Z2) {
			double mk0, mk1, mk2;
			double norm;

			mu0 = inv_fx * mu0 - cx_fx;
			mv0 = inv_fy * mv0 - cy_fy;
			norm = sqrt(mu0 * mu0 + mv0 * mv0 + 1);
			mk0 = 1. / norm;
			mu0 *= mk0;
			mv0 *= mk0;

			mu1 = inv_fx * mu1 - cx_fx;
			mv1 = inv_fy * mv1 - cy_fy;
			norm = sqrt(mu1 * mu1 + mv1 * mv1 + 1);
			mk1 = 1. / norm;
			mu1 *= mk1;
			mv1 *= mk1;

			mu2 = inv_fx * mu2 - cx_fx;
			mv2 = inv_fy * mv2 - cy_fy;
			norm = sqrt(mu2 * mu2 + mv2 * mv2 + 1);
			mk2 = 1. / norm;
			mu2 *= mk2;
			mv2 *= mk2;

			double featureVectors[3][3] = { { mu0, mu1, mu2 },
			{ mv0, mv1, mv2 },
			{ mk0, mk1, mk2 } };
			double worldPoints[3][3] = { { X0, X1, X2 },
			{ Y0, Y1, Y2 },
			{ Z0, Z1, Z2 } };

			return computePoses(featureVectors, worldPoints, R, t);
		}

		bool solve(double R[3][3], double t[3],
			double mu0, double mv0, double X0, double Y0, double Z0,
			double mu1, double mv1, double X1, double Y1, double Z1,
			double mu2, double mv2, double X2, double Y2, double Z2,
			double mu3, double mv3, double X3, double Y3, double Z3) {
			double Rs[4][3][3], ts[4][3];

			int n = solve(Rs, ts, mu0, mv0, X0, Y0, Z0, mu1, mv1, X1, Y1, Z1, mu2, mv2, X2, Y2, Z2);
			if (n == 0)
				return false;

			int ns = 0;
			double min_reproj = 0;
			for (int i = 0; i < n; i++) {
				double X3p = Rs[i][0][0] * X3 + Rs[i][0][1] * Y3 + Rs[i][0][2] * Z3 + ts[i][0];
				double Y3p = Rs[i][1][0] * X3 + Rs[i][1][1] * Y3 + Rs[i][1][2] * Z3 + ts[i][1];
				double Z3p = Rs[i][2][0] * X3 + Rs[i][2][1] * Y3 + Rs[i][2][2] * Z3 + ts[i][2];
				double mu3p = cx + fx * X3p / Z3p;
				double mv3p = cy + fy * Y3p / Z3p;
				double reproj = (mu3p - mu3) * (mu3p - mu3) + (mv3p - mv3) * (mv3p - mv3);
				if (i == 0 || min_reproj > reproj) {
					ns = i;
					min_reproj = reproj;
				}
			}

			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++)
					R[i][j] = Rs[ns][i][j];
				t[i] = ts[ns][i];
			}

			return true;
		}

		// This algorithm is from "Tong Ke, Stergios Roumeliotis, An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (Accepted by CVPR 2017)
		// See https://arxiv.org/pdf/1701.08237.pdf
		// featureVectors: 3 bearing measurements (normalized) stored as column vectors
		// worldPoints: Positions of the 3 feature points stored as column vectors
		// solutionsR: 4 possible solutions of rotation matrix of the world w.r.t the camera frame
		// solutionsT: 4 possible solutions of translation of the world origin w.r.t the camera frame
		int computePoses(const double featureVectors[3][3], const double worldPoints[3][3], double solutionsR[4][3][3],
			double solutionsT[4][3]) {

			//world point vectors
			double w1[3] = { worldPoints[0][0], worldPoints[1][0], worldPoints[2][0] };
			double w2[3] = { worldPoints[0][1], worldPoints[1][1], worldPoints[2][1] };
			double w3[3] = { worldPoints[0][2], worldPoints[1][2], worldPoints[2][2] };
			// k1
			double u0[3];
			vect_sub(w1, w2, u0);

			double nu0 = vect_norm(u0);
			double k1[3];
			vect_divide(u0, nu0, k1);
			// bi
			double b1[3] = { featureVectors[0][0], featureVectors[1][0], featureVectors[2][0] };
			double b2[3] = { featureVectors[0][1], featureVectors[1][1], featureVectors[2][1] };
			double b3[3] = { featureVectors[0][2], featureVectors[1][2], featureVectors[2][2] };
			// k3,tz
			double k3[3];
			vect_cross(b1, b2, k3);
			double nk3 = vect_norm(k3);
			vect_divide(k3, nk3, k3);

			double tz[3];
			vect_cross(b1, k3, tz);
			// ui,vi
			double v1[3];
			vect_cross(b1, b3, v1);
			double v2[3];
			vect_cross(b2, b3, v2);

			double u1[3];
			vect_sub(w1, w3, u1);
			// coefficients related terms
			double u1k1 = vect_dot(u1, k1);
			double k3b3 = vect_dot(k3, b3);
			// f1i
			double f11 = k3b3;
			double f13 = vect_dot(k3, v1);
			double f15 = -u1k1 * f11;
			//delta
			double nl[3];
			vect_cross(u1, k1, nl);
			double delta = vect_norm(nl);
			vect_divide(nl, delta, nl);
			f11 *= delta;
			f13 *= delta;
			// f2i
			double u2k1 = u1k1 - nu0;
			double f21 = vect_dot(tz, v2);
			double f22 = nk3 * k3b3;
			double f23 = vect_dot(k3, v2);
			double f24 = u2k1 * f22;
			double f25 = -u2k1 * f21;
			f21 *= delta;
			f22 *= delta;
			f23 *= delta;
			double g1 = f13 * f22;
			double g2 = f13 * f25 - f15 * f23;
			double g3 = f11 * f23 - f13 * f21;
			double g4 = -f13 * f24;
			double g5 = f11 * f22;
			double g6 = f11 * f25 - f15 * f21;
			double g7 = -f15 * f24;
			double coeffs[5] = { g5 * g5 + g1 * g1 + g3 * g3,
				2 * (g5 * g6 + g1 * g2 + g3 * g4),
				g6 * g6 + 2 * g5 * g7 + g2 * g2 + g4 * g4 - g1 * g1 - g3 * g3,
				2 * (g6 * g7 - g1 * g2 - g3 * g4),
				g7 * g7 - g2 * g2 - g4 * g4 };
			double s[4];
			solveQuartic(coeffs, s);
			polishQuarticRoots(coeffs, s);

			double temp[3];
			vect_cross(k1, nl, temp);

			double Ck1nl[3][3] =
			{ { k1[0], nl[0], temp[0] },
			{ k1[1], nl[1], temp[1] },
			{ k1[2], nl[2], temp[2] } };

			double Cb1k3tzT[3][3] =
			{ { b1[0], b1[1], b1[2] },
			{ k3[0], k3[1], k3[2] },
			{ tz[0], tz[1], tz[2] } };

			double b3p[3];
			vect_scale((delta / k3b3), b3, b3p);

			int nb_solutions = 0;
			for (int i = 0; i < 4; ++i) {
				double ctheta1p = s[i];
				if (abs(ctheta1p) > 1)
					continue;
				double stheta1p = sqrt(1 - ctheta1p * ctheta1p);
				stheta1p = (k3b3 > 0) ? stheta1p : -stheta1p;
				double ctheta3 = g1 * ctheta1p + g2;
				double stheta3 = g3 * ctheta1p + g4;
				double ntheta3 = stheta1p / ((g5 * ctheta1p + g6) * ctheta1p + g7);
				ctheta3 *= ntheta3;
				stheta3 *= ntheta3;

				double C13[3][3] =
				{ { ctheta3, 0, -stheta3 },
				{ stheta1p * stheta3, ctheta1p, stheta1p * ctheta3 },
				{ ctheta1p * stheta3, -stheta1p, ctheta1p * ctheta3 } };

				double temp_matrix[3][3];
				double R[3][3];
				mat_mult(Ck1nl, C13, temp_matrix);
				mat_mult(temp_matrix, Cb1k3tzT, R);

				// R' * p3
				double rp3[3] =
				{ w3[0] * R[0][0] + w3[1] * R[1][0] + w3[2] * R[2][0],
				w3[0] * R[0][1] + w3[1] * R[1][1] + w3[2] * R[2][1],
				w3[0] * R[0][2] + w3[1] * R[1][2] + w3[2] * R[2][2] };

				double pxstheta1p[3];
				vect_scale(stheta1p, b3p, pxstheta1p);

				vect_sub(pxstheta1p, rp3, solutionsT[nb_solutions]);

				solutionsR[nb_solutions][0][0] = R[0][0];
				solutionsR[nb_solutions][1][0] = R[0][1];
				solutionsR[nb_solutions][2][0] = R[0][2];
				solutionsR[nb_solutions][0][1] = R[1][0];
				solutionsR[nb_solutions][1][1] = R[1][1];
				solutionsR[nb_solutions][2][1] = R[1][2];
				solutionsR[nb_solutions][0][2] = R[2][0];
				solutionsR[nb_solutions][1][2] = R[2][1];
				solutionsR[nb_solutions][2][2] = R[2][2];

				nb_solutions++;
			}

			return nb_solutions;
		}

	};


	SharedCodedMatching::SharedCodedMatching(const cv::String &camPath_)
	{
		//相机参数
		cv::FileStorage fs;
		if (!fs.open(camPath_, cv::FileStorage::READ))
		{
			cerr << "cannot open the file of camParams!" << endl;
			return;
		}

		cv::Mat camLeft_temp, camRight_temp;
		fs["CamL_Intrinsic_Parameters"] >> camLeft_temp;
		fs["CamL_Distortion"] >> distortionL;

		fs["CamR_Intrinsic_Parameters"] >> camRight_temp;
		fs["CamR_Distortion"] >> distortionR;

		fs["R_Parameter"] >> R_extrinsic;
		fs["T_Parameter"] >> T_extrinsic;

		camLeft = (cv::Mat_<double>(3, 3) << camLeft_temp.ptr<double>(0)[0], 0, camLeft_temp.ptr<double>(2)[0],
			0, camLeft_temp.ptr<double>(1)[0], camLeft_temp.ptr<double>(3)[0],
			0, 0, 1);

		camRight = (cv::Mat_<double>(3, 3) << camRight_temp.ptr<double>(0)[0], 0, camRight_temp.ptr<double>(2)[0],
			0, camRight_temp.ptr<double>(1)[0], camRight_temp.ptr<double>(3)[0],
			0, 0, 1);

		fs.release();
	}

	bool SharedCodedMatching::set_camParas(const cv::String & camPath_)
	{
		//相机参数
		cv::FileStorage fs;
		if (!fs.open(camPath_, cv::FileStorage::READ))
		{
			cerr << "cannot open the file of camParams!" << endl;
			return false;
		}

		cv::Mat camLeft_temp, camRight_temp;
		fs["CamL_Intrinsic_Parameters"] >> camLeft_temp;
		fs["CamL_Distortion"] >> distortionL;

		fs["CamR_Intrinsic_Parameters"] >> camRight_temp;
		fs["CamR_Distortion"] >> distortionR;

		fs["R_Parameter"] >> R_extrinsic;
		fs["T_Parameter"] >> T_extrinsic;

		camLeft = (cv::Mat_<double>(3, 3) << camLeft_temp.ptr<double>(0)[0], 0, camLeft_temp.ptr<double>(2)[0],
			0, camLeft_temp.ptr<double>(1)[0], camLeft_temp.ptr<double>(3)[0],
			0, 0, 1);

		camRight = (cv::Mat_<double>(3, 3) << camRight_temp.ptr<double>(0)[0], 0, camRight_temp.ptr<double>(2)[0],
			0, camRight_temp.ptr<double>(1)[0], camRight_temp.ptr<double>(3)[0],
			0, 0, 1);

		fs.release();
		return true;
	}

	void SharedCodedMatching::set_cloud_all(const pcl::PointCloud<pcl::PointXYZ>& cloud_all_)
	{
		pcl::copyPointCloud(cloud_all_, cloud_all);

		cv::Point3f temp;
		for (auto i : cloud_all_.points)
		{
			temp.x = i.x;
			temp.y = i.y;
			temp.z = i.z;
			cv_CloudAll.emplace_back(temp);
		}
	}

	bool SharedCodedMatching::set_codedPointXYZ(const pcl::PointCloud<pcl::PointXYZ>& coded_, const std::vector<uint>& number_)
	{
		if (coded_.points.size() != number_.size())
		{
			cout << "The number of coded points and numbers are different!" << endl;
			return false;
		}

		for (int i = 0; i < number_.size(); ++i)
		{
			codedPointXYZ[number_[i]] = coded_.points[i];
		}

		return true;
	}

	bool SharedCodedMatching::process(const cv::Mat & left_, const cv::Mat & right_, std::vector<cv::Point3f>& result_, Eigen::Matrix4f & RT_, bool showImg_)
	{
		std::vector<std::pair<unsigned int, cv::Point2f>> results_centerL, results_centerR;
		std::vector<cv::Point2f> circles_L, circles_R;

		try {
			if (!findCircularMarker15(left_, results_centerL, circles_L))
			{
				cout << "No circlcs finding." << endl;
				return false;
			}


			if (!findCircularMarker15(right_, results_centerR, circles_R))
			{
				cout << "No circlcs finding." << endl;
				return false;
			}
		}
		catch (...)
		{
			cout << "Find CircularMarker is errr!." << endl;
			return false;
		}


		std::vector<pair<int, int>> matchesL, matchesR;

		if (showImg_)
		{
			if (!posit(left_, camLeft, distortionL, results_centerL, circles_L, matchesL))
				return false;

			if (!posit(right_, camRight, distortionR, results_centerR, circles_R, matchesR))
				return false;
		}
		else
		{
			if (!posit(camLeft, distortionL, results_centerL, circles_L, matchesL))
			{
				cout << "Cannot posit." << endl;
				return false;
			}

			if (!posit(camRight, distortionR, results_centerR, circles_R, matchesR))
			{
				cout << "Cannot posit." << endl;
				return false;
			}
		}

		std::map<int, pair<int, int>> results_matches;

		if (!matchCorrespondence(matchesL, matchesR, results_matches))
		{
			cout << "Cannot match." << endl;
			return false;
		}




		if (!stereoReconstruction(circles_L, circles_R, results_matches, result_, RT_))
			return false;


		//cout << RT_ << endl;
		return true;
	}

	bool SharedCodedMatching::process_pnp(const cv::Mat & img_, Eigen::Matrix4f & RT_, int flag_cam_, int flag_pnp_)
	{
		std::vector<std::pair<unsigned int, cv::Point2f>> results_center;
		std::vector<cv::Point2f> circles_;

		try {
			if (!findCircularMarker15(img_, results_center, circles_))
			{
				cout << "No circlcs finding." << endl;
				return false;
			}



		}
		catch (...)
		{
			cout << "Find CircularMarker is errr!." << endl;
			return false;
		}


		std::vector<pair<int, int>> matches;

		if (flag_cam_ == CAM_LEFT)
		{
			if (!posit(camLeft, distortionL, results_center, circles_, matches))
			{
				cout << "Cannot posit." << endl;
				return false;
			}
		}
		else if (flag_cam_ == CAM_RIGHT)
		{
			if (!posit(camRight, distortionR, results_center, circles_, matches))
			{
				cout << "Cannot posit." << endl;
				return false;
			}
		}

		solvePnP_rt(circles_, matches, RT_, flag_cam_, flag_pnp_);



		return true;
	}

	bool SharedCodedMatching::readTxtCloudFile(const std::string filename, pcl::PointCloud<pcl::PointXYZ>::Ptr & pnts)
	{
		filebuf *pbuf;
		ifstream fileread;
		long size;
		char * buffer;
		// 要读入整个文件，必须采用二进制打开   
		fileread.open(filename, ios::binary);
		// 获取filestr对应buffer对象的指针（获得这个流对象的指针）
		pbuf = fileread.rdbuf();
		// 调用buffer对象方法获取文件大小  （当复位位置指针指向文件缓冲区的末尾时候pubseekoff返回的值就是整个文件流大小）
		size = pbuf->pubseekoff(0, ios::end, ios::in);
		pbuf->pubseekpos(0, ios::in);   //再让pbuf指向文件流开始位置
		// 分配内存空间  
		buffer = new char[size];
		// 获取文件内容  
		pbuf->sgetn(buffer, size);
		fileread.close();
		// 输出到标准输出  
		//cout.write(buffer, size);
		//最佳方法按字符遍历整个buffer
		string temp = "";
		pcl::PointXYZ Pnts;
		float x = 0;
		float y = 0;
		float z = 0;
		bool isy = false;
		while (*buffer != '\0')
		{
			if (*buffer != '\n' && *buffer != '\r')
			{
				if (*buffer != ' ' && *buffer != ',')
				{
					temp += *buffer;
				}
				else
				{
					if (!isy)  //如果是x的值
					{
						if (!temp.empty())
						{
							isy = !isy;
							sscanf(temp.data(), "%f", &x);
							Pnts.x = x;
							temp = "";
						}
					}
					else                  //如果是y的值
					{
						if (!temp.empty())
						{
							isy = !isy;
							sscanf(temp.data(), "%f", &y);
							Pnts.y = y;
							temp = "";
						}
					}
				}
			}
			else   //这里是z
			{
				if (!temp.empty())
				{
					sscanf(temp.data(), "%f", &z);
					Pnts.z = z;
					temp = "";
					pnts->push_back(Pnts);
				}
			}
			buffer++;
		}
		return true;
	}

	bool SharedCodedMatching::saveTxtCloudFile(const std::string filename, pcl::PointCloud<pcl::PointXYZ>::Ptr & pnts)
	{
		//1.将点云团按x y z格式转换成一个整体字符串buffer
		//2.用ss直接读所有的数字，一起转成string（在读取函数时size大小出错）
		stringstream ss;
		for (int i = 0; i < pnts->size(); i++)
		{
			ss << pnts->points[i].x << " " << pnts->points[i].y << " " << pnts->points[i].z << endl;
		}
		string buffer(ss.str());
		ofstream saveCloud;
		saveCloud.open(filename, ios::binary);
		saveCloud << buffer;
		saveCloud.close();
		return true;
	}



	bool SharedCodedMatching::posit(const cv::Mat & img, const cv::Mat & K, const cv::Mat & distortion, const std::vector<std::pair<unsigned int, cv::Point2f>>& results_center, const std::vector<cv::Point2f>& circles_temp, std::vector<std::pair<int, int>>& matches)
	{
		matches.clear();

		//先存入对应的编码标志点的三维点和二维点
		vector<cv::Point3f> vecPoint3f(3);

		vector<cv::Point2f> vecPoint2f(3);
		int numAdd = 0;
		for (auto i : results_center)
		{
			//没有的点直接跳过
			if (codedPointXYZ.find(i.first) == codedPointXYZ.end())
				continue;

			vecPoint3f[numAdd].x = codedPointXYZ[i.first].x;
			vecPoint3f[numAdd].y = codedPointXYZ[i.first].y;
			vecPoint3f[numAdd].z = codedPointXYZ[i.first].z;

			vecPoint2f[numAdd].x = i.second.x;
			vecPoint2f[numAdd].y = i.second.y;
			++numAdd;
			if (numAdd >= 2)
				break;
		}

		//如果没有相同的编码标志点，匹配失败
		if (numAdd == 0)
			return false;

		//创建kd树，用来寻找编码标志点附近最近的几个点
		pcl::KdTreeFLANN<pcl::PointXYZ> KDtree_coded;
		KDtree_coded.setInputCloud(boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(cloud_all));

		int k = kd_num;
		vector<int> v_id(k);	//用于保存这几个点的id
		vector<float> v_dist(k);//用于保存这几个点到指定点的距离平方

		pcl::PointXYZ coded_temp;
		locatingNearCenter(results_center, K, coded_temp);
		//coded_temp.x = vecPoint3f[0].x; coded_temp.y = vecPoint3f[0].y; coded_temp.z = vecPoint3f[0].z;

		if (KDtree_coded.nearestKSearch(coded_temp, k, v_id, v_dist) == 0)
		{
			cerr << "未找到最近的点！" << endl;
			return false;
		}

		//要投影的三维点
		vector<cv::Point3f> vecProPoints(k);
		for (int i = 0; i < k; ++i)
		{
			vecProPoints[i].x = cloud_all.points[v_id[i]].x;
			vecProPoints[i].y = cloud_all.points[v_id[i]].y;
			vecProPoints[i].z = cloud_all.points[v_id[i]].z;
		}

		map<vector<int>, bool> matched_vec;

		//方法二，遍历挑选三维点，没有随机性，只是严格按照最近的点开始选取
		for (int i_3d = 0; i_3d < k; ++i_3d)
		{
			for (int j_3d = 0; j_3d < k; ++j_3d)
			{
				//如果只需挑一个点，就进行一次循环
				if ((numAdd == 2) && (i_3d > 0))
					break;

				if (i_3d == j_3d)
					continue;
				vector<int> vecIntAdd;
				if (numAdd == 1)
					vecIntAdd.emplace_back(i_3d);

				vecIntAdd.emplace_back(j_3d);

				if (matched_vec.find(vecIntAdd) != matched_vec.end())
					continue;
				else
					matched_vec[vecIntAdd] = true;

				if (matched_vec.size() >= ((k)*(k - 1)))
					break;

				for (int i = 0; i < vecIntAdd.size(); ++i)
				{
					vecPoint3f[numAdd + i].x = cloud_all.points[v_id[vecIntAdd[i]]].x;
					vecPoint3f[numAdd + i].y = cloud_all.points[v_id[vecIntAdd[i]]].y;
					vecPoint3f[numAdd + i].z = cloud_all.points[v_id[vecIntAdd[i]]].z;
				}

				//遍历挑选图像上的二维点
				for (int i = 0; i < circles_temp.size(); ++i)
				{
					if (vecIntAdd.size() == 1)
					{
						vecPoint2f[2].x = circles_temp[i].x;
						vecPoint2f[2].y = circles_temp[i].y;

						//利用pnp求解相机位姿pose
						vector<cv::Mat> vecR, vecT;
						cv::Mat R1, T1;
						ap3p solvep3p(K);

						cv::InputArray input1 = vecPoint3f;
						cv::InputArray input2 = vecPoint2f;


						cv::Mat matVecPoint3f = input1.getMat();
						cv::Mat matVecPoint2f = input2.getMat();

						cv::Mat cameraMatrix = cv::Mat_<double>(K);
						cv::Mat distCoeffs = cv::Mat_<double>(distortion);

						//注意，这里一定要重新定义一个mat来存储畸变矫正后数据，否则会改变上面有一个步骤的浅拷贝
						cv::Mat matVecPoint2f_undistor;
						cv::undistortPoints(matVecPoint2f, matVecPoint2f_undistor, cameraMatrix, distCoeffs);
						solvep3p.solve(vecR, vecT, matVecPoint3f, matVecPoint2f_undistor);

						for (int i_r = 0; i_r < vecR.size(); ++i_r) {
							for (int i_t = 0; i_t < vecT.size(); ++i_t) {
								cv::Mat R, T;
								R = vecR[i_r].clone();
								T = vecT[i_t].clone();

								//将三维点投影为二维点
								vector<cv::Point2f> imgPoint;
								projectPoints(cv_CloudAll, R, T, K, distortion, imgPoint);

								//匹配
								int count = 0;
								vector<pair<int, int>> matches_temp;
								count = matchPoints(imgPoint, circles_temp, matches_temp, reproError);//重投影二维点与图像二维点的匹配

								if (count >= circles_temp.size() * thresMatch)
								{
									cv::Mat img_;
									cvtColor(img, img_, cv::COLOR_GRAY2BGR);
									//在图上画出点
									for (auto i : imgPoint) {
										circle(img_, i, 5, cv::Scalar(255, 0, 0), 2);

									}

									cv::namedWindow("111", cv::WINDOW_FREERATIO);
									imshow("111", img_);
									cv::waitKey(0);
									cv::destroyWindow("111");


									for (auto i_matches : matches_temp)
									{
										matches.emplace_back(make_pair(i_matches.first, i_matches.second));
									}

									return true;
								}
							}
						}
					}
					else
					{
						for (int j = 0; j < circles_temp.size(); ++j)
						{
							if (i == j)
								continue;
							vecPoint2f[1].x = circles_temp[i].x;
							vecPoint2f[1].y = circles_temp[i].y;
							vecPoint2f[2].x = circles_temp[j].x;
							vecPoint2f[2].y = circles_temp[j].y;

							//利用pnp求解相机位姿pose
							vector<cv::Mat> vecR, vecT;
							cv::Mat R1, T1;
							ap3p solvep3p(K);

							cv::InputArray input1 = vecPoint3f;
							cv::InputArray input2 = vecPoint2f;


							cv::Mat matVecPoint3f = input1.getMat();
							cv::Mat matVecPoint2f = input2.getMat();

							cv::Mat cameraMatrix = cv::Mat_<double>(K);
							cv::Mat distCoeffs = cv::Mat_<double>(distortion);

							//注意，这里一定要重新定义一个mat来存储畸变矫正后数据，否则会改变上面有一个步骤的浅拷贝
							cv::Mat matVecPoint2f_undistor;
							cv::undistortPoints(matVecPoint2f, matVecPoint2f_undistor, cameraMatrix, distCoeffs);
							solvep3p.solve(vecR, vecT, matVecPoint3f, matVecPoint2f_undistor);


							//solveP3P(vecPoint3f, vecPoint2f, K, distortion, vecR, vecT, cv::SOLVEPNP_AP3P);

							for (int i_r = 0; i_r < vecR.size(); ++i_r) {
								for (int i_t = 0; i_t < vecT.size(); ++i_t) {
									cv::Mat R, T;
									R = vecR[i_r].clone();
									T = vecT[i_t].clone();


									//将三维点投影为二维点
									vector<cv::Point2f> imgPoint;
									projectPoints(cv_CloudAll, R, T, K, distortion, imgPoint);

									//匹配
									int count = 0;
									vector<pair<int, int>> matches_temp;
									count = matchPoints(imgPoint, circles_temp, matches_temp, reproError);//重投影二维点与图像二维点的匹配

									if (count >= circles_temp.size() * thresMatch)
									{
										cv::Mat img_;
										cvtColor(img, img_, cv::COLOR_GRAY2BGR);
										//在图上画出点
										for (auto i : imgPoint) {
											circle(img_, i, 5, cv::Scalar(255, 0, 0), 2);

										}

										cv::namedWindow("111", cv::WINDOW_FREERATIO);
										imshow("111", img_);
										cv::waitKey(0);
										cv::destroyWindow("111");


										for (auto i_matches : matches_temp)
										{
											matches.emplace_back(make_pair(i_matches.first, i_matches.second));
										}

										return true;
									}
								}
							}
						}
					}
				}
			}
		}

		return false;
	}

	bool SharedCodedMatching::posit(const cv::Mat & K, const cv::Mat & distortion, const std::vector<std::pair<unsigned int, cv::Point2f>>& results_center, const std::vector<cv::Point2f>& circles_temp, std::vector<std::pair<int, int>>& matches)
	{
		matches.clear();

		//先存入对应的编码标志点的三维点和二维点
		vector<cv::Point3f> vecPoint3f(3);

		vector<cv::Point2f> vecPoint2f(3);
		int numAdd = 0;
		for (auto i : results_center)
		{
			//没有的点直接跳过
			if (codedPointXYZ.find(i.first) == codedPointXYZ.end())
				continue;

			vecPoint3f[numAdd].x = codedPointXYZ[i.first].x;
			vecPoint3f[numAdd].y = codedPointXYZ[i.first].y;
			vecPoint3f[numAdd].z = codedPointXYZ[i.first].z;

			vecPoint2f[numAdd].x = i.second.x;
			vecPoint2f[numAdd].y = i.second.y;
			++numAdd;
			if (numAdd >= 2)
				break;
		}

		//如果没有相同的编码标志点，匹配失败
		if (numAdd == 0)
			return false;

		//创建kd树，用来寻找编码标志点附近最近的几个点
		pcl::KdTreeFLANN<pcl::PointXYZ> KDtree_coded;
		KDtree_coded.setInputCloud(boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(cloud_all));

		int k = kd_num;
		vector<int> v_id(k);	//用于保存这几个点的id
		vector<float> v_dist(k);//用于保存这几个点到指定点的距离平方

		pcl::PointXYZ coded_temp;
		locatingNearCenter(results_center, K, coded_temp);
		//coded_temp.x = vecPoint3f[0].x; coded_temp.y = vecPoint3f[0].y; coded_temp.z = vecPoint3f[0].z;

		if (KDtree_coded.nearestKSearch(coded_temp, k, v_id, v_dist) == 0)
		{
			cerr << "未找到最近的点！" << endl;
			return false;
		}

		//要投影的三维点
		vector<cv::Point3f> vecProPoints(k);
		for (int i = 0; i < k; ++i)
		{
			vecProPoints[i].x = cloud_all.points[v_id[i]].x;
			vecProPoints[i].y = cloud_all.points[v_id[i]].y;
			vecProPoints[i].z = cloud_all.points[v_id[i]].z;
		}

		map<vector<int>, bool> matched_vec;

		//方法二，遍历挑选三维点，没有随机性，只是严格按照最近的点开始选取
		for (int i_3d = 0; i_3d < k; ++i_3d)
		{
			for (int j_3d = 0; j_3d < k; ++j_3d)
			{
				//如果只需挑一个点，就进行一次循环
				if ((numAdd == 2) && (i_3d > 0))
					break;

				if (i_3d == j_3d)
					continue;
				vector<int> vecIntAdd;
				if (numAdd == 1)
					vecIntAdd.emplace_back(i_3d);

				vecIntAdd.emplace_back(j_3d);

				if (matched_vec.find(vecIntAdd) != matched_vec.end())
					continue;
				else
					matched_vec[vecIntAdd] = true;

				if (matched_vec.size() >= ((k)*(k - 1)))
					break;

				for (int i = 0; i < vecIntAdd.size(); ++i)
				{
					vecPoint3f[numAdd + i].x = cloud_all.points[v_id[vecIntAdd[i]]].x;
					vecPoint3f[numAdd + i].y = cloud_all.points[v_id[vecIntAdd[i]]].y;
					vecPoint3f[numAdd + i].z = cloud_all.points[v_id[vecIntAdd[i]]].z;
				}

				//遍历挑选图像上的二维点
				for (int i = 0; i < circles_temp.size(); ++i)
				{
					if (vecIntAdd.size() == 1)
					{
						vecPoint2f[2].x = circles_temp[i].x;
						vecPoint2f[2].y = circles_temp[i].y;

						//利用pnp求解相机位姿pose
						vector<cv::Mat> vecR, vecT;
						cv::Mat R1, T1;
						ap3p solvep3p(K);

						cv::InputArray input1 = vecPoint3f;
						cv::InputArray input2 = vecPoint2f;


						cv::Mat matVecPoint3f = input1.getMat();
						cv::Mat matVecPoint2f = input2.getMat();

						cv::Mat cameraMatrix = cv::Mat_<double>(K);
						cv::Mat distCoeffs = cv::Mat_<double>(distortion);

						//注意，这里一定要重新定义一个mat来存储畸变矫正后数据，否则会改变上面有一个步骤的浅拷贝
						cv::Mat matVecPoint2f_undistor;
						cv::undistortPoints(matVecPoint2f, matVecPoint2f_undistor, cameraMatrix, distCoeffs);
						solvep3p.solve(vecR, vecT, matVecPoint3f, matVecPoint2f_undistor);

						for (int i_r = 0; i_r < vecR.size(); ++i_r) {
							for (int i_t = 0; i_t < vecT.size(); ++i_t) {
								cv::Mat R, T;
								R = vecR[i_r].clone();
								T = vecT[i_t].clone();


								//将三维点投影为二维点
								vector<cv::Point2f> imgPoint;
								projectPoints(cv_CloudAll, R, T, K, distortion, imgPoint);

								//匹配
								int count = 0;
								vector<pair<int, int>> matches_temp;
								count = matchPoints(imgPoint, circles_temp, matches_temp, reproError);//重投影二维点与图像二维点的匹配

								if (count >= circles_temp.size() * thresMatch)
								{
									

									//因为matches信息对应的是已经挑选出来的二维点，还要重新匹配上v_id的序号
									for (auto i_matches : matches_temp)
									{
										matches.emplace_back(make_pair(i_matches.first, i_matches.second));
									}

									return true;
								}
							}
						}
					}
					else
					{
						for (int j = 0; j < circles_temp.size(); ++j)
						{
							if (i == j)
								continue;
							vecPoint2f[1].x = circles_temp[i].x;
							vecPoint2f[1].y = circles_temp[i].y;
							vecPoint2f[2].x = circles_temp[j].x;
							vecPoint2f[2].y = circles_temp[j].y;

							//利用pnp求解相机位姿pose
							vector<cv::Mat> vecR, vecT;
							cv::Mat R1, T1;
							ap3p solvep3p(K);

							cv::InputArray input1 = vecPoint3f;
							cv::InputArray input2 = vecPoint2f;


							cv::Mat matVecPoint3f = input1.getMat();
							cv::Mat matVecPoint2f = input2.getMat();

							cv::Mat cameraMatrix = cv::Mat_<double>(K);
							cv::Mat distCoeffs = cv::Mat_<double>(distortion);

							//注意，这里一定要重新定义一个mat来存储畸变矫正后数据，否则会改变上面有一个步骤的浅拷贝
							cv::Mat matVecPoint2f_undistor;
							cv::undistortPoints(matVecPoint2f, matVecPoint2f_undistor, cameraMatrix, distCoeffs);
							solvep3p.solve(vecR, vecT, matVecPoint3f, matVecPoint2f_undistor);

							for (int i_r = 0; i_r < vecR.size(); ++i_r) {
								for (int i_t = 0; i_t < vecT.size(); ++i_t) {
									cv::Mat R, T;
									R = vecR[i_r].clone();
									T = vecT[i_t].clone();


									//将三维点投影为二维点
									vector<cv::Point2f> imgPoint;
									projectPoints(cv_CloudAll, R, T, K, distortion, imgPoint);

									//匹配
									int count = 0;
									vector<pair<int, int>> matches_temp;
									count = matchPoints(imgPoint, circles_temp, matches_temp, reproError);//重投影二维点与图像二维点的匹配

									if (count >= circles_temp.size() * thresMatch)
									{

										//因为matches信息对应的是已经挑选出来的二维点，还要重新匹配上v_id的序号
										for (auto i_matches : matches_temp)
										{
											matches.emplace_back(make_pair(i_matches.first, i_matches.second));
										}

										return true;
									}
								}
							}
						}
					}
				}
			}
		}

		return false;
	}

	int SharedCodedMatching::matchPoints(const std::vector<cv::Point2f>& proPoints, const std::vector<cv::Point2f>& imgPoints, std::vector<std::pair<int, int>>& matches, const double & reproError_)
	{
		int count_matches = 0;
		matches.clear();
		for (int i = 0; i < proPoints.size(); ++i)
		{
			for (int j = 0; j < imgPoints.size(); ++j)
			{
				if ((abs(proPoints[i].x - imgPoints[j].x) < reproError_) && (abs(proPoints[i].y - imgPoints[j].y) < reproError_))
				{
					matches.emplace_back(i, j);
					++count_matches;
					break;
				}
			}
		}


		return count_matches;
	}

	void SharedCodedMatching::locatingNearCenter(const std::vector<std::pair<unsigned int, cv::Point2f>>& results_center_, const cv::Mat & K_, pcl::PointXYZ & point_)
	{
		//计算欧氏距离
		float pixel_x = K_.at<double>(0, 2), pixel_y = K_.at<double>(1, 2);
		auto distance_xy = [pixel_x, pixel_y](cv::Point2f a_) {
			return sqrt((a_.x - pixel_x)*(a_.x - pixel_x) + (a_.y - pixel_y)*(a_.y - pixel_y));
		};
		float distance_min = 100000;

		for (auto i : results_center_)
		{
			auto dis = distance_xy(i.second);
			if (dis < distance_min)
			{
				//先判断编码点中有没有这个检测到编码点
				if (codedPointXYZ.find(i.first) == codedPointXYZ.end())
					continue;
				distance_min = dis;
				point_ = codedPointXYZ[i.first];
			}
		}
	}



	bool SharedCodedMatching::matchCorrespondence(const std::vector<std::pair<int, int>> &matches_left_, const std::vector<std::pair<int, int>> &matches_right_, std::vector<std::pair<int, int>> &matches_)
	{
		for (auto i : matches_left_) {
			for (auto j : matches_right_) {
				if (i.first != j.first)
					continue;
				matches_.emplace_back(i.second, j.second);
				break;
			}
		}
		if (matches_.size() < 3)
			return false;
		else
			return true;
	}

	bool SharedCodedMatching::matchCorrespondence(const std::vector<std::pair<int, int>> &matches_left_, const std::vector<std::pair<int, int>> &matches_right_, std::map<int, std::pair<int, int>> &matches_)
	{
		for (auto i : matches_left_) {
			for (auto j : matches_right_) {
				if (i.first != j.first)
					continue;
				matches_.emplace(std::make_pair(i.first, std::make_pair(i.second, j.second)));
				break;
			}
		}
		if (matches_.size() < 3)
			return false;
		else
			return true;
	}

	bool SharedCodedMatching::stereoReconstruction(const std::vector<cv::Point2f> &circles_L, const std::vector<cv::Point2f> &circles_R, const std::map<int, std::pair<int, int>> &matches, std::vector<cv::Point3f> &result_, Eigen::Matrix4f &RT_)
	{
		//计算三维点
		cv::Mat P1 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0), P2(3, 4, CV_64FC1);
		cv::Mat r_rodrigues;
		cv::Rodrigues(R_extrinsic, r_rodrigues);
		P2.at<double>(0, 0) = r_rodrigues.at<double>(0, 0); P2.at<double>(0, 1) = r_rodrigues.at<double>(0, 1); P2.at<double>(0, 2) = r_rodrigues.at<double>(0, 2); P2.at<double>(0, 3) = T_extrinsic.at<double>(0, 0);
		P2.at<double>(1, 0) = r_rodrigues.at<double>(1, 0);
		P2.at<double>(1, 1) = r_rodrigues.at<double>(1, 1);
		P2.at<double>(1, 2) = r_rodrigues.at<double>(1, 2);
		P2.at<double>(1, 3) = T_extrinsic.at<double>(1, 0);
		P2.at<double>(2, 0) = r_rodrigues.at<double>(2, 0); P2.at<double>(2, 1) = r_rodrigues.at<double>(2, 1); P2.at<double>(2, 2) = r_rodrigues.at<double>(2, 2); P2.at<double>(2, 3) = T_extrinsic.at<double>(2, 0);

		P1 = camLeft * P1;
		P2 = camRight * P2;

		std::vector<cv::Point2f> circles_L_undistortion, circles_R_undistortion;
		cv::undistortPoints(circles_L, circles_L_undistortion, camLeft, distortionL, cv::noArray(), camLeft);
		cv::undistortPoints(circles_R, circles_R_undistortion, camRight, distortionR, cv::noArray(), camRight);

		cv::Mat pointsL = cv::Mat_<double>(2, matches.size());
		cv::Mat pointsR = cv::Mat_<double>(2, matches.size());

		int i_num = 0;
		for (auto i : matches)
		{
			pointsL.at<double>(0, i_num) = circles_L_undistortion[i.second.first].x;
			pointsL.at<double>(1, i_num) = circles_L_undistortion[i.second.first].y;

			pointsR.at<double>(0, i_num) = circles_R_undistortion[i.second.second].x;
			pointsR.at<double>(1, i_num) = circles_R_undistortion[i.second.second].y;
			++i_num;
		}

		cv::Mat out;
		cv::triangulatePoints(P1, P2, pointsL, pointsR, out);

		//cout << "三维点：" << endl;
		//ofstream ofile("./1.txt", ofstream::out);
		for (int i = 0; i < out.cols; ++i)
		{
			double x, y, z;
			x = out.at<double>(0, i) / out.at<double>(3, i);
			y = out.at<double>(1, i) / out.at<double>(3, i);
			z = out.at<double>(2, i) / out.at<double>(3, i);

			result_.emplace_back(cv::Point3f(x, y, z));
			//ofile << x << " " << y << " " << z << " 255 0 0" << endl;
		}

		//计算变换矩阵RT
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);//双目视觉创建点云
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);//摄影测量创建点云

		i_num = 0;
		for (auto i : matches)
		{
			cloud_out->points.emplace_back(cloud_all.points[i.first]);
			pcl::PointXYZ temp;
			temp.x = result_[i_num].x;
			temp.y = result_[i_num].y;
			temp.z = result_[i_num].z;
			cout << " " << temp.x << " " << temp.y << " " << temp.z << endl;
			cloud_in->points.emplace_back(temp);
			++i_num;
		}

		/*一、直接用svd奇异值分解算出转换矩阵*/
		//pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, double> registrationSVD;
		//Eigen::Matrix4d rt_temp;
		//registrationSVD.estimateRigidTransformation(*cloud_in, *cloud_out, RT_);


		/*二、先用svd奇异值算出粗配准，再用icp算出精配准*/
		/*(1)先用svd奇异值分解算出转换矩阵*/
		pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, double> registrationSVD;
		Eigen::Matrix4d rt_temp;
		registrationSVD.estimateRigidTransformation(*cloud_in, *cloud_out, rt_temp);

		pcl::transformPointCloud(*cloud_in, *cloud_in, rt_temp);

		/*(2)icp算出转换矩阵*/
		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		icp.setInputSource(cloud_in);
		icp.setInputTarget(cloud_out);
		icp.setMaxCorrespondenceDistance(1);
		////最大迭代次数
		icp.setMaximumIterations(100);
		////两次变化矩阵之间的差值
		icp.setTransformationEpsilon(1e-2);
		////均方误差
		icp.setEuclideanFitnessEpsilon(0.01);

		pcl::PointCloud<pcl::PointXYZ> Final;
		icp.align(Final);
		//打印信息
		std::cout << "has converaged: " << icp.hasConverged() << std::endl
			<< "score:" << icp.getFitnessScore() << std::endl;
		
		if (!icp.hasConverged())
		{
			std::cerr << "ICP has not converaged!" << std::endl;
			return false;
		}

		RT_ = icp.getFinalTransformation();

		Eigen::Matrix4f rt_svd;
		rt_svd << rt_temp(0, 0), rt_temp(0, 1), rt_temp(0, 2), rt_temp(0, 3),
			rt_temp(1, 0), rt_temp(1, 1), rt_temp(1, 2), rt_temp(1, 3),
			rt_temp(2, 0), rt_temp(2, 1), rt_temp(2, 2), rt_temp(2, 3),
			rt_temp(3, 0), rt_temp(3, 1), rt_temp(3, 2), rt_temp(3, 3);
		RT_ = rt_svd * RT_;
		
		return true;
	}
	void SharedCodedMatching::solvePnP_rt(const std::vector<cv::Point2f> &circles_, const std::vector<std::pair<int, int>> &matches_, Eigen::Matrix4f &RT_, int flag_cam_, int flag_pnp_)
	{
		std::vector<cv::Point3f> points3f;
		std::vector<cv::Point2f> points2f;

		for (auto i : matches_)
		{
			points3f.emplace_back(cv_CloudAll[i.first]);
			points2f.emplace_back(circles_[i.second]);
			//if (points3f.size() > 4)break;
		}

		//这里用不同的pnp方法对结果影响很大
		cv::Mat R_, T_;
		bool flag_success;
		if (flag_cam_ == CAM_LEFT)
		{
			flag_success = cv::solvePnP(points3f, points2f, camLeft, distortionL, R_, T_, false, flag_pnp_);
		}
		else if (flag_cam_ == CAM_RIGHT)
		{
			flag_success = cv::solvePnP(points3f, points2f, camRight, distortionR, R_, T_, false, flag_pnp_);
		}

		cv::Rodrigues(R_, R_);
		R_ = R_.inv();
		T_ = R_ * (-T_);

		//R_.convertTo(R_, CV_32FC1);
		//T_.convertTo(T_, CV_32FC1);

		RT_ << R_.at<double>(0, 0), R_.at<double>(0, 1), R_.at<double>(0, 2), T_.at<double>(0, 0),
			R_.at<double>(1, 0), R_.at<double>(1, 1), R_.at<double>(1, 2), T_.at<double>(1, 0),
			R_.at<double>(2, 0), R_.at<double>(2, 1), R_.at<double>(2, 2), T_.at<double>(2, 0),
			0, 0, 0, 1;

	}
}
