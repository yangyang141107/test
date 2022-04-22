#include "pcl.h"
// Eigen3 矩阵
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include<stdio.h>
#include<malloc.h>
#include <math.h>
#include <iostream>//输入输出流

#include <opencv2/core/core.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
using namespace std;//标准库　命名空间
using namespace cv;


double* CrossProduct(double a[3], double b[3]);
double DotProduct(double a[3], double b[3]);
double Normalize(double v[3]);
double** RotationMatrix(double angle, double u[3]);

// 利用两个矢量，计算旋转矩阵
void Calculation(double vectorBefore[3], double vectorAfter[3])
{
	double *rotationAxis;
	double rotationAngle;
	double **rotationMatrix;
	rotationAxis = CrossProduct(vectorBefore, vectorAfter);
	rotationAngle = acos(DotProduct(vectorBefore, vectorAfter) / Normalize(vectorBefore) / Normalize(vectorAfter));
	rotationMatrix = RotationMatrix(rotationAngle, rotationAxis);
}

// SVD 分解 求解旋转矩阵（两个点数据集合）
void pose_estimation_3d3d(	const vector<Point3f>& pts1,	const vector<Point3f>& pts2,	Mat& R, Mat& t	)
{
	//【1】 求中心点
	Point3f p1, p2;     //三维点集的中心点  center of mass
	int N = pts1.size(); //点对数量
	for (int i = 0; i<N; i++)
	{
		p1 += pts1[i];//各维度求和
		p2 += pts2[i];
	}
	p1 = Point3f(Vec3f(p1) / N);//求均值 得到中心点
	p2 = Point3f(Vec3f(p2) / N);
	// 【2】得到去中心坐标
	vector<Point3f>     q1(N), q2(N); // remove the center
	for (int i = 0; i<N; i++)
	{
		q1[i] = pts1[i] - p1;
		q2[i] = pts2[i] - p2;
	}

	// 【3】计算需要进行奇异值分解的 W = sum(qi * qi’转置) compute q1*q2^T
	Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
	for (int i = 0; i<N; i++)
	{
		W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
	}
	cout << "W=" << W << endl;

	// 【4】对  W 进行SVD 奇异值分解
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	cout << "U=" << U << endl;
	cout << "V=" << V << endl;
	// 【5】计算旋转 和平移矩阵 R  和 t 
	//  R= U * V转置 
	Eigen::Matrix3d R_ = U* (V.transpose());
	// t =  p - R * p'
	Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

	// 【6】格式转换 convert to cv::Mat
	R = (Mat_<double>(3, 3) <<
		R_(0, 0), R_(0, 1), R_(0, 2),
		R_(1, 0), R_(1, 1), R_(1, 2),
		R_(2, 0), R_(2, 1), R_(2, 2)
		);
	t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));

}



double* CrossProduct(double a[3], double b[3])
{
	double *c = new double[3];

	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];

	return c;
}

double DotProduct(double a[3], double b[3])
{
	double result;
	result = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

	return result;
}

double Normalize(double v[3])
{
	double result;

	result = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

	return result;
}

//函数只能返回变量、结构体和指针
double** RotationMatrix(double angle, double u[3])
{
	double norm = Normalize(u);
	////一维指针（动态申请地址）
	//double *b = (int *)malloc(3*sizeof(double));

	//二维数组 （动态申请地址）
	double **rotatinMatrix = (double **)malloc(3 * sizeof(double *));
	for (int i = 0; i<3; i++)
		rotatinMatrix[i] = (double *)malloc(3 * sizeof(double));//然后依次按一维申请

	u[0] = u[0] / norm;
	u[1] = u[1] / norm;
	u[2] = u[2] / norm;

	rotatinMatrix[0][0] = cos(angle) + u[0] * u[0] * (1 - cos(angle));
	rotatinMatrix[0][1] = u[0] * u[1] * (1 - cos(angle)) - u[2] * sin(angle);
	rotatinMatrix[0][2] = u[1] * sin(angle) + u[0] * u[2] * (1 - cos(angle));

	rotatinMatrix[1][0] = u[2] * sin(angle) + u[0] * u[1] * (1 - cos(angle));
	rotatinMatrix[1][1] = cos(angle) + u[1] * u[1] * (1 - cos(angle));
	rotatinMatrix[1][2] = -u[0] * sin(angle) + u[1] * u[2] * (1 - cos(angle));

	rotatinMatrix[2][0] = -u[1] * sin(angle) + u[0] * u[2] * (1 - cos(angle));
	rotatinMatrix[2][1] = u[0] * sin(angle) + u[1] * u[2] * (1 - cos(angle));
	rotatinMatrix[2][2] = cos(angle) + u[2] * u[2] * (1 - cos(angle));

	return rotatinMatrix;
}