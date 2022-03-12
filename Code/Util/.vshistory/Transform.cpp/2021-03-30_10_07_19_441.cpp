//
//  Transform.cpp
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#include "Transform.h"

/// <summary>
/// 根据特征点的权值列表,计算出权值的总体标准差,最后得到了一个3x3的矩阵(与均值和标准差相关);
/// 变异系数=标准差 / 均值:
/// 反映单位均值上的离散程度，常用在两个总体均值不等的离散程度的比较上。
/// 若两个总体的均值相等，则比较标变异系数与比较标准差是等价的。
/// 
/// </summary>
/// <param name="pts">特征点的权值列表</param>
/// <returns>
///	result:
///		[根2/X的标准差 ,     0 ,       -根2/X的标准差* 中心点的X;   
///			 0,        -根2/Y的标准差,   -根2/X的标准差* 中心点的Y;
///			 0,             0 ,               1     ]
/// </returns>
Mat getConditionerFromPts(const vector<Point2>& pts) {
	//pts_ref:n行1列2通道
	Mat pts_ref(pts);
	//mean_pts:x,y的各个平均值;std_pts:x,y的各个样本标准差;
	Scalar mean_pts, std_pts;
	//求 X和Y 上的所有权值的平均值 和 样本标准差;
	//mean_pts[0]:每个点 的X的权值 平均值;mean_pts[1]:每个点 的Y的权值平均值; 
	//std_pts[0]:每个点 的X的权值 标准差;mean_pts[1]:每个点 的Y的标准差; 
	meanStdDev(pts_ref, mean_pts, std_pts);

	// 样本方差/n-1 = 样本偏差(无偏估计)   再*n = 总体方差; 
	//std_pts:每个点的 X/Y权值的 总体方差;
	std_pts = (std_pts.mul(std_pts) * pts_ref.rows / (double)(pts_ref.rows - 1));
	//std_pts:每个点的 X/Y权值的 总体标准差;
	sqrt(std_pts, std_pts);

	//防止存在标准差为0的情况?
	std_pts.val[0] = std_pts.val[0] + (std_pts.val[0] == 0);
	std_pts.val[1] = std_pts.val[1] + (std_pts.val[1] == 0);

	/* result:
		[根2/X权值的标准差 ,     0 ,       -根2 * X权值的平均值 / X权值的标准差 ;
			 0,        -根2/Y权值的标准差,   -根2 * Y权值的平均值/ Y权值的标准差;
			 0,             0 ,               1     ]
	*/
	Mat result(3, 3, CV_64FC1);
	result.at<double>(0, 0) = sqrt(2) / (double)std_pts.val[0];
	result.at<double>(0, 1) = 0;
	result.at<double>(0, 2) = -(sqrt(2) / (double)std_pts.val[0]) * (double)mean_pts.val[0];

	result.at<double>(1, 0) = 0;
	result.at<double>(1, 1) = sqrt(2) / (double)std_pts.val[1];
	result.at<double>(1, 2) = -(sqrt(2) / (double)std_pts.val[1]) * (double)mean_pts.val[1];

	result.at<double>(2, 0) = 0;
	result.at<double>(2, 1) = 0;
	result.at<double>(2, 2) = 1;

	return result;
}

/// <summary>
/// 将图像的特征点 都进行运算,得到每个特征点对应的权值;以及一个3x3矩阵(与欧氏距离均值 和 每个点x/y一维与中心的距离 相关)
/// (scale权值= 根号2/ 所有点到中心点距离的 平均值;)
/// </summary>
/// <param name="pts">图像的特征点列表</param>
/// <param name="newpts">图像特征点对应的权值列表</param>
/// <returns>图像加权矩阵
///		[scale ,  0 , -scale*中心点的x;
///		   0,   scale,-scale*中心点的y;
///		   0,     0,        1     ]</returns>
Mat getNormalize2DPts(const vector<Point2>& pts, vector<Point2>& newpts) {
	//Mat的多通道数据结构:[b,g,r,b,g,r,b,g,r;b,g,r,b,g,r,b,g,r]:为2行3列,通道数为3的数据;

	//pts_ref:图像的特征点mat,结构为[1,1;2,2;3,3;...](n行1列,2通道); 
	//npts:为每个特征点与均值点的x,y距离;
	Mat pts_ref(pts), npts;
	//计算出每个通道的均值.scalar是一个数组大小为4的一维数组;
	//经过运算:mean_p[0]=所有特征点x的均值;mean_p[1]=所有特征点y的均值;即特征点中心点;
	Scalar mean_p = mean(pts_ref);
	//npts为每个特征点与均值点的x,y距离.
	npts = pts_ref - mean_p;
	//dist为每个特征点与均值点的x,y距离平方.[dx2,dy2;dx2,dy2;dx2,dy2;...];
	Mat dist = npts.mul(npts);
	//通道数从2变为1,行数不变.那么列数变为2;则变为了:n行2列1通道;
	dist = dist.reshape(1);
	//获得所有特征点与均值之间的欧氏距离
	sqrt(dist.col(0) + dist.col(1), dist);

	//加权:根号2 / 所有点到中心点距离的 平均值.
	double scale = sqrt(2) / mean(dist).val[0];

	/* result:
		[scale , 0 ,  -根2*中心点的X / 所有点到中心点欧式距离的 平均值;
		  0,   scale, -根2*中心点的Y/ 所有点到中心点欧式距离的 平均值;
		  0,     0,        1     ]
	*/
	Mat result(3, 3, CV_64FC1);
	result.at<double>(0, 0) = scale;
	result.at<double>(0, 1) = 0;
	result.at<double>(0, 2) = -scale * (double)mean_p.val[0];

	result.at<double>(1, 0) = 0;
	result.at<double>(1, 1) = scale;
	result.at<double>(1, 2) = -scale * (double)mean_p.val[1];

	result.at<double>(2, 0) = 0;
	result.at<double>(2, 1) = 0;
	result.at<double>(2, 2) = 1;

#ifndef DP_LOG
	if (newpts.empty() == false) {
		newpts.clear();
		printError("F(getNormalize2DPts) newpts is not empty");
	}
#endif
	newpts.reserve(pts.size());
	//newpts存放每个特征点加权后的point2(x',y') : x'=加权*(x到 中心点x 的距离) ,y'= 加权*(y到 中心点y 的距离)
	//即 x'=根2*(x到 中心点x 的距离 / 所有点到中心点距离的 平均值) ,y'= 根2*(y到 中心点y 的距离 / 所有点到中心点距离的 平均值)	
	for (int i = 0; i < pts.size(); ++i) {
		newpts.emplace_back(pts[i].x * result.at<double>(0, 0) + result.at<double>(0, 2),
			pts[i].y * result.at<double>(1, 1) + result.at<double>(1, 2));
	}

	return result;
}

/// <summary>
/// 控制在-180 -- +180度范围内
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="x"></param>
/// <returns></returns>
template <typename T>
T normalizeAngle(T x) {
	x = fmod(x + 180, 360);
	if (x < 0) {
		x += 360;
	}
	return x - 180;
}

/// <summary>
/// 将x,y通过单应性矩阵matT进行投影变换
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="x"></param>
/// <param name="y"></param>
/// <param name="matT"></param>
/// <returns></returns>
template <typename T>
Point_<T> applyTransform3x3(T x, T y, const Mat& matT) {
	double denom = 1. / (matT.at<double>(2, 0) * x + matT.at<double>(2, 1) * y + matT.at<double>(2, 2));
	return Point_<T>((matT.at<double>(0, 0) * x + matT.at<double>(0, 1) * y + matT.at<double>(0, 2)) * denom,
		(matT.at<double>(1, 0) * x + matT.at<double>(1, 1) * y + matT.at<double>(1, 2)) * denom);
}

template <typename T>
Point_<T> applyTransform2x3(T x, T y, const Mat& matT) {
	return Point_<T>((matT.at<double>(0, 0) * x + matT.at<double>(0, 1) * y + matT.at<double>(0, 2)),
		(matT.at<double>(1, 0) * x + matT.at<double>(1, 1) * y + matT.at<double>(1, 2)));
}

template <typename T>
Size_<T> normalizeVertices(vector<vector<Point_<T> > >& vertices) {
	T min_x = FLT_MAX, max_x = -FLT_MAX;
	T min_y = FLT_MAX, max_y = -FLT_MAX;
	//找到最小,最大的 x,y
	for (int i = 0; i < vertices.size(); ++i) {
		for (int j = 0; j < vertices[i].size(); ++j) {
			min_x = min(min_x, vertices[i][j].x);
			min_y = min(min_y, vertices[i][j].y);
			max_x = max(max_x, vertices[i][j].x);
			max_y = max(max_y, vertices[i][j].y);
		}
	}
	//将拼成的大图左上角移动至(0,0)点
	for (int i = 0; i < vertices.size(); ++i) {
		for (int j = 0; j < vertices[i].size(); ++j) {
			vertices[i][j].x = (vertices[i][j].x - min_x);
			vertices[i][j].y = (vertices[i][j].y - min_y);
		}
	}
	return Size_<T>(max_x - min_x, max_y - min_y);
}

template <typename T>
Rect_<T> getVerticesRects(const vector<Point_<T> >& vertices) {
	vector<vector<Point_<T> > > tmp(1, vertices);
	return getVerticesRects(tmp).front();
}

/// <summary>
/// 返回每张图像的4个点坐标.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="vertices"></param>
/// <returns></returns>
template <typename T>
vector<Rect_<T> > getVerticesRects(const vector<vector<Point_<T> > >& vertices) {
	vector<Rect_<T> > result;
	result.reserve(vertices.size());
	for (int i = 0; i < vertices.size(); ++i) {
		T min_ix = FLT_MAX, max_ix = -FLT_MAX;
		T min_iy = FLT_MAX, max_iy = -FLT_MAX;
		for (int j = 0; j < vertices[i].size(); ++j) {
			min_ix = min(min_ix, vertices[i][j].x);
			max_ix = max(max_ix, vertices[i][j].x);
			min_iy = min(min_iy, vertices[i][j].y);
			max_iy = max(max_iy, vertices[i][j].y);
		}
		result.emplace_back(min_ix, min_iy,
			max_ix - min_ix, max_iy - min_iy);
	}
	return result;
}

template <typename T>
T getSubpix(const Mat& img, const Point2f& pt) {
	Mat patch;
	cv::getRectSubPix(img, Size(1, 1), pt, patch);
	return patch.at<T>(0, 0);
}

/// <summary>
/// 获取img 某点的像素点,返回为列向量.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="img"></param>
/// <param name="pt"></param>
/// <returns></returns>
template <typename T, size_t n>
Vec<T, n> getSubpix(const Mat& img, const Point2f& pt) {
	Mat patch;
	cv::getRectSubPix(img, Size(1, 1), pt, patch);
	return patch.at<Vec<T, n> >(0, 0);
}

template <typename T>
Vec<T, 3> getEulerZXYRadians(const Mat_<T>& rot_matrix) {
	const T r00 = rot_matrix.template at<T>(0, 0);
	const T r01 = rot_matrix.template at<T>(0, 1);
	const T r02 = rot_matrix.template at<T>(0, 2);
	const T r10 = rot_matrix.template at<T>(1, 0);
	const T r11 = rot_matrix.template at<T>(1, 1);
	const T r12 = rot_matrix.template at<T>(1, 2);
	const T r22 = rot_matrix.template at<T>(2, 2);

	Vec<T, 3> result;
	if (r12 < 1) {
		if (r12 > -1) {
			result[0] = asin(-r12);
			result[1] = atan2(r02, r22);
			result[2] = atan2(r10, r11);
		}
		else {
			result[0] = M_PI_2;
			result[1] = -atan2(-r01, r00);
			result[2] = 0.;
		}
	}
	else {
		result[0] = -M_PI_2;
		result[1] = -atan2(-r01, r00);
		result[2] = 0.;
	}
	return result;
}

template <typename T>
bool isEdgeIntersection(const Point_<T>& src_1, const Point_<T>& dst_1,
	const Point_<T>& src_2, const Point_<T>& dst_2,
	double* scale_1, double* scale_2) {
	const Point_<T> s1 = dst_1 - src_1, s2 = dst_2 - src_2;
	const double denom = -s2.x * s1.y + s1.x * s2.y;

	if (denom <  std::numeric_limits<double>::epsilon() &&
		denom > -std::numeric_limits<double>::epsilon()) {
		return false;
	}

	double tmp_scale_1 = (s2.x * (src_1.y - src_2.y) - s2.y * (src_1.x - src_2.x)) / denom;
	double tmp_scale_2 = (-s1.y * (src_1.x - src_2.x) + s1.x * (src_1.y - src_2.y)) / denom;

	if (scale_1) *scale_1 = tmp_scale_1;
	if (scale_2) *scale_2 = tmp_scale_2;

	return (tmp_scale_1 >= 0 && tmp_scale_1 <= 1 &&
		tmp_scale_2 >= 0 && tmp_scale_2 <= 1);
}

template <typename T>
bool isRotationInTheRange(const T rotation, const T min_rotation, const T max_rotation) {
	const Point_<T> b(cos(rotation), sin(rotation));
	const Point_<T> a(cos(min_rotation), sin(min_rotation));
	const Point_<T> c(cos(max_rotation), sin(max_rotation));
	const T direction_a_b = a.x * b.y - a.y * b.x;
	const T direction_a_c = a.x * c.y - a.y * c.x;
	const T direction_b_c = b.x * c.y - b.y * c.x;

	return (direction_a_b * direction_a_c >= 0) && (direction_a_b * direction_b_c >= 0);
}

template  float normalizeAngle< float>(float x);
template double normalizeAngle<double>(double x);

template Point_< float> applyTransform3x3< float>(float x, float y, const Mat& matT);
template Point_<double> applyTransform3x3<double>(double x, double y, const Mat& matT);

template Point_< float> applyTransform2x3< float>(float x, float y, const Mat& matT);
template Point_<double> applyTransform2x3<double>(double x, double y, const Mat& matT);

template Size_<   int> normalizeVertices<   int>(vector<vector<Point_<   int> > >& vertices);
template Size_< float> normalizeVertices< float>(vector<vector<Point_< float> > >& vertices);
template Size_<double> normalizeVertices<double>(vector<vector<Point_<double> > >& vertices);

template Rect_< float> getVerticesRects< float>(const vector<Point_< float> >& vertices);
template Rect_<double> getVerticesRects<double>(const vector<Point_<double> >& vertices);

template vector<Rect_< float> > getVerticesRects< float>(const vector<vector<Point_< float> > >& vertices);
template vector<Rect_<double> > getVerticesRects<double>(const vector<vector<Point_<double> > >& vertices);

template          float getSubpix<   float>(const Mat& img, const Point2f& pt);
template Vec< uchar, 1> getSubpix<uchar, 1>(const Mat& img, const Point2f& pt);
template Vec< uchar, 3> getSubpix<uchar, 3>(const Mat& img, const Point2f& pt);

template Vec< float, 3> getEulerZXYRadians< float>(const Mat_< float>& rot_matrix);
template Vec<double, 3> getEulerZXYRadians<double>(const Mat_<double>& rot_matrix);

template bool isEdgeIntersection< float>(const Point_< float>& src_1, const Point_< float>& dst_1,
	const Point_< float>& src_2, const Point_< float>& dst_2,
	double* scale_1, double* scale_2);

template bool isEdgeIntersection<double>(const Point_<double>& src_1, const Point_<double>& dst_1,
	const Point_<double>& src_2, const Point_<double>& dst_2,
	double* scale_1, double* scale_2);

template bool isRotationInTheRange< float>(const  float rotation, const  float min_rotation, const  float max_rotation);
template bool isRotationInTheRange<double>(const double rotation, const double min_rotation, const double max_rotation);
//字符串分割
void SpiltString(string str, vector<string>& res, string delim)
{
	while (res.size() != 0)
		res.pop_back();
	string::size_type pos1, pos2;
	pos2 = str.find(delim);
	pos1 = 0;
	while (string::npos != pos2)
	{
		res.push_back(str.substr(pos1, pos2 - pos1));
		pos1 = pos2 + delim.size();
		pos2 = str.find(delim, pos1);
	}
	if (pos1 != str.length())
	{
		res.push_back(str.substr(pos1));
	}
}

//数据类型转换，string 转换为常用的数值类型  int ， double ， float
template <class Type>
Type stringToNum(string str)
{
	istringstream s(str);
	Type num;
	s >> num;
	return num;
}

/// <summary>
/// 将point转化为3维向量
/// </summary>
/// <param name="point"></param>
/// <returns></returns>
Vector3d trans2Vector(Point2f point) {
	Vector3d vector(point.x, point.y, 0);
	return vector;
}
