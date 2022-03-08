//
//  APAP_Stitching.cpp
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#include "APAP_Stitching.h"

/// <summary>
/// 这个方法：
/// 1.将特征点们，进行归一化和标准化
/// 2.将变换后的特征点们，进行DLT，SVD算出每个网格的H
/// 3.将原来网格位置进行H投影变换，得到投影后的网格点位置，放到_dst数据中。
/// <image url="$(ProjectDir)CommentImage\APAP_Stitching_apap_project.png" scale="1"/>
/// </summary>
/// <param name="_p_src">第一张图像的特征点列表</param>
/// <param name="_p_dst">第二张图像的特征点列表</param>
/// <param name="_src">第一张图像的网格点的points</param>
/// <param name="_dst">第一张图像的变形后的网格点points</param>
/// <param name="_homographies">存放每个网格点生成的H</param>
void APAP_Stitching::apap_project(const vector<Point2>& _p_src,
	const vector<Point2>& _p_dst,
	const vector<Point2>& _src,
	vector<Point2>& _dst,
	vector<Mat>& _homographies) {

	//nf1,nf2:图像一/二 特征点对应的归一化后的数据;
	//cf1,cf2:标准化后的数据;
	vector<Point2> nf1, nf2, cf1, cf2;
	//N1,N2:图一图二的归一化矩阵;
	Mat N1, N2, C1, C2;

	//N1获取矩阵(归一化矩阵);nf1:图像特征点对应归一化后的点数据;
	N1 = getNormalize2DPts(_p_src, nf1);
	N2 = getNormalize2DPts(_p_dst, nf2);

	//获取一个与标准差和变异系数有关的矩阵(); 将归一化后的数据进行标准化.
	C1 = getConditionerFromPts(nf1);
	C2 = getConditionerFromPts(nf2);
	cf1.reserve(nf1.size());
	cf2.reserve(nf2.size());

	//cf1:图一放:Point( ( 根2 * (X标后 - X标后平均值)/X标后 标准差 ) , ( 根2 * (Y标后 - Y标后平均值)/Y标后 标准差 )  ) 列表
	//cf2:图二放:Point( ( 根2 * (X标后 - X标后平均值)/X标后 标准差 ) , ( 根2 * (Y标后 - Y标后平均值)/Y标后 标准差 )  ) 于cf1中点 对应 的点
	//标准化之后权值控制在了0-根2;
	for (int i = 0; i < nf1.size(); ++i) {
		cf1.emplace_back(nf1[i].x * C1.at<double>(0, 0) + C1.at<double>(0, 2),
			nf1[i].y * C1.at<double>(1, 1) + C1.at<double>(1, 2));

		cf2.emplace_back(nf2[i].x * C2.at<double>(0, 0) + C2.at<double>(0, 2),
			nf2[i].y * C2.at<double>(1, 1) + C2.at<double>(1, 2));
	}

	double sigma_inv_2 = 1. / (APAP_SIGMA * APAP_SIGMA), gamma = APAP_GAMMA;
	//MatrixXd : 类型为double ,动态大小的类型. 传入宽高无用;
	MatrixXd A = MatrixXd::Zero(cf1.size() * DIMENSION_2D,
		HOMOGRAPHY_VARIABLES_COUNT);

#ifndef DP_NO_LOG
	if (_dst.empty() == false) {
		_dst.clear();
		printError("F(apap_project) dst is not empty");
	}
	if (_homographies.empty() == false) {
		_homographies.clear();
		printError("F(apap_project) homographies is not empty");
	}
#endif
	_dst.reserve(_src.size());
	_homographies.reserve(_src.size());
	//第一张图像网格点循环
	for (int i = 0; i < _src.size(); ++i) {
		//第1张图像特征点循环
		for (int j = 0; j < _p_src.size(); ++j) {
			//得到每个特征点距离每个网格点的距离.
			Point2 d = _src[i] - _p_src[j];
			double www = MAX(gamma, exp(-sqrt(d.x * d.x + d.y * d.y) * sigma_inv_2));

			//A:每个特征点与网格点i的关系.一个特征点有两行数据;
			A(2 * j, 0) = www * cf1[j].x;
			A(2 * j, 1) = www * cf1[j].y;
			A(2 * j, 2) = www * 1;
			A(2 * j, 6) = www * -cf2[j].x * cf1[j].x;
			A(2 * j, 7) = www * -cf2[j].x * cf1[j].y;
			A(2 * j, 8) = www * -cf2[j].x;

			A(2 * j + 1, 3) = www * cf1[j].x;
			A(2 * j + 1, 4) = www * cf1[j].y;
			A(2 * j + 1, 5) = www * 1;
			A(2 * j + 1, 6) = www * -cf2[j].y * cf1[j].x;
			A(2 * j + 1, 7) = www * -cf2[j].y * cf1[j].y;
			A(2 * j + 1, 8) = www * -cf2[j].y;
		}
		//奇异矩阵分解.求每个
		JacobiSVD<MatrixXd, HouseholderQRPreconditioner> jacobi_svd(A, ComputeThinV);
		MatrixXd V = jacobi_svd.matrixV();
		Mat H(3, 3, CV_64FC1);
		for (int j = 0; j < V.rows(); ++j) {
			H.at<double>(j / 3, j % 3) = V(j, V.rows() - 1);
		}
		H = C2.inv() * H * C1;
		H = N2.inv() * H * N1;

		//将src的网格点移动至dst，对网格点进行投影变换
		_dst.emplace_back(applyTransform3x3(_src[i].x, _src[i].y, H));
		_homographies.emplace_back(H);
	}
}
