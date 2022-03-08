//
//  FeatureController.cpp
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#include "FeatureController.h"

void FeatureDescriptor::addDescriptor(const Mat& _descriptor) {
	data.emplace_back(_descriptor);
}

/// <summary>
/// 计算两个特征点的描述子 之间的距离.
/// 距离就是:两向量item 之间差 的平方.
/// 因为有多个方向,即对应多个特征向量,所以一个个的算向量距离.最后返回最小的距离值.
/// </summary>
/// <param name="_descriptor1">第一个特征点描述子</param>
/// <param name="_descriptor2">另一个特征点描述子</param>
/// <param name="_threshold">阈值,如果距离大于这个阈值,那就认为这俩向量不能匹配为相似.</param>
/// <returns>两个特征点的描述子 最小的差距值.</returns>
double FeatureDescriptor::getDistance(const FeatureDescriptor& _descriptor1,
	const FeatureDescriptor& _descriptor2,
	const double _threshold) {
	const vector<Mat>& data1 = _descriptor1.data;
	const vector<Mat>& data2 = _descriptor2.data;
	double result = FLT_MAX;
	/*因为一个特征点会对应多个方向,即对应多个特征向量,所以一个个的算向量距离.最后返回最小的距离.*/
	for (int i = 0; i < data1.size(); ++i) {
		for (int j = 0; j < data2.size(); ++j) {
			double distance = 0;
			for (int k = 0; k < SIFT_DESCRIPTOR_DIM; ++k) {
				distance += ((data1[i].at<vl_sift_pix>(k) - data2[j].at<vl_sift_pix>(k)) *
					(data1[i].at<vl_sift_pix>(k) - data2[j].at<vl_sift_pix>(k)));

				/* at<vl_sift_pix>(k) == at<vl_sift_pix>(0, k) */

				if (distance >= _threshold) {
					break;
				}
			}
			result = min(result, distance);
		}
	}
	return result;
}

/*传入灰度图,返回该图的特征点,特征点描述.*/
void FeatureController::detect(const Mat& _grey_img,
	vector<Point2>& _feature_points,
	vector<FeatureDescriptor>& _feature_descriptors) {
#ifndef DP_LOG
	if (_feature_points.empty() == false) {
		_feature_points.clear();
		printError("F(detect) feature points is not empty");
	}
	if (_feature_descriptors.empty() == false) {
		_feature_descriptors.clear();
		printError("F(detect) feature descriptors is not empty");
	}
#endif
	//要把图像内容的像素信息转为浮点,来进行矩阵计算.
	Mat grey_img_float = _grey_img.clone();
	grey_img_float.convertTo(grey_img_float, CV_32FC1);

	//图像像素宽高
	const int  width = _grey_img.cols;
	const int height = _grey_img.rows;

	// noctaves: numbers of octaves 组数
	// nlevels: numbers of levels per octave 每组的层数
	// o_min: first octave index 第一组的索引号
	VlSiftFilt* vlSift = vl_sift_new(width, height,
		log2(min(width, height)),     //如:1024x768;9组.
		SIFT_LEVEL_COUNT,
		SIFT_MINIMUM_OCTAVE_INDEX);
	//极值点阈值设置
	vl_sift_set_peak_thresh(vlSift, SIFT_PEAK_THRESH);
	//边缘剔除阈值
	vl_sift_set_edge_thresh(vlSift, SIFT_EDGE_THRESH);

	if (vl_sift_process_first_octave(vlSift, (vl_sift_pix const*)grey_img_float.data) != VL_ERR_EOF) {
		//循环处理每一组.得到每一组的特征点,特征描述.
		do {
			vl_sift_detect(vlSift);
			for (int i = 0; i < vlSift->nkeys; ++i) {
				//将图像这个特征点放入_feature_points 列表
				_feature_points.emplace_back(vlSift->keys[i].x, vlSift->keys[i].y);

				//存放特征点的方向:主/辅 
				double angles[4];
				FeatureDescriptor descriptor;
				// 计算特征点的方向，包括主方向和辅方向，最多4个
				int angleCount = vl_sift_calc_keypoint_orientations(vlSift, angles, &vlSift->keys[i]);
				//对于方向多于一个的特征点，每个方向分别计算特征描述符.
				//因为对于每个方向,计算特征点之前,会将这个方向定为0度(起始角度).
				//这样不同的起始角度,就有不同的图像所有的点的方向数据都会变化,进而特征点方向不同,特征描述向量也不同.
				for (int j = 0; j < angleCount; ++j) {
					//存放一个特征点一个方向的描述向量
					Mat descriptor_array(1, SIFT_DESCRIPTOR_DIM, CV_32FC1);
					//计算出一个特征点一个方向的描述向量
					vl_sift_calc_keypoint_descriptor(vlSift, (vl_sift_pix*)descriptor_array.data, &vlSift->keys[i], angles[j]);
					//加入这个特征点的 描述向量 列表.
					descriptor.addDescriptor(descriptor_array);
				}
				//一个特征点的描述完成,加入这幅图像描述符 列表.
				_feature_descriptors.emplace_back(descriptor);
			}
		} while (vl_sift_process_next_octave(vlSift) != VL_ERR_EOF);
	}
	vl_sift_delete(vlSift);
}
