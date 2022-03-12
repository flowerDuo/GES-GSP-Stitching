﻿//
//  NISwGSP_Stitching.cpp
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#include "NISwGSP_Stitching.h"

NISwGSP_Stitching::NISwGSP_Stitching(const MultiImages& _multi_images) : MeshOptimization(_multi_images) {

}

/*交给MeshOptimization来做*/
void NISwGSP_Stitching::setWeightToAlignmentTerm(const double _weight) {
	MeshOptimization::setWeightToAlignmentTerm(_weight);
}

/*交给MeshOptimization来做*/
void NISwGSP_Stitching::setWeightToLocalSimilarityTerm(const double _weight) {
	MeshOptimization::setWeightToLocalSimilarityTerm(_weight);
}

/*交给MeshOptimization来做*/
void NISwGSP_Stitching::setWeightToGlobalSimilarityTerm(const double _weight_beta,
	const double _weight_gamma,
	const enum GLOBAL_ROTATION_METHODS _global_rotation_method) {
	MeshOptimization::setWeightToGlobalSimilarityTerm(_weight_beta, _weight_gamma, _global_rotation_method);
}

void NISwGSP_Stitching::setWeightToContentPreservingTerm(const double _weight) {
	MeshOptimization::setWeightToContentPreservingTerm(_weight);
}

/*正式开始*/
Mat NISwGSP_Stitching::solve(const BLENDING_METHODS& _blend_method, vector<vector<Point2> >& original_vertices) {
	const MultiImages& multi_images = getMultiImages();

	vector<Triplet<double> > triplets;
	vector<pair<int, double> > b_vector;

	//准备3term需要的数据
	reserveData(triplets, b_vector, DIMENSION_2D);

	triplets.emplace_back(0, 0, STRONG_CONSTRAINT);
	triplets.emplace_back(1, 1, STRONG_CONSTRAINT);
	b_vector.emplace_back(0, STRONG_CONSTRAINT);
	b_vector.emplace_back(1, STRONG_CONSTRAINT);

	prepareAlignmentTerm(triplets);
	prepareSimilarityTerm(triplets, b_vector);

	//求解,得到每张图像最后变形的网格点坐标.注意:坐标是大图坐标系,不是每个小图的坐标系,大图坐标系的原点为图1左上角.
	original_vertices = getImageVerticesBySolving(triplets, b_vector);

	//大图左上移动至原点.
	Size2 target_size = normalizeVertices(original_vertices);

	//得到最终的大图图像数据.
	Mat result = multi_images.textureMapping(original_vertices, target_size, _blend_method);

#ifndef DP_NO_LOG
	multi_images.writeResultWithMesh(result, original_vertices, "-[NISwGSP]" +
		GLOBAL_ROTATION_METHODS_NAME[getGlobalRotationMethod()] +
		BLENDING_METHODS_NAME[_blend_method] +
		"[Mesh]", false);
	multi_images.writeResultWithMesh(result, original_vertices, "-[NISwGSP]" +
		GLOBAL_ROTATION_METHODS_NAME[getGlobalRotationMethod()] +
		BLENDING_METHODS_NAME[_blend_method] +
		"[Border]", true);
#endif
	return result;
}

Mat NISwGSP_Stitching::solve_content(const BLENDING_METHODS& _blend_method, vector<vector<Point2> >& original_vertices) {
	const MultiImages& multi_images = getMultiImages();

	vector<Triplet<double> > triplets;
	vector<pair<int, double> > b_vector;

	//准备数据
	reserveData_content(triplets, b_vector, DIMENSION_2D);

	triplets.emplace_back(0, 0, STRONG_CONSTRAINT);
	triplets.emplace_back(1, 1, STRONG_CONSTRAINT);
	b_vector.emplace_back(0, STRONG_CONSTRAINT);
	b_vector.emplace_back(1, STRONG_CONSTRAINT);

	prepareAlignmentTerm(triplets);
	prepareSimilarityTerm(triplets, b_vector);
	//准备矩阵数据.
	prepareContentPreservingTerm(triplets, b_vector);

	original_vertices = getImageVerticesBySolving(triplets, b_vector);

	Size2 target_size = normalizeVertices(original_vertices);

	Mat result = multi_images.textureMapping(original_vertices, target_size, _blend_method);

#ifndef DP_NO_LOG
	multi_images.writeResultWithMesh(result, original_vertices, "-[DPS]" +
		GLOBAL_ROTATION_METHODS_NAME[getGlobalRotationMethod()] +
		BLENDING_METHODS_NAME[_blend_method] +
		"[Mesh]", false);
	multi_images.writeResultWithMesh(result, original_vertices, "-[DPS]" +
		GLOBAL_ROTATION_METHODS_NAME[getGlobalRotationMethod()] +
		BLENDING_METHODS_NAME[_blend_method] +
		"[Border]", true);
#endif

	return result;
}

/*写出图片*/
void NISwGSP_Stitching::writeImage(const Mat& _image, const string _blend_method_name) const {
	const MultiImages& multi_images = getMultiImages();
	const Parameter& parameter = multi_images.parameter;
	string file_name = parameter.file_name;

	/*imwrite(parameter.result_dir + file_name + "-" +
		"[NISwGSP]" +
		GLOBAL_ROTATION_METHODS_NAME[getGlobalRotationMethod()] +
		_blend_method_name +
		".png", _image)*/;

		if (RUN_TYPE == 1) {
			imwrite(parameter.result_dir + file_name + "-" +
				"带曲线约束" +
				".png", _image);
		}
		else {
			imwrite(parameter.result_dir + file_name + "-" +
				"原始GSP" +
				".png", _image);
		}

}

/// <summary>
/// 客观评价指标
/// </summary>
/// <param name="original_vertices">变形后的各图像的网格点坐标</param>
void NISwGSP_Stitching::assessment(const vector<vector<Point2>> original_vertices)
{
	double RMSE = getRMSE(original_vertices);
	double W_Residual = getWarpingResidual(original_vertices);

}

double NISwGSP_Stitching::getWarpingResidual(vector<vector<Point2>> _vertices)
{
	return 0.0;
}


double NISwGSP_Stitching::getRMSE(vector<vector<Point2> > _vertices) {
	double RMSE = getMultiImages().getRMSE(_vertices);
	return RMSE;
}
