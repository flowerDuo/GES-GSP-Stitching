

#include "NISwGSP_Stitching.h"

NISwGSP_Stitching::NISwGSP_Stitching(const MultiImages& _multi_images) : MeshOptimization(_multi_images) {

}


void NISwGSP_Stitching::setWeightToAlignmentTerm(const double _weight) {
	MeshOptimization::setWeightToAlignmentTerm(_weight);
}


void NISwGSP_Stitching::setWeightToLocalSimilarityTerm(const double _weight) {
	MeshOptimization::setWeightToLocalSimilarityTerm(_weight);
}


void NISwGSP_Stitching::setWeightToGlobalSimilarityTerm(const double _weight_beta,
	const double _weight_gamma,
	const enum GLOBAL_ROTATION_METHODS _global_rotation_method) {
	MeshOptimization::setWeightToGlobalSimilarityTerm(_weight_beta, _weight_gamma, _global_rotation_method);
}

void NISwGSP_Stitching::setWeightToContentPreservingTerm(const double _weight) {
	MeshOptimization::setWeightToContentPreservingTerm(_weight);
}


Mat NISwGSP_Stitching::solve(const BLENDING_METHODS& _blend_method, vector<vector<Point2> >& original_vertices) {
	const MultiImages& multi_images = getMultiImages();

	vector<Triplet<double> > triplets;
	vector<pair<int, double> > b_vector;

	reserveData(triplets, b_vector, DIMENSION_2D);

	triplets.emplace_back(0, 0, STRONG_CONSTRAINT);
	triplets.emplace_back(1, 1, STRONG_CONSTRAINT);
	b_vector.emplace_back(0, STRONG_CONSTRAINT);
	b_vector.emplace_back(1, STRONG_CONSTRAINT);

	prepareAlignmentTerm(triplets);
	prepareSimilarityTerm(triplets, b_vector);


	original_vertices = getImageVerticesBySolving(triplets, b_vector);


	Size2 target_size = normalizeVertices(original_vertices);


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


	reserveData_content(triplets, b_vector, DIMENSION_2D);

	triplets.emplace_back(0, 0, STRONG_CONSTRAINT);
	triplets.emplace_back(1, 1, STRONG_CONSTRAINT);
	b_vector.emplace_back(0, STRONG_CONSTRAINT);
	b_vector.emplace_back(1, STRONG_CONSTRAINT);

	prepareAlignmentTerm(triplets);
	prepareSimilarityTerm(triplets, b_vector);

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


void NISwGSP_Stitching::writeImage(const Mat& _image, const string _blend_method_name) const {
	const MultiImages& multi_images = getMultiImages();
	const Parameter& parameter = multi_images.parameter;
	string file_name = parameter.file_name;

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
/// Assessment
/// </summary>
void NISwGSP_Stitching::assessment(const vector<vector<Point2>> original_vertices)
{
	double RMSE = getRMSE(original_vertices);
	//MDR
	pair<double, double> W_Residual = getWarpingResidual(original_vertices);

}

pair<double, double> NISwGSP_Stitching::getWarpingResidual(vector<vector<Point2>> _vertices)
{
	return getMultiImages().getWarpingResidual(_vertices);
}


double NISwGSP_Stitching::getRMSE(vector<vector<Point2> > _vertices) {
	return getMultiImages().getRMSE(_vertices);
}
