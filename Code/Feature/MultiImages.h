//
//  MultiImages.h
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#ifndef __UglyMan_Stitiching__MultiImages__
#define __UglyMan_Stitiching__MultiImages__

#include <queue>
#include "../Configure.h"
#include "../Stitching/Parameter.h"
#include "ImageData.h"
#include "../Util/Statistics.h"
#include "../Debugger/ImageDebugger.h"
#include "../Stitching/APAP_Stitching.h"
#include "../Util/Blending.h"
#include "../Debugger/ColorMap.h"

#include <opencv2/calib3d.hpp> /* CV_RANSAC */
#include <opencv2/stitching/detail/autocalib.hpp> /* ImageFeatures, MatchesInfo */
#include <opencv2/stitching/detail/camera.hpp> /* CameraParams */
#include <opencv2/stitching/detail/motion_estimators.hpp> /* BundleAdjusterBase */

const int PAIR_COUNT = 2;

class FeatureDistance {
public:
	double distance;
	int feature_index[PAIR_COUNT];
	FeatureDistance() {
		feature_index[0] = feature_index[1] = -1;
		distance = FLT_MAX;
	}
	FeatureDistance(const double _distance,
		const int _p_1,
		const int _feature_index_1,
		const int _feature_index_2) {
		distance = _distance;
		feature_index[_p_1] = _feature_index_1;
		feature_index[!_p_1] = _feature_index_2;
	}
	bool operator < (const FeatureDistance& fd) const {
		return distance > fd.distance;
	}
private:
};

class SimilarityElements {
public:
	double scale;
	double theta;
	SimilarityElements(const double _scale,
		const double _theta) {
		scale = _scale;
		theta = _theta;
	}
private:
};

class MultiImages {
public:
	MultiImages(const string& _file_name,
		LINES_FILTER_FUNC* _width_filter = &LINES_FILTER_NONE,
		LINES_FILTER_FUNC* _length_filter = &LINES_FILTER_NONE);

	const vector<detail::ImageFeatures>& getImagesFeaturesByMatchingPoints() const;
	const vector<detail::MatchesInfo>& getPairwiseMatchesByMatchingPoints() const;
	const vector<detail::CameraParams>& getCameraParams() const;

	const vector<vector<bool> >& getImagesFeaturesMaskByMatchingPoints() const;

	const vector<vector<vector<pair<int, int> > > >& getFeaturePairs() const;
	const vector<vector<vector<Point2> > >& getFeatureMatches() const;

	const vector<vector<vector<bool> > >& getAPAPOverlapMask() const;
	const vector<vector<vector<Mat> > >& getAPAPHomographies() const;
	const vector<vector<vector<Point2> > >& getAPAPMatchingPoints() const;

	const vector<vector<InterpolateVertex> >& getInterpolateVerticesOfMatchingPoints() const;

	const vector<int>& getImagesVerticesStartIndex() const;
	const vector<SimilarityElements>& getImagesSimilarityElements(const enum GLOBAL_ROTATION_METHODS& _global_rotation_method) const;
	const vector<vector<pair<double, double> > >& getImagesRelativeRotationRange() const;

	const vector<vector<double> >& getImagesGridSpaceMatchingPointsWeight(const double _global_weight_gamma) const;

	const  vector<vector<vector<double> >>& getSamplesWeight() const;

	const vector<Point2>& getImagesLinesProject(const int _from, const int _to) const;

	const vector<Mat>& getImages() const;

	//Wasted code.
	const double getRansacDiffWeight(const pair<int, int>& _index_pair) const;
	const float getOverlap(pair<int, int> _mask_pair) const;


	FLOAT_TYPE getImagesMinimumLineDistortionRotation(const int _from, const int _to) const;
	const vector<vector<vector<Point> > >& getContentSamplePoints() const;
	const vector<vector<vector<InterpolateVertex> > >& getSamplesInterpolation() const;
	const vector < vector<vector<pair<double, double> >> >& getTermUV() const;

	Mat textureMapping(const vector<vector<Point2> >& _vertices,
		const Size2& _target_size,
		const BLENDING_METHODS& _blend_method) const;

	Mat textureMapping(const vector<vector<Point2> >& _vertices,
		const Size2& _target_size,
		const BLENDING_METHODS& _blend_method,
		vector<Mat>& _warp_images) const;

	void writeResultWithMesh(const Mat& _result,
		const vector<vector<Point2> >& _vertices,
		const string& _postfix,
		const bool _only_border) const;

	void drawRansac(const int img_index, const int img_index_second,
		const vector<pair<int, int> >& _initial_indices, const vector<char>& _mask) const;

	vector<ImageData> images_data;
	Parameter parameter;
	mutable vector<vector<double > >            content_line_weights;

	double getRMSE(vector<vector<Point2> > _vertices) const;
	pair<double, double> getWarpingResidual(vector<vector<Point2> > _vertices) const;
private:
	/*** Debugger ***/
	void writeImageOfFeaturePairs(const string& _name,
		const pair<int, int>& _index_pair,
		const vector<pair<int, int> >& _pairs) const;
	/****************/

	void doFeatureMatching() const;
	void initialFeaturePairsSpace() const;
	void initialRansacDiffPairs() const;
	//Wasted code 
	void updataRansacDiff(const pair<int, int>& _index_pair, const vector<Point2> srcPoints, const vector<Point2> dstPoints, const vector<char> final_mask, const Mat H) const;
	//Wasted code 
	double generateRansacAvgDiff(const pair<int, int>& _index_pair) const;
	//Wasted code 
	void generateRansacDiffWeight(vector<pair<int, int> > pair) const;

	vector<pair<int, int> > getInitialFeaturePairs(const pair<int, int>& _match_pair) const;

	vector<pair<int, int> > getFeaturePairsBySequentialRANSAC(const pair<int, int>& _match_pair,
		const vector<Point2>& _X,
		const vector<Point2>& _Y,
		const vector<pair<int, int> >& _initial_indices) const;

	const  vector<vector<pair<double, double>>> calcTriangleUV(const vector<vector<Point>> samples) const;

	const vector<vector<vector<Point2> > >& getTwoImgFeatureMatches(pair<int, int> _mask_pair_) const;

	vector<pair<int, int>> getTwoImgFeaturePairs(pair<int, int> _mask_pair_) const;


	mutable vector<vector<vector<double>>> ransacDiff;
	mutable vector<vector<double>> ransacAvgDiff;
	mutable vector<vector<double>> ransacDiffWeight;
	const string txtName = "E://RansacDst//";

	mutable vector<detail::ImageFeatures> images_features;
	mutable vector<detail::MatchesInfo>   pairwise_matches;
	mutable vector<detail::CameraParams>  camera_params;
	mutable vector<vector<bool> > images_features_mask;

	mutable vector<vector<vector<pair<int, int> > > > feature_pairs;
	mutable vector<vector<vector<Point2> > > feature_matches; /* [m1][m2][j], img1 j_th matches */

	mutable vector<vector<vector<bool> > >   apap_overlap_mask;
	mutable vector<vector<vector<Mat> > >    apap_homographies;
	mutable vector<vector<vector<Point2> > > apap_matching_points;

	mutable vector<vector<InterpolateVertex> > mesh_interpolate_vertex_of_feature_pts;
	mutable vector<vector<InterpolateVertex> > mesh_interpolate_vertex_of_matching_pts;

	mutable vector<int> images_vertices_start_index;
	mutable vector<SimilarityElements> images_similarity_elements_2D;
	mutable vector<SimilarityElements> images_similarity_elements_3D;
	mutable vector<vector<pair<double, double> > > images_relative_rotation_range;

	mutable vector<vector<double> > images_polygon_space_matching_pts_weight;
	mutable vector<vector<vector<double> > >  samplesWeight;
	/* Line */
	mutable vector<vector<FLOAT_TYPE> > images_minimum_line_distortion_rotation;
	mutable vector<vector<vector<Point2> > > images_lines_projects; /* [m1][m2] img1 lines project on img2 */

	mutable vector<Mat> images;

	mutable vector<vector<vector<Point> > >            content_sample_points;

	mutable vector<vector<vector<InterpolateVertex> > > content_mesh_interpolation;
	mutable vector < vector<vector<pair<double, double>> >> content_term_uv;


};
double Point2dDis(Point2d p1, Point2d p2);
int VerifyVertices(Point2d p1, int m1Tom2_p1_index, const vector<Indices>& m1_polygons_indice, const vector<Point2>& m1_vertices);
Mat _getAffineTransform(vector<vector<Point2>>& _vertices, int m1, const vector<Indices>& m1_polygons_indice, int m1Tom2_p1_index, int m1Tom2_p1_temp, const vector<Point2>& m1_vertices);
pair<double, double> getLineResidual(vector<Point2f> _vertices, Vec4f _line_param);
#endif /* defined(__UglyMan_Stitiching__MultiImages__) */
