﻿
//#define DP_NO_LOG
#define ASSESSMENT

#ifndef __UglyMan_Stitiching__Configure__
#define __UglyMan_Stitiching__Configure__

#include "./Debugger/ErrorController.h"
#include "./Debugger/TimeCalculator.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>
#include <dirent.h>
#include <algorithm>
#include <direct.h>
using namespace std;

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
#include "opencv2/ximgproc.hpp"


#include <Eigen/SVD>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra> 
#include "opencv2/dnn.hpp"
#include <opencv2/dnn/layer.details.hpp>
using namespace Eigen;

#include "vl/sift.h"
using namespace cv::dnn;


/******************************/
/******* you may adjust *******/
/******************************/

const int RUN_TYPE = 1;//0:GSP 1:GES-GSP
const string TXT_NAME = "-STITCH-GRAPH.txt";


/*** data setting ***/
const int GRID_SIZE = 40;
const int DOWN_SAMPLE_IMAGE_SIZE = 800 * 600;
//Contour length/image shortest edge ratio
const double CONTENT_LENGTH_THRESHOLD = 0.15;

//HED threshold
const double HED_THRESHOLD = 0.5;
const int threshold_value = 120;

/*** APAP ***/
const double APAP_GAMMA = 0.0015;
const double APAP_SIGMA = 8.5;

/*** matching method ***/
const string FEATURE_RATIO_TEST_THRESHOLD_STRING = "15e-1";//15*10^-1=1.5
const double FEATURE_RATIO_TEST_THRESHOLD = atof(FEATURE_RATIO_TEST_THRESHOLD_STRING.c_str());

/*** homography based ***/
const double GLOBAL_HOMOGRAPHY_MAX_INLIERS_DIST = 5.;
const double  LOCAL_HOMOGRAPHY_MAX_INLIERS_DIST = 3.;
const    int  LOCAL_HOMOGRAPHY_MIN_FEATURES_COUNT = 40;

/*** vlfeat sift ***/
const    int SIFT_LEVEL_COUNT = 3;
const    int SIFT_MINIMUM_OCTAVE_INDEX = 0;
const double SIFT_PEAK_THRESH = 0.;
const double SIFT_EDGE_THRESH = 10.;

/*** init feature ***/
const double INLIER_TOLERANT_STD_DISTANCE = 4.25; /* mean + 4.25 * std */

/*** sRANSAC ***/
const double GLOBAL_TRUE_PROBABILITY = 0.225;
const double LOCAL_TRUE_PROBABILITY = 0.2;
const double OPENCV_DEFAULT_CONFIDENCE = 0.995;

/*** sparse linear system ***/
const double STRONG_CONSTRAINT = 1e4;

/*** bundle adjustment ***/
const int CRITERIA_MAX_COUNT = 1000;
const double CRITERIA_EPSILON = DBL_EPSILON;

/*** 2D Method ***/
const double TOLERANT_ANGLE = 1.5;

/*** 3D Method ***/
const double LAMBDA_GAMMA = 10;

/******************************/
/******************************/
/******************************/

/*** rotation method setting ***/
enum GLOBAL_ROTATION_METHODS {
	GLOBAL_ROTATION_2D_METHOD = 0, GLOBAL_ROTATION_3D_METHOD, GLOBAL_ROTATION_METHODS_SIZE
};
const string GLOBAL_ROTATION_METHODS_NAME[GLOBAL_ROTATION_METHODS_SIZE] = {
	"[2D]", "[3D]"
};

/* blending method setting */
enum BLENDING_METHODS {
	BLEND_AVERAGE = 0, BLEND_LINEAR, BLEND_METHODS_SIZE
};
const string BLENDING_METHODS_NAME[BLEND_METHODS_SIZE] = {
	"[BLEND_AVERAGE]", "[BLEND_LINEAR]"
};


/* type */
typedef float FLOAT_TYPE;
typedef Size_<FLOAT_TYPE> Size2;
typedef Point_<FLOAT_TYPE> Point2;
typedef Rect_<FLOAT_TYPE> Rect2;

const int DIMENSION_2D = 2;
const int HOMOGRAPHY_VARIABLES_COUNT = 9;

/* AutoStitch */
enum  AUTO_STITCH_WAVE_CORRECTS { WAVE_X = 0, WAVE_H, WAVE_V };
const AUTO_STITCH_WAVE_CORRECTS   WAVE_CORRECT = WAVE_H;
const string AUTO_STITCH_WAVE_CORRECTS_NAME[] = { "", "[WAVE_H]", "[WAVE_V]" };

#endif /* defined(__UglyMan_Stitiching__Configure__) */
