//
//  ImageData.h
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#ifndef __UglyMan_Stitiching__ImageData__
#define __UglyMan_Stitiching__ImageData__

#include <memory>
#include "../Util/Statistics.h"
#include "FeatureController.h"
#include "../Mesh/MeshGrid.h"
//#include "../Util/ReadLatALon.h"
#include <opencv2\imgproc\types_c.h>
#include "../Util/EdgeDetection.h"
#include "../Util/Thin.h"
#include <opencv2/ximgproc.hpp>
using namespace cv::ximgproc;

class LineData {
public:
	LineData(const Point2& _a,
		const Point2& _b,
		const double _width,
		const double _length);
	Point2 data[2];
	double width, length;
private:
};

typedef const bool (LINES_FILTER_FUNC)(const double _data, \
	const Statistics& _statistics);

LINES_FILTER_FUNC LINES_FILTER_NONE;
LINES_FILTER_FUNC LINES_FILTER_WIDTH;
LINES_FILTER_FUNC LINES_FILTER_LENGTH;


class ImageData {
public:
	string file_name, file_extension;
	pair<double, double> LatALon; //图像经纬度 （纬度，经度）
	const string* file_dir, * debug_dir;
	ImageData(const string& _file_dir,
		const string& _file_full_name,
		LINES_FILTER_FUNC* _width_filter,
		LINES_FILTER_FUNC* _length_filter,
		const string* _debug_dir = NULL);

	const Mat& getGreyImage() const;
	const vector<LineData>& getLines() const;
	const vector<Point2>& getFeaturePoints() const;
	const vector<FeatureDescriptor>& getFeatureDescriptors() const;
	const vector<vector<Point>> getContentSamplesPoint(vector<double>& weights) const;

	//vector<vector<Point>>* output
	void clear();

	Mat img, rgba_img, alpha_mask;
	unique_ptr<Mesh2D> mesh_2d;

private:
	LINES_FILTER_FUNC* width_filter, * length_filter;

	mutable Mat grey_img;
	mutable vector<LineData> img_lines;
	mutable vector<Point2> feature_points;
	mutable vector<FeatureDescriptor> feature_descriptors;
	vector<Vec4f> findLine(Mat& gray) const;
};

bool sortForPoint(Point a, Point b);

bool equalForPoint(Point a, Point b);
void connectSmallLine(vector<vector<Point>> contours, vector<Vec4i> hierarchy, vector<vector<Point>>& contoursConnected);
vector<vector<Point>> connectCollineationLine(vector<vector<Point>>& contours, vector <double >& lengths_out, vector<vector<Point>>& static_sample, int image_width, int image_height);
pair<Point, Point> findStartEndPoint(vector<Point > contour, Vec4i fitline);
pair<Point, Point> findStartEndPoint(vector<Point > contour);

pair<Point, Point> findLineMinAndMax(vector<Point > contour);
bool isClose(pair<Point, Point> pair1, pair<Point, Point> pair2);
bool isExtend(pair<Point, Point> mmpair1, pair<Point, Point> mmpair2, pair<Point, Point> sepair1, pair<Point, Point> sepair2);
bool isParallax(pair<Point, Point> mmpair1, pair<Point, Point> mmpair2);
double PointDist(Point p1, Point p2);
double getLineWeight(vector<Point> line);

void transLines2Contours(vector<vector<Point>>& contours, vector<Vec4f> lines);
pair<double, double> transRectangular2Polar(Vec4f, int image_width, int image_height);
#endif /* defined(__UglyMan_Stitiching__ImageData__) */
