#pragma once
#include "../Configure.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "../Util/Thin.h"
#include "../Util/EdgeDetection.h"
#include "../Util/Transform.h"
#include <math.h>
#include <opencv2/ximgproc.hpp>

using namespace cv;
using namespace cv::ximgproc;

const double MESH_GRID_SIZE = 40;

bool sortForPoint1(Point a, Point b);
bool equalForPoint1(Point a, Point b);
void connectSmallLine1(vector<vector<Point>> contours, vector<Vec4i> hierarchy, vector<vector<Point>>& contoursConnected);
pair<Point, Point> findStartEndPoint1(vector<Point > contour, Vec4i fitline);
bool isClose1(pair<Point, Point> pair1, pair<Point, Point> pair2);
double PointDist1(Point p1, Point p2);
void breakCurveInflectionPoint(vector<vector<Point>>& curves);
vector<double> getCurvature(std::vector<cv::Point> const& vecContourPoints, int step);
double getLineWeight1(vector<Point> line);
vector<Vec4f> delectParallaxAndNear(vector<Vec4f> lines);
vector<Vec4f> findLine1(Mat& gray);

struct ContoursConnectionObj {
	pair<Point, Point> SEPoint;
	double k;

};

namespace TestSP {
	int testTriangle(double x1, double y1, double x2, double y2, double x3, double y3);
	void testContours();
	void testVector();
	void testNormalWayExtractLine();
	void testLineProcess();
}
