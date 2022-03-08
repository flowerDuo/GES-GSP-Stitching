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
#include <io.h>

using namespace cv;
using namespace cv::ximgproc;


struct ContoursConnectionObj {
	pair<Point, Point> SEPoint;
	double k;

};

namespace TestSP {
	int testTriangle(double x1, double y1, double x2, double y2, double x3, double y3);
	void testContours();
	void testVector();
	void testNormalWayExtract();
	void testLineProcess();

}
