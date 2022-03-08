#pragma once
#include "../Configure.h"
#include <imgproc/types_c.h>
#include "../Feature/ImageData.h"


void thin(Mat srcImage, Mat& dst, double kernalSizeTimes);
void thinTest(Mat srcImage, Mat& dst, double kernalSizeTimes);
