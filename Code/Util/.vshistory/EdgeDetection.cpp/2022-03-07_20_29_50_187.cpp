#pragma once
#include "EdgeDetection.h"
/// <summary>
/// HED detector
/// </summary>
/// <param name="src"></param>
/// <param name="dst"></param>
/// <param name="threshold"></param>
void edgeDetection(cv::Mat& src, cv::Mat& dst, double threshold)
{
	Mat img = src.clone();
	Size reso(img.rows, img.cols);
	Mat blob = cv::dnn::blobFromImage(img, threshold, reso, false, false);
	string modelCfg = R"(F:\Projects\C++\NISwGSP_Stitching\model\deploy.prototxt)";
	string modelBin = R"(F:\Projects\C++\NISwGSP_Stitching\model\hed_pretrained_bsds.caffemodel)";
	Net net = cv::dnn::readNet(modelCfg, modelBin);
	if (net.empty()) {
		std::cout << "net empty" << std::endl;
	}
	net.setInput(blob);
	Mat out = net.forward();
	resize(out.reshape(1, reso.height), out, img.size());

	out.convertTo(dst, CV_8UC1, 255);
}
