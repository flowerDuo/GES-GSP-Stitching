#pragma once

#include <gdal.h>
#include <gdalexif.h>
#include <gdal_priv.h>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <iomanip>
#include <fstream>
#define ACCEPT_USE_OF_DEPRECATED_PROJ_API_H 1
#include "ogrsf_frmts.h"
#include "ogr_srs_api.h"
#include "ogr_spatialref.h"
#include "ogr_api.h"
#include "proj_api.h"
#include "../Configure.h"
#include "./Transform.h"
#define BYTE short 

using namespace std;

/*ReadJPGLonALon(),该函数目前只能读取.jpg格式的图像，jpg和tiff的头文件存储方式不同
	返回信息： （高度，（纬度，经度））
	使用的库： GDAL
	参数：图像路径数组
*/
vector <pair<double, double>> ReadJPGLonALon(vector<string> ImageName);

pair<double, double> ReadJPGLonALon(string ImageName); //单个图像

/*ReadTIFLonALon() 函数用于读取.tif格式图像的经纬度
	返回信息：（纬度，经度）
	使用的库： GDAL + Proj4
	参数：图像路径数组
*/
vector <pair<double, double>> ReadTIFLonALon(vector<string> ImageName);

pair<double, double> ReadTIFLonALon(string ImageName); //单个图像

/* 大疆精灵4 多光谱版 需要从xmp元数据中读取*/
vector <pair<double, double>> ReadPhantom4TIFLonALon(vector<string> ImageName);

pair<double, double> ReadPhantom4TIFLonALon(string ImageName); //单个图像
/*
	从文件中读取经纬度
	文件格式：仅支持.GPS格式文件
		图像名称,纬度（DDMM.MMMM）,经度,不知名,高度
		TTC28078,4100.0830,10706.4553,14803.01,1038.1
	返回：map<string,pair<double,double>>   <图像名称，<纬度，经度>>
*/

map<string, pair<double, double>> ReadLatALonByGPS(string GPS_Path);

map<string, pair<double, double>> ReadLatALonByTXT(string GPS_Path);
