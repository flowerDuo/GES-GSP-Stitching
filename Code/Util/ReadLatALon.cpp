#include "ReadLatALon.h"

vector <pair<double, double>> ReadJPGLonALon(vector<string> ImageName) {
	vector <pair<double, double>> result;
	for (int k = 0; k < ImageName.size(); k++) {
		result.emplace_back(ReadJPGLonALon(ImageName[k]));
	}
	return result;
}


pair<double, double> ReadJPGLonALon(string ImageName) {
	double height = 0;
	double lat = 0.0, lon = 0.0;
	GDALAllRegister();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	GDALDataset* poDataset = (GDALDataset*)GDALOpen(ImageName.c_str(), GA_ReadOnly);

	char** papszMetadata = poDataset->GetMetadata(NULL);

	char** papszMetadataGps = NULL;
	if (CSLCount(papszMetadata) > 0) {
		for (int i = 0; papszMetadata[i] != NULL; i++) {
			if (EQUALN(papszMetadata[i], "EXIF_GPS", 8)) {
				papszMetadataGps = CSLAddString(papszMetadataGps, papszMetadata[i]);
				//cout << papszMetadata[i] << endl;
			}
		}
	}

	int iGpsCount = CSLCount(papszMetadataGps);
	if (iGpsCount <= 0) {
		CSLDestroy(papszMetadataGps);
	}
	if (papszMetadataGps == NULL)
		return pair<double, double>(0, 0);
	for (int i = 0; papszMetadataGps[i] != NULL; i++) {
		vector<string> res;
		SpiltString(papszMetadataGps[i], res, "=");
		if (res.size() != 2)
			continue;
		string name = res[0];
		string value = res[1];
		if (name.empty() || value.empty())
			continue;
		if (name == "EXIF_GPSAltitude") //海拔高度
			height = stringToNum<int>(value.substr(1, value.length() - 1));
		if (name == "EXIF_GPSLatitude") { //纬度
			string temp = value.substr(1, value.length() - 1);
			SpiltString(temp, res, ") (");
			if (res.size() != 3)
				continue;
			lat = stringToNum<double>(res[0]) + stringToNum <double>(res[1]) / 60.0 + stringToNum<double>(res[2]) / 3600.0;
		}
		if (name == "EXIF_GPSLongitude") { //经度
			string temp = value.substr(1, value.length() - 1);
			SpiltString(temp, res, ") (");
			if (res.size() != 3)
				continue;
			lon = stringToNum<double>(res[0]) + stringToNum <double>(res[1]) / 60.0 + stringToNum<double>(res[2]) / 3600.0;
		}
	}
	return pair<double, double>(lat, lon); //纬度，经度
}



vector <pair<double, double>> ReadTIFLonALon(vector<string> ImageName) {

	vector<pair<double, double>> result;
	for (int k = 0; k < ImageName.size(); k++) {
		result.emplace_back(ReadTIFLonALon(ImageName[k]));
	}
	return result;
}

pair<double, double> ReadTIFLonALon(string ImageName) {

	pair<double, double> result;

	GDALAllRegister();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	GDALDataset* poDataset = (GDALDataset*)GDALOpen(ImageName.c_str(), GA_ReadOnly);

	const char* projRef = poDataset->GetProjectionRef();
	double a[6];
	poDataset->GetGeoTransform(a);

	/*
	//可更改为获取某个像素点的经纬度，当前为图像左上角经纬度
	double plat = a[3] + col*a[4]+row*a[5];
	double plon = a[0] + col*a[1]+row*a[2];
	*/
	OGRSpatialReference fRef, tRef;
	char* tmp = NULL;
	/** 获得projRef的一份拷贝 **/
	/** 由于projRef是const char*,下面的一个函数不接受，所以需要转换成非const **/
	tmp = (char*)malloc(strlen(projRef) + 1);
	strcpy_s(tmp, strlen(projRef) + 1, projRef);

	fRef.importFromWkt(&tmp);
	/** 设置转换后的坐标 **/
	tRef.SetWellKnownGeogCS("WGS84");

	/** 下面进行坐标转换，到此为止都不需要proj，但是下面的内容如果不安装proj将会无法编译 **/
	OGRCoordinateTransformation* coordTrans = OGRCreateCoordinateTransformation(&fRef, &tRef);
	coordTrans->Transform(1, &a[0], &a[3]);
	//（纬度，经度）

	return pair<double, double>(a[3], a[0]);  //（纬度                                                                                                                                                                                                                                                                                                                                                   ，经度）
}

vector <pair<double, double>> ReadPhantom4TIFLonALon(vector<string> ImageName) {
	vector<pair<double, double>> result;
	for (int k = 0; k < ImageName.size(); k++) {
		result.emplace_back(ReadPhantom4TIFLonALon(ImageName[k]));
	}
	return result;
}

pair<double, double> ReadPhantom4TIFLonALon(string ImageName) {
	pair<double, double> result;

	GDALAllRegister();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	GDALDataset* poDataset = (GDALDataset*)GDALOpen(ImageName.c_str(), GA_ReadOnly);
	double height = 0;
	double lat = 0.0, lon = 0.0;

	char** papszMetadata = poDataset->GetMetadata("xml:XMP");//精灵四存储在XMP元数据中。
															 //cout << papszMetadata[0] << endl;
															 //将元数据根据回车符分点
	int count = 0;
	string name_xmp = papszMetadata[0];
	//获取以drone-dji开头的子串
	string dji_temp = name_xmp.substr(name_xmp.find("drone-dji:"), name_xmp.rfind("drone-dji:") - name_xmp.find("drone-dji:"));

	vector<string> dji_xmp;
	int temp = 0;
	for (int i = 0; i < dji_temp.length(); i++) {
		if (dji_temp[i] == '\n') {
			if (count >= 1) {
				//cout << dji_temp.substr(temp+4, i - (temp+4)) << endl;
				dji_xmp.push_back(dji_temp.substr(temp + 4, i - (temp + 4)));
			}
			else {
				//cout << dji_temp.substr(0, i) << endl;
				dji_xmp.push_back(dji_temp.substr(0, i));
			}
			temp = i;
			count++;
		}
	}
	for (int i = 0; i < count; i++) {
		vector<string> res;
		SpiltString(dji_xmp[i], res, "=");
		if (res.size() != 2)
			continue;
		string name = res[0];
		string value = res[1];
		if (name.empty() || value.empty())
			continue;
		if (name == "drone-dji:RelativeAltitude") //相对高度
			height = stringToNum<double>(value.substr(2, value.length() - 1));
		if (name == "drone-dji:GpsLatitude") { //纬度
			lat = stringToNum<double>(value.substr(1, value.length() - 1));
		}
		if (name == "drone-dji:GpsLongtitude") { //经度
			lon = stringToNum<double>(value.substr(1, value.length() - 1));
		}
	}
	return pair<double, double>(lat, lon); //纬度，经度
}

/*添加从文件中读取经纬度的函数*/
//只针对DDMM.MMMM格式的转换
void  FormatConversion(string str, vector<string>& res, string delim)
{
	while (res.size() != 0)
		res.pop_back();
	string::size_type pos1, pos2;
	pos2 = str.find(delim) - 2;
	pos1 = 0;
	string s1, s2;
	s1 = str.substr(pos1, pos2);
	s2 = str.substr(pos2);
	res.push_back(s1);
	res.push_back(s2);
}

map<string, pair<double, double>> ReadLatALonByGPS(string GPS_Path) {
	map<string, pair<double, double>> result;

	ifstream gps;
	gps.open(GPS_Path.data());
	cout << GPS_Path << endl;
	assert(gps.is_open());
	string line;
	vector<string> resTemp;
	vector<string> latituderes;//纬度
	vector<string> longituderes;//经度
	string delim = ",";
	//逐行处理
	while (getline(gps, line))
	{
		SpiltString(line, resTemp, delim);
		if (resTemp.size() >= 3) {
			//纬度计算
			FormatConversion(resTemp[1], latituderes, ".");
			int lat_D = stringToNum<int>(latituderes[0]);
			float lat_M = (stringToNum<float>(latituderes[1])) / (float)60.00000000;
			float lat = (float)lat_D + lat_M;
			//经度计算
			FormatConversion(resTemp[2], longituderes, ".");
			int Long_D = stringToNum<int>(longituderes[0]);
			double Long_M = (stringToNum<double>(longituderes[1])) / 60.00000000;
			double Long = (double)Long_D + Long_M;
			result[resTemp[0]] = pair<double, double>(lat, Long);
		}
	}
	return result;
}

map<string, pair<double, double>> ReadLatALonByTXT(string GPS_Path) {
	map<string, pair<double, double>> result;

	ifstream gps;
	gps.open(GPS_Path.data());
	assert(gps.is_open());
	string line;
	vector<string> resTemp;
	vector<string> latituderes;//纬度
	vector<string> longituderes;//经度
	string delim = ",";
	//逐行处理
	while (getline(gps, line))
	{
		SpiltString(line, resTemp, delim);
		if (resTemp.size() >= 3) {
			//纬度计算
			double lat = stringToNum<double>(resTemp[1]);
			//经度计算
			double Long = stringToNum<double>(resTemp[2]);

			result[resTemp[0]] = pair<double, double>(lat, Long);
		}
	}
	return result;
}
