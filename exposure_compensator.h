#ifndef __VIDEO_STITCH_MY_EXPOSURE_COMPENSATOR_H__
#define __VIDEO_STITCH_MY_EXPOSURE_COMPENSATOR_H__

#include "stdafx.h"

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;


/*
 * 使用的是BlocksGainCompensator
 */
class MyExposureCompensator
{
public:
	MyExposureCompensator(int bl_width = 32, int bl_height = 32)
		: bl_width_(bl_width), bl_height_(bl_height) {}

	void createWeightMaps(const vector<Point> &corners, const vector<Mat> &images,
		const vector<Mat> &masks, vector<Mat_<float>> &ec_maps);

	void createWeightMaps(const vector<Point> &corners, const vector<Mat> &images,
		const vector<pair<Mat,uchar>> &masks, vector<Mat_<float>> &ec_maps);

	void feed(const vector<Point> &corners, const vector<Mat> &images, vector<Mat> &masks);

	void gainMapResize(vector<Size> sizes_, vector<Mat_<float>> &ec_maps);

	void apply(int index, Mat &image);

private:
	int bl_width_, bl_height_;
	vector<Mat_<float> > ec_maps_;
};



#endif