
#include "stdafx.h"

#include "blender.h"

using namespace std;
using namespace cv;
using namespace cv::detail;

static const float WEIGHT_EPS = 1e-10f;

void MyFeatherBlender::createWeightMaps( Rect dst_roi, vector<Point> corners, vector<Mat> &masks, vector<Mat> &weight_maps )
{
	dst_weight_map_.create(dst_roi.size(), CV_32F);
	dst_weight_map_.setTo(0);

	// 为每一幅图片创建weight map
	int image_num = masks.size();
	weight_maps.resize(image_num);
	for(int i = 0; i < image_num; i++)
	{
		createWeightMap(masks[i], m_sharpness_, weight_maps[i]);
		//cout << weight_maps[i].size() << endl;
		int dx = corners[i].x - dst_roi.x;
		int dy = corners[i].y - dst_roi.y;
		for (int y = 0; y < weight_maps[i].rows; ++y)
		{
			float* weight_row = weight_maps[i].ptr<float>(y);
			float* dst_weight_row = dst_weight_map_.ptr<float>(dy + y);
			for (int x = 0; x < weight_maps[i].cols; ++x)
			{
				//weight_row[x] = pow(weight_row[x], 0.1f);
				dst_weight_row[dx + x] += weight_row[x];
			}
		}
	}
	for(int i = 0; i < image_num; i++)
	{
		int dx = corners[i].x - dst_roi.x;
		int dy = corners[i].y - dst_roi.y;
		for (int y = 0; y < weight_maps[i].rows; ++y)
		{
			float* weight_row = weight_maps[i].ptr<float>(y);
			float* dst_weight_row = dst_weight_map_.ptr<float>(dy + y);
			for (int x = 0; x < weight_maps[i].cols; ++x)
				weight_row[x] = weight_row[x] / (dst_weight_row[dx + x] + WEIGHT_EPS);
		}
	}
}

void MyFeatherBlender::prepare( Rect dst_roi, vector<Point> corners, vector<Mat> &masks )
{
	dst_.create(dst_roi.size(), CV_16SC3);
	dst_.setTo(Scalar::all(0));
	dst_mask_.create(dst_roi.size(), CV_8U);
	dst_mask_.setTo(Scalar::all(0));
	dst_roi_ = dst_roi;

	this->createWeightMaps(dst_roi, corners, masks, weight_maps_);
}

void MyFeatherBlender::feed( const Mat &img, const Mat &mask, Point tl, int img_idx )
{
	CV_Assert(img.type() == CV_16SC3);
	CV_Assert(mask.type() == CV_8U);

	int dx = tl.x - dst_roi_.x;
	int dy = tl.y - dst_roi_.y;

	for (int y = 0; y < img.rows; ++y)
	{
		const Point3_<short>* src_row = img.ptr<Point3_<short> >(y);
		Point3_<short>* dst_row = dst_.ptr<Point3_<short> >(dy + y);
		const float* weight_row = weight_maps_[img_idx].ptr<float>(y);

		for (int x = 0; x < img.cols; ++x)
		{
			dst_row[dx + x].x += static_cast<short>(src_row[x].x * weight_row[x]);
			dst_row[dx + x].y += static_cast<short>(src_row[x].y * weight_row[x]);
			dst_row[dx + x].z += static_cast<short>(src_row[x].z * weight_row[x]);
		}
	}
}

void MyFeatherBlender::blend( Mat &dst, Mat &dst_mask )
{
	dst_mask_ = dst_weight_map_ > WEIGHT_EPS;
	dst = dst_;
	dst_mask = dst_mask_;
}
