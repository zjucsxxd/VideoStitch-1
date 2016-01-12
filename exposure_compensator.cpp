
#include "stdafx.h"

#include "exposure_compensator.h"




void MyExposureCompensator::createWeightMaps( const vector<Point> &corners, const vector<Mat> &images, 
	const vector<Mat> &masks, vector<Mat_<float>> &ec_maps )
{
	vector<pair<Mat,uchar> > level_masks;
	for (size_t i = 0; i < masks.size(); ++i)
		level_masks.push_back(make_pair(masks[i], 255));
	createWeightMaps(corners, images, level_masks, ec_maps);
}

void MyExposureCompensator::createWeightMaps( const vector<Point> &corners, const vector<Mat> &images, 
	const vector<pair<Mat,uchar>> &masks, vector<Mat_<float>> &ec_maps )
{
	CV_Assert(corners.size() == images.size() && images.size() == masks.size());

	const int num_images = static_cast<int>(images.size());

	vector<Size> bl_per_imgs(num_images);
	vector<Point> block_corners;
	vector<Mat> block_images;
	vector<pair<Mat,uchar> > block_masks;

	// Construct blocks for gain compensator
	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		Size bl_per_img((images[img_idx].cols + bl_width_ - 1) / bl_width_,
			(images[img_idx].rows + bl_height_ - 1) / bl_height_);
		int bl_width = (images[img_idx].cols + bl_per_img.width - 1) / bl_per_img.width;
		int bl_height = (images[img_idx].rows + bl_per_img.height - 1) / bl_per_img.height;
		bl_per_imgs[img_idx] = bl_per_img;
		for (int by = 0; by < bl_per_img.height; ++by)
		{
			for (int bx = 0; bx < bl_per_img.width; ++bx)
			{
				Point bl_tl(bx * bl_width, by * bl_height);
				Point bl_br(min(bl_tl.x + bl_width, images[img_idx].cols),
					min(bl_tl.y + bl_height, images[img_idx].rows));

				block_corners.push_back(corners[img_idx] + bl_tl);
				block_images.push_back(images[img_idx](Rect(bl_tl, bl_br)));
				block_masks.push_back(make_pair(masks[img_idx].first(Rect(bl_tl, bl_br)), masks[img_idx].second));
			}
		}
	}

	GainCompensator compensator;
	compensator.feed(block_corners, block_images, block_masks);
	vector<double> gains = compensator.gains();
	ec_maps.resize(num_images);

	Mat_<float> ker(1, 3);
	ker(0,0) = 0.25; ker(0,1) = 0.5; ker(0,2) = 0.25;

	int bl_idx = 0;
	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		Size bl_per_img = bl_per_imgs[img_idx];
		ec_maps[img_idx].create(bl_per_img);

		for (int by = 0; by < bl_per_img.height; ++by)
			for (int bx = 0; bx < bl_per_img.width; ++bx, ++bl_idx)
				ec_maps[img_idx](by, bx) = static_cast<float>(gains[bl_idx]);

		sepFilter2D(ec_maps[img_idx], ec_maps[img_idx], CV_32F, ker, ker);
		sepFilter2D(ec_maps[img_idx], ec_maps[img_idx], CV_32F, ker, ker);
	}

	double max_ec = 1.0f;
	double max_ec_i, min_ec_i;
	for (int i = 0; i < num_images; i++)
	{
		cv::minMaxIdx(ec_maps[i], &min_ec_i, &max_ec_i);
		max_ec = std::max(max_ec, max_ec_i);
	}
	for (int i = 0; i < num_images; i++)
		ec_maps[i] = ec_maps[i] / ((float)(max_ec));
	ec_maps_ = ec_maps;
}

void MyExposureCompensator::feed( const vector<Point> &corners, const vector<Mat> &images, vector<Mat> &masks )
{
	vector<pair<Mat,uchar> > level_masks;
	for (size_t i = 0; i < masks.size(); ++i)
		level_masks.push_back(make_pair(masks[i], 255));
	createWeightMaps(corners, images, level_masks, ec_maps_);
}

void MyExposureCompensator::gainMapResize( vector<Size> sizes_, vector<Mat_<float>> &ec_maps )
{
	int n = sizes_.size();
	for(int i = 0; i < n; i++)
	{
		Mat_<float> gain_map;
		resize(ec_maps[i], gain_map, sizes_[i], 0, 0, INTER_LINEAR);
		ec_maps[i] = gain_map.clone();
	}
}

void MyExposureCompensator::apply( int index, Mat &image )
{
	CV_Assert(image.type() == CV_8UC3);

	Mat_<float> gain_map;
	if (ec_maps_[index].size() == image.size())
		gain_map = ec_maps_[index];
	else
		resize(ec_maps_[index], gain_map, image.size(), 0, 0, INTER_LINEAR);

	for (int y = 0; y < image.rows; ++y)
	{
		const float* gain_row = gain_map.ptr<float>(y);
		Point3_<uchar>* row = image.ptr<Point3_<uchar> >(y);
		for (int x = 0; x < image.cols; ++x)
		{
			row[x].x = saturate_cast<uchar>(row[x].x * gain_row[x]);
			row[x].y = saturate_cast<uchar>(row[x].y * gain_row[x]);
			row[x].z = saturate_cast<uchar>(row[x].z * gain_row[x]);
		}
	}
}
