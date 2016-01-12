
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>

#include "gutil.h"

typedef struct _point
{
	int x;
	int y;
}
Point;

typedef struct _image_size
{
	int height;
	int width;
}
ImageSize;

typedef struct _image_xy_map
{
	float *xmap;
	float *ymap;
}
ImageXYMap;

typedef struct _image_weight
{
	float *blend_weight;
	float *ec_weight;
	float *total_weight;
}
ImageWeight;

typedef struct _const_data
{
	int height;
	int width;

	int warped_height;
	int warped_width;

	int corner_x;
	int corner_y;
}
ConstDataGPU;

cudaError_t gCudaStatus;

#define CUDA_CHECK_CALL(fun, err_msg, return_code)					\
	gCudaStatus = fun;												\
	if(gCudaStatus != cudaSuccess){									\
		fprintf(stderr, "error_code%d: %s", gCudaStatus, err_msg);	\
		return return_code;											\
	}

ConstDataGPU *const_data;
__constant__ ConstDataGPU dev_const_data[100];
ImageSize pano_size_;
ImageXYMap *dev_maps_;
ImageWeight *dev_weights_;
GPUImageData *dev_imgs_;
static int image_num_;
unsigned char *dev_pano_;

#define USE_STREAM 1
#define DST_IMAGE_CHANNEL 3

int testGPU()
{
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		return cudaStatus;
	else
		return 0;
}

int initGPU(int n)
{
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return -1;
	}
	image_num_ = n;
	const_data = (ConstDataGPU *)(malloc(n * sizeof(ConstDataGPU)));
	dev_maps_ = (ImageXYMap *)(malloc(n * sizeof(ImageXYMap)));
	dev_weights_ = (ImageWeight *)(malloc(n * sizeof(ImageWeight)));
	dev_imgs_ = (GPUImageData *)(malloc(n * sizeof(GPUImageData)));//dev_imgs_[0].data = 0;
	return 0;
}


int initdataCopy2GPU(C2GInitData *c2g_data, int pano_height, int pano_width)
{
	for(int i = 0; i < image_num_; i++)
	{
		const_data[i].warped_height = c2g_data[i].warped_height;
		const_data[i].warped_width = c2g_data[i].warped_width;
		const_data[i].height = c2g_data[i].height;
		const_data[i].width = c2g_data[i].width;
		const_data[i].corner_x = c2g_data[i].corner_x;
		const_data[i].corner_y = c2g_data[i].corner_y;

		int xy_map_size = c2g_data[i].warped_height * c2g_data[i].warped_width * sizeof(float);
		int img_size = c2g_data[i].height * c2g_data[i].width * 3 * sizeof(unsigned char);

		//	给xmap和ymap在显存上分配空间
		CUDA_CHECK_CALL(cudaMalloc((void**)&(dev_maps_[i].xmap), xy_map_size), "cudaMalloc failed!\n", -2);
		CUDA_CHECK_CALL(cudaMalloc((void**)&(dev_maps_[i].ymap), xy_map_size), "cudaMalloc failed!\n", -2);

		//	给权重矩阵在显存上分配空间
		CUDA_CHECK_CALL(cudaMalloc((void**)&(dev_weights_[i].ec_weight), xy_map_size), "cudaMalloc failed!\n", -2);
		CUDA_CHECK_CALL(cudaMalloc((void**)&(dev_weights_[i].blend_weight), xy_map_size), "cudaMalloc failed!\n", -2);
		CUDA_CHECK_CALL(cudaMalloc((void**)&(dev_weights_[i].total_weight), xy_map_size), "cudaMalloc failed!\n", -2);

		//	给每一帧图像分配显存
		CUDA_CHECK_CALL(cudaMalloc((void**)&(dev_imgs_[i].data), img_size), "cudaMalloc failed!\n", -2);

		//	复制数据
		CUDA_CHECK_CALL(cudaMemcpy(dev_maps_[i].xmap, c2g_data[i].xmap, xy_map_size, cudaMemcpyHostToDevice), 
			"cudaMemcpy xmap failed!\n", -2);
		CUDA_CHECK_CALL(cudaMemcpy(dev_maps_[i].ymap, c2g_data[i].ymap, xy_map_size, cudaMemcpyHostToDevice), 
			"cudaMemcpy ymap failed!\n", -2);
		CUDA_CHECK_CALL(cudaMemcpy(dev_weights_[i].ec_weight, c2g_data[i].ec_weight, xy_map_size, cudaMemcpyHostToDevice), 
			"cudaMemcpy ec_weight failed!\n", -2);
		CUDA_CHECK_CALL(cudaMemcpy(dev_weights_[i].blend_weight, c2g_data[i].blend_weight, xy_map_size, cudaMemcpyHostToDevice), 
			"cudaMemcpy blend_weight failed!\n", -2);
		CUDA_CHECK_CALL(cudaMemcpy(dev_weights_[i].total_weight, c2g_data[i].total_weight, xy_map_size, cudaMemcpyHostToDevice), 
			"cudaMemcpy blend_weight failed!\n", -2);
	}
	//	常数存储器
	CUDA_CHECK_CALL(cudaMemcpyToSymbol(dev_const_data, const_data, image_num_ * sizeof(ConstDataGPU)), 
		"cudaMemcpyToSymbol failed\n", -2);

	pano_size_.height = pano_height;
	pano_size_.width = pano_width;
	int pano_malloc_size = pano_height * pano_width * DST_IMAGE_CHANNEL * sizeof(unsigned char);
	//	给全景图结果在显存上分配空间
	CUDA_CHECK_CALL(cudaMalloc((void**)&(dev_pano_), pano_malloc_size), "cudaMalloc failed!\n", -2);
	return 0;
}

__global__ void compose(unsigned char *image, ImageXYMap xymap, ImageWeight weight, unsigned char *dst, int img_idx, ImageSize pano_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if((i < dev_const_data[img_idx].warped_width) && (j < dev_const_data[img_idx].warped_height))
	{
		int data_idx = j * dev_const_data[img_idx].warped_width + i;
		float map_x = xymap.xmap[data_idx];
		int map_x1 = (int)map_x;
		if(map_x1 >= 0)
		{
			float map_y = xymap.ymap[data_idx];
			int map_y1 = (int)map_y;
			int map_x2 = map_x1 + 1;
			int map_y2 = map_y1 + 1;

			int dst_data_idx = ((j + dev_const_data[img_idx].corner_y) * pano_size.width + i + dev_const_data[img_idx].corner_x) * DST_IMAGE_CHANNEL;
			
			float dx1 = map_x - map_x1;
			float dy1 = map_y - map_y1;
			float dx2 = map_x2 - map_x;
			float dy2 = map_y2 - map_y;
			int img_data_idx11 = (map_y1 * dev_const_data[img_idx].width + map_x1) * 3;
			int img_data_idx12 = (map_y2 * dev_const_data[img_idx].width + map_x1) * 3;
			int img_data_idx21 = (map_y1 * dev_const_data[img_idx].width + map_x2) * 3;
			int img_data_idx22 = (map_y2 * dev_const_data[img_idx].width + map_x2) * 3;
			float total_weight = weight.total_weight[data_idx];
			
			for(int channel = 0; channel < 3; channel++)
			{
				dst[dst_data_idx + channel] += (unsigned char)((
					image[img_data_idx11 + channel] * dx2 * dy2 + 
					image[img_data_idx12 + channel] * dx2 * dy1 + 
					image[img_data_idx21 + channel] * dx1 * dy2 + 
					image[img_data_idx22 + channel] * dx1 * dy1
					) * total_weight);
			}
		}
	}
}

#define STREAM_NUM 2

int composeGPU(GPUImageData *images, unsigned char *dst)
{
	int pano_malloc_size = pano_size_.height * pano_size_.width * DST_IMAGE_CHANNEL * sizeof(unsigned char);
	CUDA_CHECK_CALL(cudaMemset(dev_pano_, 0, pano_malloc_size), "cudaMemset failed!\n", -2);
	
	for(int i = 0; i < image_num_; i++)
	{
		int img_size = const_data[i].height * const_data[i].width * 3 * sizeof(unsigned char);
		CUDA_CHECK_CALL(cudaMemcpy(dev_imgs_[i].data, images[i].data, img_size, cudaMemcpyHostToDevice), 
			"cudaMemcpy images failed\n", -2);		//	2ms/f
		dim3 dimBlock(32, 16);
		dim3 dimGrid((const_data[i].warped_width + dimBlock.x - 1) / dimBlock.x, 
			(const_data[i].warped_height + dimBlock.y - 1) / dimBlock.y);
		compose<<<dimGrid, dimBlock>>>(dev_imgs_[i].data, dev_maps_[i], dev_weights_[i], dev_pano_, i, pano_size_);		//	4.1ms/f
	}
	
	CUDA_CHECK_CALL(cudaThreadSynchronize(), "cudaThreadSynchronize failed!\n", -2);
	CUDA_CHECK_CALL(cudaMemcpy(dst, dev_pano_, pano_malloc_size, cudaMemcpyDeviceToHost), 
		"cudaMemcpy to dst failed\n", -2);			//	1.4ms/f
	return 0;
}

int freeGPU()
{
	for(int i = 0; i < image_num_; i++)
	{
		cudaFree(dev_maps_[i].xmap);
		cudaFree(dev_maps_[i].ymap);
		cudaFree(dev_weights_[i].ec_weight);
		cudaFree(dev_weights_[i].blend_weight);
		cudaFree(dev_weights_[i].total_weight);
	}
	free(const_data);
	free(dev_imgs_);
	free(dev_maps_);
	free(dev_weights_);
	return 0;
}