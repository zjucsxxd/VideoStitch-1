#ifndef __MY_GPU_UTIL_H__
#define __MY_GPU_UTIL_H__

typedef struct _cpu2gpu_init_data
{
	int height;
	int width;

	int warped_height;
	int warped_width;

	int corner_x;
	int corner_y;

	float *xmap;
	float *ymap;

	float *blend_weight;
	float *ec_weight;
	float *total_weight;
}
C2GInitData;

typedef struct _gpu_image_data
{
	unsigned char *data;
}
GPUImageData;

//typedef unsigned char * uchar_ptr;


#endif