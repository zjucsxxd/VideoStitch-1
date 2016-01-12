#ifndef __MY_COMPOSER_GPU_H__
#define __MY_COMPOSER_GPU_H__

#include "gutil.h"

extern int testGPU();

extern int initGPU(int n);

extern int initdataCopy2GPU(C2GInitData *c2g_data, int pano_height, int pano_width);

extern int composeGPU(GPUImageData *images, unsigned char *dst);

extern int freeGPU();

#endif
