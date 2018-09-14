/*
    Matrix Multiplication on GPU
*/

#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include "common.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

////////// FUNCTION DECLARATIONS //////////

__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int step);
void blurImage(const cv::Mat& input, cv::Mat& output);

////////// FUNCTION DEFINITIONS //////////

// Kernel for the blur image
__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int step){
	// thread index
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// inside correct threads
	if((xIndex < width) && (yIndex < height)){
		int blueAVG = 0, greenAVG = 0, redAVG = 0, iterator = 0; // initialize to 0
		const int image_tid = yIndex * step + (3 * xIndex); // pixel location
		for(int kernelX = -2; kernelX < 3; kernelX++){ // Considers from -2 to 3 == 5 for the filter
			for(int kernelY = -2; kernelY < 3; kernelY++){ // same
				int tid = (yIndex + kernelY) * step + (3 * (xIndex + kernelX)); // change in pixel matrix
                if((yIndex + kernelY) % height > 1 && (xIndex + kernelX) % width > 1 ) {
                    blueAVG += input[tid];
                    greenAVG += input[tid + 1];
                    redAVG += input[tid + 2];
                    iterator++;
                }
			}
		}
		// Each color average for output image
		output[image_tid] = static_cast<unsigned char>(blueAVG / iterator);
        output[image_tid + 1] = static_cast<unsigned char>(greenAVG / iterator);
        output[image_tid + 2] = static_cast<unsigned char>(redAVG / iterator);
	}
}

void blurImage(const cv::Mat& input, cv::Mat& output){

	std::cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << std::endl;

	const int bytes = input.step * input.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc(&d_input, bytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc(&d_output, bytes), "CUDA Malloc Failed");
	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), bytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(1, 128);

	// Calculate grid size to cover the whole image
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
	printf("blur_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Launch the blur kernel
	auto start_cpu =  std::chrono::high_resolution_clock::now();
	blur_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step));
	auto end_cpu =  std::chrono::high_resolution_clock::now();

	std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
	printf("Time elapsed %f ms\n", duration_ms.count());

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, bytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}

int main(int argc, char *argv[]){

	std::string imagePath; // path to image

	if(argc < 2){
		imagePath = "./images/image_4k.jpg";
	} else{
		imagePath = argv[1];
	}

	// Read input image
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty()){ // if no image found
		std::cout << "Image Not Found!" << std::endl;
		std::cin.get();
		return -1;
	}

	//Create output image
	cv::Mat output(input.rows, input.cols, CV_8UC3);
	//output = input.clone();

	// BlurImage function
	blurImage(input, output);

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
