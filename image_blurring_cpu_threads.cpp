/*
    Image Blurring on CPU with Threads
*/

#include <iostream>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <omp.h>

// Custom OpenCV libraries
#include <opencv2/core/core.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// float kernel[5][5] = {{0.04, 0.04, 0.04, 0.04, 0.04},
//                       {0.04, 0.04, 0.04, 0.04, 0.04},
//                       {0.04, 0.04, 0.04, 0.04, 0.04},
//                       {0.04, 0.04, 0.04, 0.04, 0.04},
//                       {0.04, 0.04, 0.04, 0.04, 0.04}};

////////// FUNCTION DECLARATIONS /////////

void blurImage(const cv::Mat& input, cv::Mat& output);

////////// FUNCTION DEFINITIONS /////////

// Blur image function
void blurImage(const cv::Mat& input, cv::Mat& output){
    // Variables to uses
    int i, j, i2, j2;
    int kernelX, kernelY;

    #pragma omp parallel for private(i, j, kernelX, kernelY) shared(input, output) // parallel
    // Iteration from rows and colmns
    for(i = 0; i < input.rows; i++) {
        for(j = 0; j < input.cols; j++) {
            float blueAVG = 0.0, greenAVG = 0.0, redAVG = 0.0, iterator = 0.0; // initialize to 0
            for(kernelX = -2; kernelX < 3; kernelX++){ // Considers from -2 to 3 == 5 for the filter
                for(kernelY = -2; kernelY < 3; kernelY++){ // same
                    if((kernelY + j > 0 && kernelY + j < input.cols) && (kernelX + i > 0 && kernelX + i < input.rows)){ // border
                        // sum each element through iteration
                        blueAVG += input.at<cv::Vec3b>(kernelX + i , kernelY + j)[0];
                        greenAVG += input.at<cv::Vec3b>(kernelX + i , kernelY + j)[1];
                        redAVG += input.at<cv::Vec3b>(kernelX + i , kernelY + j)[2];
                        iterator += 1.0;
                    }
                }
            }
            // Average per color
            output.at<cv::Vec3b>(i, j)[0] = blueAVG / iterator;
            output.at<cv::Vec3b>(i, j)[1] = greenAVG / iterator;
            output.at<cv::Vec3b>(i, j)[2] = redAVG / iterator;
        }
    }
}

// Main function
int main(int argc, char *argv[]) {
    std::cout << "\n";
    std::cout << "---------- IMAGE BLURRING CPU (THREADS) ----------" << "\n";
    std::cout << "\n";

    std::string imagePath; // path to image

    if(argc < 2){
        imagePath = "./images/image_4k.jpg";
    } else {
        imagePath = argv[1];
    }

    // Read input image from the disk
	cv::Mat input = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (input.empty()){ // in case of not finding the image
        std::cout << "Image Not Found!" << std::endl;
		std::cin.get();
		return -1;
	}

	cv::Mat output(input.rows, input.cols, CV_8UC3); // Create output image
    //output = input.clone(); // Info from input to output

    omp_set_num_threads(6); // Thread control

    auto start_cpu = std::chrono::high_resolution_clock::now();
	blurImage(input, output); // Call the blurImage function
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    std::cout << "Time elapsed: " <<  duration_ms.count() << " ms" << '\n'; // Time elapsed

    //imwrite( "output.jpg", output );

    // Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
	imshow("Output", output);

	cv::waitKey(); // Wait for key press

    return 0;
}
