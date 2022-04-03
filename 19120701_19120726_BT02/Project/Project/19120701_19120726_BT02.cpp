#include "opencv2\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

Mat ProcessConvolution(Mat input, vector<int> kernel) {

	int row = input.rows;
	int col = input.cols;
	int kernelSize = sqrt(kernel.size());

	Mat output(row, col, CV_8UC1);

	unsigned char* data_Input = (unsigned char*) (input.data);
	unsigned char* data_Output = (unsigned char*)(output.data);
	
	for (int row = 0;  row< input.rows; row++)
		for (int col = 0; col < input.cols; col++)
		{
			if (row < kernelSize / 2 || row >= input.rows - kernelSize / 2 ||
				col < kernelSize / 2 || col >= input.cols - kernelSize / 2) 

			{
				data_Output[output.step * row + col] = 0;
				continue;
			}

			int Sum = 0, pos_Kernel = 0;

			for (int row_Kernel = -kernelSize / 2; row_Kernel <= kernelSize / 2; ++row_Kernel) 
			{
				for (int col_Kernel = -kernelSize / 2; col_Kernel <= kernelSize / 2; ++col_Kernel) 
				{
					Sum += kernel[pos_Kernel] * data_Input[input.step * (row + row_Kernel) + col + col_Kernel];
					pos_Kernel++;
				}
			}

			if (Sum < 0) Sum = 0;
			else if (Sum > 255) Sum = 255;
			data_Output[output.step * row + col] = Sum;

		}
	return output;
}



int detectBySobel(Mat src, Mat dst) {
	vector<int> S = { 1,0,-1,2,0,-2,1,0,-1 };
	vector<int> S1 = { 1 ,2 ,1, 0 ,0,0, -1, -2,-1 };

	
	Mat X_direction= ProcessConvolution(src, S);
	Mat Y_direction = ProcessConvolution(src, S1);

	namedWindow("Display window X", WINDOW_AUTOSIZE); 
	imshow("Display window X", X_direction);

	namedWindow("Display window Y", WINDOW_AUTOSIZE);
	imshow("Display window Y", Y_direction);

	unsigned char* X_data = (unsigned char*)X_direction.data;
	unsigned char* Y_data = (unsigned char*)Y_direction.data;
	unsigned char* destination = (unsigned char*)dst.data;
	int row = src.rows;
	int col = src.cols;

	for (int i = 0; i < row * col; i++)
	{
		destination[0] = MIN(sqrt(X_data[0] * X_data[0] + Y_data[0] * Y_data[0]), 255);
		X_data++, Y_data++, destination++;
	}

	namedWindow("Display window XY", WINDOW_AUTOSIZE);
	imshow("Display window XY", dst);

	waitKey(0);
	return 1;

}



int detectByPrewitt(Mat src, Mat dst) {
	vector<int> S = { 1,0,-1,1,0,-1,1,0,-1 };
	vector<int> S1 = { 1 ,1 ,1, 0 ,0 , 0, -1, -1,-1 };


	Mat X_direction = ProcessConvolution(src, S);
	Mat Y_direction = ProcessConvolution(src, S1);

	namedWindow("Display window X_1", WINDOW_AUTOSIZE);
	imshow("Display window X_1", X_direction);

	namedWindow("Display window Y_1", WINDOW_AUTOSIZE);
	imshow("Display window Y_1", Y_direction);

	unsigned char* X_data = (unsigned char*)X_direction.data;
	unsigned char* Y_data = (unsigned char*)Y_direction.data;
	unsigned char* destination = (unsigned char*)dst.data;
	int row = src.rows;
	int col = src.cols;

	for (int i = 0; i < row * col; i++)
	{
		destination[0] = MIN(sqrt(X_data[0] * X_data[0] + Y_data[0] * Y_data[0]), 255);
		X_data++, Y_data++, destination++;
	}

	namedWindow("Display window XY_1", WINDOW_AUTOSIZE);
	imshow("Display window XY_1", dst);

	waitKey(0);
	return 1;
}



int main(int argc, char** argv)
{
	if (argc == 3) {
		Mat src;
		src = imread(argv[1], CV_8UC1);
		if (!src.data) {
			cout << "Khong the mo anh" << std::endl;
			return -1;
		}

		if (strcmp(argv[2], "Sobel") == 0) {
			Mat dst(src.rows, src.cols, CV_8UC1);
			detectBySobel(src, dst);
		}
		else if (strcmp(argv[2], "Prewitt") == 0) {
			Mat dst(src.rows, src.cols, CV_8UC1);
			detectByPrewitt(src, dst);
		}
		else {
			cout << " Loi ma lenh " << endl;
		}

		return 0;
	}

	
}



