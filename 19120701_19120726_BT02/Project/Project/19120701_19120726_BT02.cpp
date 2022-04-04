#include "opencv2\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\imgcodecs.hpp"
#include <iostream>

using namespace std;
using namespace cv;

Mat ProcessConvolution(Mat input, vector<int> kernel) {

	int row = input.rows; //get row and col of image input
	int col = input.cols;

	int kernelSize = sqrt(kernel.size()); //get size of mask

	Mat output(row, col, CV_8UC1); //create a image output

	unsigned char* data_Input = (unsigned char*) (input.data);	//point to memory contain data of input and output image
	unsigned char* data_Output = (unsigned char*)(output.data);
	
	for (int row = 0;  row< input.rows; row++)
		for (int col = 0; col < input.cols; col++)
		{
			if (row < kernelSize / 2 || row >= input.rows - kernelSize / 2 ||  //check if the boundary
				col < kernelSize / 2 || col >= input.cols - kernelSize / 2) 

			{
				data_Output[output.step * row + col] = 0; // set value of boundary = 0 
				continue;
			}

			int Sum = 0, pos_Kernel = 0; 

			for (int row_Kernel = -kernelSize / 2; row_Kernel <= kernelSize / 2; ++row_Kernel) 
			{
				for (int col_Kernel = -kernelSize / 2; col_Kernel <= kernelSize / 2; ++col_Kernel) 
				{
					Sum += kernel[pos_Kernel] * data_Input[input.step * (row + row_Kernel) + col + col_Kernel];  //convolution input image with mask is kernel parameters 
					pos_Kernel++;
				}
			}

			if (Sum < 0) Sum = 0;
			else if (Sum > 255) Sum = 255;	//set color value in [0, 255]

			data_Output[output.step * row + col] = Sum; //assign the results found above to the output

		}
	return output;
}



int detectBySobel(Mat src, Mat dst) {
	vector<int> S = { 1,0,-1,2,0,-2,1,0,-1 }; //mask with X direction  
	vector<int> S1 = { 1 ,2 ,1, 0 ,0,0, -1, -2,-1 };//mask with Y direction 

	
	Mat X_direction= ProcessConvolution(src, S);	//convolution with X direction
	Mat Y_direction = ProcessConvolution(src, S1);	//convolution with Y direction

	namedWindow("X direction", WINDOW_GUI_NORMAL);//display image
	imshow("X direction", X_direction);

	namedWindow("Y direction", WINDOW_GUI_NORMAL);
	imshow("Y direction", Y_direction);

	unsigned char* X_data = (unsigned char*)X_direction.data; //point to memory contain data of X and Y direction and destination image
	unsigned char* Y_data = (unsigned char*)Y_direction.data;
	unsigned char* destination = (unsigned char*)dst.data;

	int row = src.rows;	//get row and col of src image
	int col = src.cols;

	for (int i = 0; i < row * col; i++)
	{
		destination[0] = MIN(sqrt(X_data[0] * X_data[0] + Y_data[0] * Y_data[0]), 255);  //calculate gradient XY_direction 
		X_data++, Y_data++, destination++;
	}

	namedWindow("XY direction" , WINDOW_GUI_NORMAL);
	imshow("XY direction", dst);

	waitKey(0);
	return 1;

}



int detectByPrewitt(Mat src, Mat dst) {
	vector<int> S = { 1,0,-1,1,0,-1,1,0,-1 };  //mask with X direction  
	vector<int> S1 = { 1 ,1 ,1, 0 ,0 , 0, -1, -1,-1 };//mask with Y direction 


	Mat X_direction = ProcessConvolution(src, S);//convolution with X direction
	Mat Y_direction = ProcessConvolution(src, S1);//convolution with Y direction

	namedWindow("X direction", WINDOW_GUI_NORMAL);//display image
	imshow("X direction", X_direction);

	namedWindow("Y direction", WINDOW_GUI_NORMAL);
	imshow("Y direction", Y_direction);

	unsigned char* X_data = (unsigned char*)X_direction.data; //point to memory contain data of X and Y direction and destination image
	unsigned char* Y_data = (unsigned char*)Y_direction.data;
	unsigned char* destination = (unsigned char*)dst.data;
	int row = src.rows;
	int col = src.cols;

	for (int i = 0; i < row * col; i++)
	{
		destination[0] = MIN(sqrt(X_data[0] * X_data[0] + Y_data[0] * Y_data[0]), 255); //calculate gradient XY_direction 
		X_data++, Y_data++, destination++;
	}

	namedWindow("XY direction", WINDOW_GUI_NORMAL);
	imshow("XY direction", dst);

	waitKey(0);
	return 1;
}



int detectByLaplace(Mat src, Mat dst) {
	vector<int> S = { 0, -1, 0, -1, 4, -1, 0, -1, 0 };  //mask

	int tmp = 0;
	int counter = 0;
	Mat laplaceImage = Mat::ones(src.rows - 2, src.cols - 2, CV_32F);
    dst = Mat::ones(src.rows - 2, src.cols - 2, CV_32F);


	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{

			for (int k = i - 1; k < i + 2; k++)
			{
				for (int l = j - 1; l < j + 2; l++)
				{
					tmp += S[counter] * static_cast<int>(src.at<uchar>(k, l));
					counter++;
				}
			}
			//laplaceImage.at<float>(i - 1, j - 1) = tmp;
			dst.at<float>(i - 1, j - 1) = tmp;
			tmp = 0;
			counter = 0;
		}
	}

	dst.convertTo(dst, CV_8U);

	namedWindow("Detect by Laplace", WINDOW_GUI_NORMAL);
	imshow("Detect by Laplace", dst);

	waitKey(0);
	return 1;
}



int detectByCanny(Mat src, Mat dst) {
	
	// Upper threshold and Lower threshold
	int upperThreshold = 80;
	int lowerThreshold = 1;

	cv::Mat workImg = cv::Mat(src);
	//Clone source image
	workImg = src.clone();

	// Step 1: Noise reduction
	cv::GaussianBlur(src, workImg, cv::Size(5, 5), 1.4);

	// Step 2: Calculating gradient magnitudes and directions
	cv::Mat magX = cv::Mat(src.rows, src.cols, CV_32F);
	cv::Mat magY = cv::Mat(src.rows, src.cols, CV_32F);
	cv::Sobel(workImg, magX, CV_32F, 1, 0, 3);
	cv::Sobel(workImg, magY, CV_32F, 0, 1, 3);

	//Calculate slope: derivative of y by x.
	cv::Mat direction = cv::Mat(workImg.rows, workImg.cols, CV_32F);
	cv::divide(magY, magX, direction);

	//Magnitude of gradient 
	cv::Mat sum = cv::Mat(workImg.rows, workImg.cols, CV_64F);
	cv::Mat prodX = cv::Mat(workImg.rows, workImg.cols, CV_64F);
	cv::Mat prodY = cv::Mat(workImg.rows, workImg.cols, CV_64F);
	cv::multiply(magX, magX, prodX);
	cv::multiply(magY, magY, prodY);
	sum = prodX + prodY;
	cv::sqrt(sum, sum);

	dst = cv::Mat(src.rows, src.cols, CV_8U);

	// Initialie image to return to zero
	dst.setTo(cv::Scalar(0));


	cv::MatIterator_<float>itMag = sum.begin<float>();
	cv::MatIterator_<float>itDirection = direction.begin<float>();

	cv::MatIterator_<unsigned char>itRet = dst.begin<unsigned char>();

	cv::MatIterator_<float>itend = sum.end<float>();


	for (; itMag != itend; ++itDirection, ++itRet, ++itMag)
	{
		const cv::Point pos = itRet.pos();

		float currentDirection = atan(*itDirection) * 180 / 3.142;

		while (currentDirection < 0) currentDirection += 180;

		*itDirection = currentDirection;

		if (*itMag < upperThreshold) continue;

		bool flag = true;
		if (currentDirection > 112.5 && currentDirection <= 157.5)
		{
			if (pos.y > 0 && pos.x < workImg.cols - 1 && *itMag <= sum.at<float>(pos.y - 1, pos.x + 1)) flag = false;
			if (pos.y < workImg.rows - 1 && pos.x>0 && *itMag <= sum.at<float>(pos.y + 1, pos.x - 1)) flag = false;
		}

		else if (currentDirection > 67.5 && currentDirection <= 112.5)
		{
			if (pos.y > 0 && *itMag <= sum.at<float>(pos.y - 1, pos.x)) flag = false;
			if (pos.y < workImg.rows - 1 && *itMag <= sum.at<float>(pos.y + 1, pos.x)) flag = false;
		}
		else if (currentDirection > 22.5 && currentDirection <= 67.5)
		{
			if (pos.y > 0 && pos.x > 0 && *itMag <= sum.at<float>(pos.y - 1, pos.x - 1)) flag = false;
			if (pos.y < workImg.rows - 1 && pos.x < workImg.cols - 1 && *itMag <= sum.at<float>(pos.y + 1, pos.x + 1)) flag = false;
		}
		else
		{
			if (pos.x > 0 && *itMag <= sum.at<float>(pos.y, pos.x - 1)) flag = false;
			if (pos.x < workImg.cols - 1 && *itMag <= sum.at<float>(pos.y, pos.x + 1)) flag = false;
		}

		if (flag)
		{
			*itRet = 255;
		}
	}

	// Hysterysis threshold
	bool imageChanged = true;
	int i = 0;
	while (imageChanged)
	{
		imageChanged = false;
		printf("Iteration %d", i);
		i++;
		itMag = sum.begin<float>();
		itDirection = direction.begin<float>();
		itRet = dst.begin<unsigned char>();
		itend = sum.end<float>();
		for (; itMag != itend; ++itMag, ++itDirection, ++itRet)
		{
			cv::Point pos = itRet.pos();
			if (pos.x<2 || pos.x>src.cols - 2 || pos.y<2 || pos.y>src.rows - 2)
				continue;
			float currentDirection = *itDirection;
			// Do we have a pixel we already know as an edge?
			if (*itRet == 255)
			{
				*itRet = (unsigned char)64;
				if (currentDirection > 112.5 && currentDirection <= 157.5)
				{
					if (pos.y > 0 && pos.x > 0)
					{
						if (lowerThreshold <= sum.at<float>(pos.y - 1, pos.x - 1) &&
							dst.at<unsigned char>(pos.y - 1, pos.x - 1) != 64 &&
							direction.at<float>(pos.y - 1, pos.x - 1) > 112.5 &&
							direction.at<float>(pos.y - 1, pos.x - 1) <= 157.5 &&
							sum.at<float>(pos.y - 1, pos.x - 1) > sum.at<float>(pos.y - 2, pos.x) &&
							sum.at<float>(pos.y - 1, pos.x - 1) > sum.at<float>(pos.y, pos.x - 2))
						{
							dst.ptr<unsigned char>(pos.y - 1, pos.x - 1)[0] = 255;
							imageChanged = true;
						}
					}
					if (pos.y < workImg.rows - 1 && pos.x < workImg.cols - 1)
					{
						if (lowerThreshold <= sum.at<float>(cv::Point(pos.x + 1, pos.y + 1)) &&
							dst.at<unsigned char>(pos.y + 1, pos.x + 1) != 64 &&
							direction.at<float>(pos.y + 1, pos.x + 1) > 112.5 &&
							direction.at<float>(pos.y + 1, pos.x + 1) <= 157.5 &&
							sum.at<float>(pos.y - 1, pos.x - 1) > sum.at<float>(pos.y + 2, pos.x) &&
							sum.at<float>(pos.y - 1, pos.x - 1) > sum.at<float>(pos.y, pos.x + 2))
						{
							dst.ptr<unsigned char>(pos.y + 1, pos.x + 1)[0] = 255;
							imageChanged = true;
						}
					}
				}

				else if (currentDirection > 67.5 && currentDirection <= 112.5)
				{
					if (pos.x > 0)
					{
						if (lowerThreshold <= sum.at<float>(cv::Point(pos.x - 1, pos.y)) &&
							dst.at<unsigned char>(pos.y, pos.x - 1) != 64 &&
							direction.at<float>(pos.y, pos.x - 1) > 67.5 &&
							direction.at<float>(pos.y, pos.x - 1) <= 112.5 &&
							sum.at<float>(pos.y, pos.x - 1) > sum.at<float>(pos.y - 1, pos.x - 1) &&
							sum.at<float>(pos.y, pos.x - 1) > sum.at<float>(pos.y + 1, pos.x - 1))
						{
							dst.ptr<unsigned char>(pos.y, pos.x - 1)[0] = 255;
							imageChanged = true;
						}
					}
					if (pos.x < workImg.cols - 1)
					{
						if (lowerThreshold <= sum.at<float>(cv::Point(pos.x + 1, pos.y)) &&
							dst.at<unsigned char>(pos.y, pos.x + 1) != 64 &&
							direction.at<float>(pos.y, pos.x + 1) > 67.5 &&
							direction.at<float>(pos.y, pos.x + 1) <= 112.5 &&
							sum.at<float>(pos.y, pos.x + 1) > sum.at<float>(pos.y - 1, pos.x + 1) &&
							sum.at<float>(pos.y, pos.x + 1) > sum.at<float>(pos.y + 1, pos.x + 1))
						{
							dst.ptr<unsigned char>(pos.y, pos.x + 1)[0] = 255;
							imageChanged = true;
						}
					}
				}
				else if (currentDirection > 22.5 && currentDirection <= 67.5)
				{
					if (pos.y > 0 && pos.x < workImg.cols - 1)
					{
						if (lowerThreshold <= sum.at<float>(cv::Point(pos.x + 1, pos.y - 1)) &&
							dst.at<unsigned char>(pos.y - 1, pos.x + 1) != 64 &&
							direction.at<float>(pos.y - 1, pos.x + 1) > 22.5 &&
							direction.at<float>(pos.y - 1, pos.x + 1) <= 67.5 &&
							sum.at<float>(pos.y - 1, pos.x + 1) > sum.at<float>(pos.y - 2, pos.x) &&
							sum.at<float>(pos.y - 1, pos.x + 1) > sum.at<float>(pos.y, pos.x + 2))
						{
							dst.ptr<unsigned char>(pos.y - 1, pos.x + 1)[0] = 255;
							imageChanged = true;
						}
					}
					if (pos.y < workImg.rows - 1 && pos.x>0)
					{
						if (lowerThreshold <= sum.at<float>(cv::Point(pos.x - 1, pos.y + 1)) &&
							dst.at<unsigned char>(pos.y + 1, pos.x - 1) != 64 &&
							direction.at<float>(pos.y + 1, pos.x - 1) > 22.5 &&
							direction.at<float>(pos.y + 1, pos.x - 1) <= 67.5 &&
							sum.at<float>(pos.y + 1, pos.x - 1) > sum.at<float>(pos.y, pos.x - 2) &&
							sum.at<float>(pos.y + 1, pos.x - 1) > sum.at<float>(pos.y + 2, pos.x))
						{
							dst.ptr<unsigned char>(pos.y + 1, pos.x - 1)[0] = 255;
							imageChanged = true;
						}
					}
				}
				else
				{
					if (pos.y > 0)
					{
						if (lowerThreshold <= sum.at<float>(cv::Point(pos.x, pos.y - 1)) &&
							dst.at<unsigned char>(pos.y - 1, pos.x) != 64 &&
							(direction.at<float>(pos.y - 1, pos.x) < 22.5 ||
								direction.at<float>(pos.y - 1, pos.x) >= 157.5) &&
							sum.at<float>(pos.y - 1, pos.x) > sum.at<float>(pos.y - 1, pos.x - 1) &&
							sum.at<float>(pos.y - 1, pos.x) > sum.at<float>(pos.y - 1, pos.x + 2))
						{
							dst.ptr<unsigned char>(pos.y - 1, pos.x)[0] = 255;
							imageChanged = true;
						}
					}
					if (pos.y < workImg.rows - 1)
					{
						if (lowerThreshold <= sum.at<float>(cv::Point(pos.x, pos.y + 1)) &&
							dst.at<unsigned char>(pos.y + 1, pos.x) != 64 &&
							(direction.at<float>(pos.y + 1, pos.x) < 22.5 ||
								direction.at<float>(pos.y + 1, pos.x) >= 157.5) &&
							sum.at<float>(pos.y + 1, pos.x) > sum.at<float>(pos.y + 1, pos.x - 1) &&
							sum.at<float>(pos.y + 1, pos.x) > sum.at<float>(pos.y + 1, pos.x + 1))
						{
							dst.ptr<unsigned char>(pos.y + 1, pos.x)[0] = 255;
							imageChanged = true;
						}
					}
				}
			}
		}
	}
	cv::MatIterator_<unsigned char>current = dst.begin<unsigned char>();    cv::MatIterator_<unsigned char>final = dst.end<unsigned char>();
	for (; current != final; ++current)
	{
		if (*current == 64)
			*current = 255;
	}

	namedWindow("Detect by Canny", WINDOW_GUI_NORMAL);
	imshow("Detect by Canny", dst);

	waitKey(0);
	return 1;
}



int detectByCannyOpenCV(Mat src, Mat dst) {
	Mat src_gray;
	Mat detected_edges;
	Mat src_blurred, src_canny;


	cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);        // convert to grayscale

	cv::GaussianBlur(src_gray, src_blurred, cv::Size(5, 5), 1.5);                               

	cv::Canny(src_blurred, dst, 1, 80); 

	namedWindow("Detect by Canny OpenCV function", WINDOW_GUI_NORMAL);
	imshow("Detect by Canny OpenCV function", dst);

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
		else if (strcmp(argv[2], "Laplace") == 0) {
			Mat dst(src.rows, src.cols, CV_8UC1);
			detectByLaplace(src, dst);
		}
		else if (strcmp(argv[2], "Canny") == 0) {
			Mat dst(src.rows, src.cols, CV_8UC1);
			detectByCanny(src, dst);
		}
		else if (strcmp(argv[2], "CannyOpenCV") == 0) {
			src = imread(argv[1], IMREAD_COLOR);
			Mat dst;
			dst.create(src.size(), src.type());
			detectByCannyOpenCV(src, dst);
		}
		else {
			cout << " Loi ma lenh " << endl;
		}

		return 0;
	}

	
}



