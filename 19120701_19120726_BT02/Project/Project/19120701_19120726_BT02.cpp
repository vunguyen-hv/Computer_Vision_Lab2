#include "opencv2\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	const char* fname = "testing.png";
	Mat image;
	image = imread(fname, IMREAD_COLOR);
	if (!image.data)
	{
		cout << "Khong the mo anh" << std::endl;
		return -1;
	}
	namedWindow("Display window", WINDOW_AUTOSIZE); // (3)
	imshow("Display window", image); // (4)
	waitKey(0);
	return 0;
}