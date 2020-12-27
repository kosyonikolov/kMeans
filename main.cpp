#include <iostream>
#include <array>
#include <random>

#include <opencv2/opencv.hpp>

int main(int argc, char** argv) 
{
    cv::Mat image = cv::Mat::zeros(512, 1024, CV_8UC3);

    std::array<cv::Scalar, 3> colors = 
    { 
        cv::Scalar(0, 0, 255), 
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 0, 0)
    };

    std::random_device rng;
    std::default_random_engine dre(rng());

    std::uniform_int_distribution<int> xDist(10, image.cols - 10);
    std::uniform_int_distribution<int> yDist(10, image.rows - 10);

    for (auto color : colors)
    {
        for (int i = 0; i < 50; i++)
        {
            cv::Point2i point = { xDist(dre), yDist(dre) };
            cv::circle(image, point, 2, color, cv::FILLED);
        }
    }

    cv::imshow("output", image);
    cv::waitKey();

    return 0;
}
