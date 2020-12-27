#include <iostream>
#include <array>
#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

#include <opencv2/opencv.hpp>

struct ImageDesciptor
{
    double xMin, xMax;
    double yMin, yMax;

    double pt2ImgX, pt2ImgXOffset;
    double pt2ImgY, pt2ImgYOffset;

    cv::Point2d pt2img(const cv::Point2d & point) const
    {
        return cv::Point2d(pt2ImgX * point.x + pt2ImgXOffset,
                           pt2ImgY * point.y + pt2ImgYOffset);
    }
};

std::vector<cv::Point2d> readPoints(std::istream & stream)
{
    std::vector<cv::Point2d> result;

    cv::Point2d current;
    while ((stream >> current.x >> current.y))
    {
        result.push_back(current);
    }

    return result;
}

ImageDesciptor createDescriptor(const std::vector<cv::Point2d> & points, 
                                const int width, const int height)
{
    ImageDesciptor result;

    double xMin = points[0].x;
    double xMax = xMin;
    double yMin = points[0].y;
    double yMax = yMin;

    for (int i = 1; i < points.size(); i++)
    {
        const auto & pt = points[i];
        
        xMin = std::min(xMin, pt.x);
        xMax = std::max(xMax, pt.x);
        yMin = std::min(xMin, pt.y);
        yMax = std::max(xMax, pt.y);
    }

    result.xMin = xMin;
    result.xMax = xMax;
    result.yMin = yMin;
    result.yMax = yMax;

    // xMax - xMin -> width - 1, xMin -> 0
    result.pt2ImgX = (width - 1) / (xMax - xMin);
    result.pt2ImgXOffset = -result.pt2ImgX * xMin;

    result.pt2ImgY = (height - 1) / (yMax - yMin);
    result.pt2ImgYOffset = -result.pt2ImgY * yMin;

    return result;
};

int main(int argc, char** argv) 
{
    const std::string USAGE_MSG = "Usage: ./kMeans [points] [cluster count]";

    if (argc != 3)
    {
        std::cerr << USAGE_MSG << "\n";
        return 1;
    }

    std::ifstream inFile(argv[1]);
    if (!inFile.is_open())
    {
        std::cerr << "Can't open file " << argv[1] << "\n";
        return 1;
    }

    std::vector<cv::Point2d> points = readPoints(inFile);
    
    const int clusters = std::stoi(argv[2]);
    if (points.size() < clusters)
    {
        std::cerr << "Less points than clusters\n";
        return 1;
    }

    cv::Mat image = cv::Mat::zeros(700, 700, CV_8UC3);
    const ImageDesciptor desc = createDescriptor(points, image.cols, image.rows);

    for (auto & pt : points)
    {
        cv::Point2i point = desc.pt2img(pt);
        cv::circle(image, point, 2, cv::Scalar(255, 255 ,255), cv::FILLED);
    }

    cv::imshow("output", image);
    cv::waitKey();

    return 0;
}
