#include <iostream>
#include <array>
#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

#include <opencv2/opencv.hpp>

// ================================================
// ================= Boring stuff =================
// ================================================

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

// ================================================
// ================= Debug images =================
// ================================================

void drawPoints(cv::Mat & image,
                const std::vector<cv::Point2d> & points,
                const std::vector<cv::Point2d> & centers,
                const std::vector<int> & clusterId,
                const ImageDesciptor & desc)
{
    cv::rectangle(image,
                  cv::Rect(0, 0, image.cols, image.rows),
                  cv::Scalar(0, 0, 0),
                  cv::FILLED);

    std::array<cv::Scalar, 8> colors = 
    {
        cv::Scalar(0, 0, 255), // red
        cv::Scalar(0, 255, 0), // green
        cv::Scalar(255, 0, 0), // blue
        cv::Scalar(0, 0, 127), // dark red
        cv::Scalar(0, 127, 0), // dark green
        cv::Scalar(127, 0, 0), // dark blue
        cv::Scalar(0, 255, 255), // orange
        cv::Scalar(255, 255, 0)  // purple
    };

    // ======== Draw points ========
    for (int i = 0; i < points.size(); i++)
    {
        const cv::Point2i point = desc.pt2img(points[i]);
        cv::circle(image, point, 2, colors[clusterId[i]], cv::FILLED);
    }

    // ======== Draw centers ========
    for (int i = 0; i < centers.size(); i++)
    {
        const cv::Point2i point = desc.pt2img(centers[i]);
        const cv::Scalar color  = colors[i];
        
        cv::line(image, 
                 point - cv::Point2i(0, 5),
                 point + cv::Point2i(0, 5),
                 color);

        cv::line(image, 
                 point - cv::Point2i(5, 0),
                 point + cv::Point2i(5, 0),
                 color);
    }
}

// ================================================
// =================== Algorithm ==================
// ================================================

double sqDistance(const cv::Point2d & a, const cv::Point2d & b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

template<typename _URNG>
std::vector<cv::Point2d> makeInitialCenters(const ImageDesciptor & desc,
                                            const int count,
                                            _URNG & rng)
{
    std::vector<cv::Point2d> result(count);

    std::uniform_real_distribution<double> xDist(desc.xMin, desc.xMax);
    std::uniform_real_distribution<double> yDist(desc.yMin, desc.yMax);

    for (int i = 0; i < count; i++)
    {
        result[i].x = xDist(rng);
        result[i].y = yDist(rng);
    }

    return result;
}
    
bool classifyAndUpdate(const std::vector<cv::Point2d> & points,
                       const std::vector<cv::Point2d> & centers,
                       std::vector<int> & outClusterId,
                       std::vector<cv::Point2d> & outNewCenters)
{
    const int clusters = centers.size();
    for (int i = 0; i < clusters; i++) outNewCenters[i] = cv::Point2d(0, 0);

    std::vector<int> count(clusters);

    // ======== Assign each point to one cluster ========
    for (int i = 0; i < points.size(); i++)
    {
        const auto & pt = points[i];
        
        int id = 0;
        double dist = sqDistance(pt, centers[0]);
        for (int j = 1; j < clusters; j++)
        {
            const double cand = sqDistance(pt, centers[j]);
            if (cand < dist)
            {
                id = j;
                dist = cand;
            }
        }

        outClusterId[i] = id;
        count[id]++;
        outNewCenters[id] += pt;
    }

    // ======== Compute new cluster centers and compare with old ========
    bool haveDifferent = false;
    for (int i = 0; i < clusters; i++)
    {
        outNewCenters[i] /= count[i];
        haveDifferent |= outNewCenters[i] != centers[i];
    }

    return haveDifferent;
}

int main(int argc, char** argv) 
{
    const std::string USAGE_MSG = "Usage: ./kMeans [points] [cluster count]";

    cv::Point2d test;

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

    std::random_device rng;
    std::default_random_engine dre(rng());

    std::vector<cv::Point2d> centers = makeInitialCenters(desc, clusters, dre);
    std::vector<cv::Point2d> newCenters = centers;
    std::vector<int> clusterId(points.size());

    while (classifyAndUpdate(points, centers, clusterId, newCenters))
    {
        drawPoints(image, points, centers, clusterId, desc);
        cv::imshow("output", image);
        cv::waitKey();
        centers = newCenters;
    }

    drawPoints(image, points, centers, clusterId, desc);
    cv::imshow("output", image);

    return 0;
}
