#include <iostream>
#include <array>
#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <utility>
#include <cmath>

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
        yMin = std::min(yMin, pt.y);
        yMax = std::max(yMax, pt.y);
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
        cv::circle(image, point, 1, colors[clusterId[i]], cv::FILLED);
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
std::vector<cv::Point2d> makeInitialCenters(const std::vector<cv::Point2d> & points,
                                            const int count,
                                            _URNG & rng)
{
    std::vector<cv::Point2d> result(count);

    std::uniform_int_distribution<int> idxDist(0, points.size() - 1);

    for (int i = 0; i < count; i++)
    {
        result[i] = points[idxDist(rng)];
    }

    return result;
}

template<typename _URNG>
std::vector<cv::Point2d> makeInitialCenters2(const std::vector<cv::Point2d> & points,
                                             const int count,
                                             _URNG & rng)
{
    std::vector<cv::Point2d> result(count);

    std::uniform_int_distribution<int> idxDist(0, points.size() - 1);
    result[0] = points[idxDist(rng)];

    std::vector<std::pair<double, int>> distance(points.size());

    for (int i = 1; i < count; i++)
    {
        double distSum = 0;
        
        // calc all distances
        for (int j = 0; j < points.size(); j++)
        {
            distance[j].second = j;

            const auto & pt = points[j];
            double dist = sqDistance(pt, result[0]);
            for (int k = 1; k < i; k++) dist = std::min(dist, sqDistance(pt, result[k]));
            
            distance[j].first = dist;
            distSum += dist;
        }

        // sort by distances, then select point
        std::sort(distance.begin(), distance.end(), std::greater<std::pair<double, int>>());

        std::uniform_real_distribution<double> pDist(0, distSum);
        const double p = pDist(rng);

        int idx = 0;
        double sum = 0;
        while (idx < distance.size())
        {
            sum += distance[idx].first;
            if (sum >= p) break;
            idx++;
        } 
    
        idx = std::min(idx, (int)points.size() - i - 1); // prevent reusing already used centers
        result[i] = points[distance[idx].second];
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
        outNewCenters[i] /= std::max(1, count[i]);
        haveDifferent |= outNewCenters[i] != centers[i];
    }

    return haveDifferent;
}

void calcStDevAvgDist(const std::vector<cv::Point2d> & points,
                      const std::vector<cv::Point2d> & centers,
                      const std::vector<int> & clusterId,
                      std::vector<double> & outStDev,
                      std::vector<double> & outAvgDist)
{
    for (int i = 0; i < centers.size(); i++)
    {
        double sqSum = 0;
        double sum = 0;
        int count = 0;

        const auto & center = centers[i];

        for (int j = 0; j < points.size(); j++)
        {
            if (clusterId[j] != i) continue;
            
            const double sqDist = sqDistance(center, points[j]);
            sqSum += sqDist;
            sum += std::sqrt(sqDist);
            count++;
        }

        outStDev[i] = std::sqrt(sqSum / std::max(1, count - 1)); // avoid div0
        outAvgDist[i] = sum / std::max(1, count);
    }
}

double calcScore(const std::vector<double> & stDev,
                 const std::vector<double> & avgDist)
{
    double score = 0;
    for (int i = 0; i < stDev.size(); i++)
    {
        score += stDev[i] * avgDist[i];
    }

    return std::sqrt(score);
}

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

    std::random_device rng;
    std::default_random_engine dre(rng());

    const int nRestarts = 20;

    // output buffers
    std::vector<cv::Point2d> bestCenters;
    std::vector<int> bestIds;
    double bestScore = 1e300;

    // work buffers
    std::vector<int> clusterId(points.size());
    std::vector<double> stDev(points.size());
    std::vector<double> avgDist(points.size());

    for (int i = 0; i < nRestarts; i++)
    {
        std::vector<cv::Point2d> centers = makeInitialCenters2(points, clusters, dre);
        std::vector<cv::Point2d> newCenters = centers;

        int nIters = 0;

        while (classifyAndUpdate(points, centers, clusterId, newCenters))
        {
            drawPoints(image, points, centers, clusterId, desc);
            //cv::imshow("iteration", image);
            //cv::waitKey(16);
            centers = newCenters;
            nIters++;
        }

        calcStDevAvgDist(points, centers, clusterId, stDev, avgDist);
        const double score = calcScore(stDev, avgDist);

        if (score < bestScore)
        {
            bestScore = score;
            bestCenters = centers;
            bestIds = clusterId;
        }

        std::cout << i << "\t" << nIters << "\t" << score << "\n";

        // std::cout << "AvgDist\tStDev\n";
        // for (int i = 0; i < centers.size(); i++)
        // {
        //     std::cout << avgDist[i] << "\t" << stDev[i] << "\n";
        // }
    }

    //cv::destroyAllWindows();

    drawPoints(image, points, bestCenters, bestIds, desc);
    cv::imshow("output", image);
    cv::imwrite("output.bmp", image);
    cv::waitKey();

    return 0;
}
