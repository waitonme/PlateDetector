//
//  PlateDetector.hpp
//  PlateDetector
//
//  Created by near on 2020/05/12.
//  Copyright © 2020 near. All rights reserved.
//

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class PlateDetector {
    

public:
    
    vector<Mat> detect_plate(Mat image);
    vector<Mat> detect(Mat image);
    Mat detectCharacterCandidates(Mat image,Mat region);
    vector<Mat> scissor(Mat threshc);
    
private:
    vector<vector<cv::Point>> hull;
    
};

