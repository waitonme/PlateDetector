//
//  trainAdvanced.cpp
//  PlateDetector
//
//  Created by near on 2020/06/04.
//  Copyright © 2020 near. All rights reserved.
//

#include "trainAdvanced.hpp"
#include <glob.h>
#include <stdexcept>

std::vector<cv::String> glob(const std::string& pattern) {
    using namespace std;

    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if(return_value != 0) {
        globfree(&glob_result);
        stringstream ss;
        ss << "glob() failed with return_value " << return_value << endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a std::list<std::string>
    vector<string> filenames;
    for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.push_back(string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);

    // done
    return filenames;
}

using namespace cv;
using namespace cv::ml;
using namespace std;




//alphabetData = []
//digitsData = []
//alphabetLabels = []
//digitsLabels = []

void train(){
int *digitsLabels;
int *alphabetLabels;
const char path[] = "/Users/near/Documents/venv/examples";

vector<cv::String> filenames= glob(path);

for (int i = 0; i < filenames.size(); i++) {
    String filename = filenames[i];
    Mat input = imread(filename);
    cout << filenames[i] << endl;
};


Ptr<SVM> svm = SVM::create();

}

/* loop over the sample character paths
for samplePath in sorted(glob.glob(args["samples"] + "/*")):
    # extract the sample name, grab all images in the sample path, and sample them
    sampleName = samplePath[samplePath.rfind("/") + 1:]
    imagePaths = list(paths.list_images(samplePath))
    imagePaths = random.sample(imagePaths, min(len(imagePaths), args["min_samples"]))

    # loop over all images in the sample path
    for imagePath in imagePaths:
        # load the character, convert it to grayscale, preprocess it, and describe it
        char = cv2.imread(imagePath)
        char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
        char = LicensePlateDetector.preprocessChar(char)
        features = desc.describe(char)

        # check to see if we are examining a digit
        if sampleName.isdigit():
            digitsData.append(features)
            digitsLabels.append(sampleName)

        # otherwise, we are examining an alphabetical character
        else:
            alphabetData.append(features)
            alphabetLabels.append(sampleName)
*/
