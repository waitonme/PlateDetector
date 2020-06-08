//
//  PlateDetector.cpp
//  PlateDetector
//
//  Created by near on 2020/05/12.
//  Copyright © 2020 near. All rights reserved.
//

#include "PlateDetector.hpp"


using namespace cv;
using namespace std;




vector<Point> grab_contours(vector<vector<Point>> cnts){
    vector<Point> result;
    if (cnts.size() == 2)
        result = cnts[0];
    else if (cnts.size() == 3)
        result = cnts[0];

        return result;
    
}

int getMaxAreaContourId(vector<vector<cv::Point>> contours) {
    
    double maxArea = 0;
    int maxAreaContourId = -1;
    for (int j = 0; j < contours.size(); j++) {
        double newArea = cv::contourArea(contours[j]);
        if (newArea > maxArea) {
            maxArea = newArea;
            maxAreaContourId = j;
        } // End if
    } // End for
    return maxAreaContourId;
} // End function

Mat four_point_transform(Mat image,Mat pst){
    Rect rect = boundingRect(pst);
    Point tl,tr,br,bl ;
    
    tl = rect.tl();
    tr = Point(tl.x + rect.width, tl.y);
    br = rect.br();
    bl = Point(br.x - rect.width, br.y);

//
//    rect = order_points(pts)
//    (tl, tr, br, bl) = rect
//
//    # compute the width of the new image, which will be the
//    # maximum distance between bottom-right and bottom-left
//    # x-coordiates or the top-right and top-left x-coordinates
    float widthA =  sqrt((pow(br.x - bl.x,2) ) + (pow(br.y - bl.y,2)));
    float widthB =  sqrt((pow(tr.x - tl.x,2)) + (pow(tr.y - tl.y,2)));
    int maxWidth =  max(int(widthA), int(widthB));
//    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
//    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
//    maxWidth = max(int(widthA), int(widthB))
//
//    # compute the height of the new image, which will be the
//    # maximum distance between the top-right and bottom-right
//    # y-coordinates or the top-left and bottom-left y-coordinates
    float heightA =sqrt((pow(tr.x - br.x,2) ) + (pow(tr.y - br.y,2)));
    float heightB =sqrt((pow(tl.x - bl.x,2) ) + (pow(tl.y - bl.y,2)));
    int maxHeight = max(int(heightA), int(heightB));
//    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
//    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
//    maxHeight = max(int(heightA), int(heightB))
//
//    # now that we have the dimensions of the new image, construct
//    # the set of destination points to obtain a "birds eye view",
//    # (i.e. top-down view) of the image, again specifying points
//    # in the top-left, top-right, bottom-right, and bottom-left
//    # order
    Point2f src[4], dst[4];
    src[0] = tl;
    src[1] = tr;
    src[2] = br;
    src[3] = bl;
    
    dst[0] = Point2f(0,0);
    dst[1] = Point2f(maxWidth-1,0);
    dst[2] = Point2f(maxWidth-1,maxHeight-1);
    dst[3] = Point2f(0,maxHeight-1);
//    dst.push_back((0,0));
    Mat trans = getPerspectiveTransform(src, dst);
    Mat result;
    warpPerspective(image,result, trans, Size(maxWidth,maxHeight));
//    dst = np.array([
//        [0, 0],
//        [maxWidth - 1, 0],
//        [maxWidth - 1, maxHeight - 1],
//        [0, maxHeight - 1]], dtype="float32")
//
//    # compute the perspective transform matrix and then apply it
//    M = cv2.getPerspectiveTransform(rect, dst)
//    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
//
//    # return the warped image
    return result;
}

double getAverage(vector<double> vector, int nElements) {
    
    double sum = 0;
    int initialIndex = 0;
    int last30Lines = int(vector.size()) - nElements;
    if (last30Lines > 0) {
        initialIndex = last30Lines;
    }
    
    for (int i=(int)initialIndex; i<vector.size(); i++) {
        sum += vector[i];
    }
    
    int size;
    if (vector.size() < nElements) {
        size = (int)vector.size();
    } else {
        size = nElements;
    }
    return (double)sum/size;
}

bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i < j );
}

bool compareRectX ( Rect r1, Rect r2 ) {
    int i = r1.x;
    int j = r2.x;
    return ( i < j );
}

vector<Mat> PlateDetector::detect(Mat image){
    vector<vector<Mat>> result;
    vector<Mat> result2;
    vector<Mat> plates = detect_plate(image);
    for(int i=0; i< plates.size();i++) {
        Mat lpRegion = plates[i];
        Mat T = detectCharacterCandidates(image, lpRegion);
        vector<Mat> chars = scissor(T);
        if (chars.size() < 3)
            continue;
        
        
        
    
//        result.push_back(chars);
        result2.push_back(T);
//        cvtColor(T, T, COLOR_GRAY2BGR);
//        Rect rect = boundingRect(lpRegion);
//        Mat insetImage(image, rect);
//        T.copyTo(insetImage);
//        c.push_back(T);
//        result.push_back(c);
        
    }
    
    return result2;
    
}



Mat PlateDetector::detectCharacterCandidates(Mat image,Mat region){
    Mat plate = four_point_transform(image, region);
//      plate = perspective.four_point_transform(self.image, region)
    Mat HSV[3];
    Mat tmp;
//    Mat plate = region;
    cvtColor(plate, tmp, COLOR_BGR2HSV);
    split(tmp, HSV);
    Mat T;
    adaptiveThreshold(HSV[2], T, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 29, 15);
    Mat thresh =  (HSV[2] > T) * 255;
    
    Mat threshc;
    thresh.copyTo(threshc);
    bitwise_not(thresh, thresh);
    
    
    resize(plate, plate, Size(400, plate.size().height));
    resize(threshc, threshc, Size(400, threshc.size().height));
    resize(thresh, thresh, Size(400, thresh.size().height));
    
    
    //           V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
    //           T = threshold_local(V, 29, offset=15, method="gaussian")
    //           thresh = (V > T).astype("uint8") * 255
    //           thresh = cv2.bitwise_not(thresh)
    //           threshc = thresh.copy()
    //           # thresh = cv2.erode(thresh, None, iterations=1)
    //           # resize the license plate region to a canonical size
    //           plate = imutils.resize(plate, width=400)
    //           threshc = imutils.resize(threshc, width=400)
    //           thresh = imutils.resize(thresh, width=400)
    //           cv2.imshow("LP Threshold", thresh)
    Mat charCandidates = Mat::zeros(threshc.size(), CV_8UC1);
    vector<vector<float>> hulls;
    vector<vector<Point>> hullz ;
//    vector<vector<Point>> contours;
//    findContours(threshc, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
//    drawContours(threshc, contours, -1, cv::Scalar(0, 0, 0), 1);
    Mat labels, stats, centroids;
    int nlabels = connectedComponentsWithStats(threshc,
        labels, stats, centroids ,8);
    
    for ( int i=0; i<nlabels; i++ )
    {
            if (i < 2)    continue;
        
            int *label = labels.ptr<int>(i);
            Mat labelMask =Mat::zeros(threshc.size(), CV_8UC1);
//            int* pixel = labelMask.ptr<int>(i);
            int* p = stats.ptr<int>(i);
            rectangle(labelMask, Rect(p[0], p[1], p[2], p[3]), Scalar(255, 255, 255), -1);
            bitwise_and(labelMask, threshc, labelMask);
//            bitwise_or(labelMask, *label, labelMask);
//            return labelMask;
//            bitwise_or(charCandidates, labelMask, charCandidates);
            vector<vector<Point>> cnts;
            vector<Vec4i> hierarchy;
            findContours(labelMask, cnts,hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//            vector<Point> cnt = grab_contours(cnts);
            sort(cnts.begin(), cnts.end(), compareContourAreas);
        
            // labelMask = np.zeros(thresh.shape, dtype="uint8")
            //              labelMask[labels == label] = 255
            //              cnts = cv2.findContours(
            //                  labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            //              cnts = imutils.grab_contours(cnts)

        if (cnts.size() > 0 ){
            int maxi = int(cnts.size()-1);
            vector<Point> biggestContour = cnts[maxi];
            if (contourArea(biggestContour) == 0)
                           continue;

            Rect rect = boundingRect(biggestContour);
            float boxX = rect.x;
            float boxY = rect.y;
            float boxW = rect.width;
            float boxH = rect.height;

            float aspectRatio = boxW / float(boxH);
            float solidity = contourArea(biggestContour) / float(boxW * boxH);
            float heightRatio = boxH / float(plate.size().height);
            
            bool keepAspectRatio = (aspectRatio < 1.0);
            bool keepSolidity = (solidity > 0.15);
            bool keepHeight = (heightRatio > 0.4 && heightRatio < 0.95);
            
            if (keepAspectRatio && keepSolidity &&keepHeight ){
                vector<Point> hull;
                convexHull(biggestContour, hull);
                vector<float> temp = {boxX,boxY,boxW ,boxH};
                hulls.push_back(temp);
                hullz.push_back(hull);
                drawContours(charCandidates,hullz, int(hullz.size())-1, Scalar(255,255,255),-1);
//                 drawContours(threshc, hulls, int(hulls.size())-1, Scalar(150,255,150),-1);
                
                
//                rectangle(threshc, Point(boxX, boxY),Point(boxX + boxW, boxY + boxH ),Scalar(0, 255, 0), 30);
            }
        }
        //  if len(cnts) > 0:
        //
        //                  c = max(cnts, key=cv2.contourArea)
        //                  (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
        //
        //                  # compute the aspect ratio, solidity, and height ratio for the component
        //                  aspectRatio = boxW / float(boxH)
        //                  solidity = cv2.contourArea(c) / float(boxW * boxH)
        //                  heightRatio = boxH / float(plate.shape[0])
        //
        //                  # determine if the aspect ratio, solidity, and height of the contour pass
        //                  # the rules tests
        //                  keepAspectRatio = (aspectRatio < 1.0)
        //                  keepSolidity = (solidity > 0.15)
        //                  keepHeight = (heightRatio > 0.4 and heightRatio < 0.95)
        //
        //                  # check to see if the component passes all the tests
        //                  if keepAspectRatio and keepSolidity and keepHeight:
        //                      # compute the convex hull of the contour and draw it on the character
        //                      # candidates mask
        //                      hull = cv2.convexHull(c)
        //                      cv2.drawContours(charCandidates, [hull], -1, 255, -1)
        //                      hulls.append((boxX, boxY, boxW, boxH))
    }
    
    
    
    vector<float> extra;
    if (hulls.size() >= 6 and hulls.size() <8){
        vector<float> x;
        vector<float> y;
        float sumy = 0;
        for(int i = 0; i < hulls.size(); i++){
            x.push_back(hulls[i][0]);
            y.push_back(hulls[i][1]);
            sumy +=hulls[i][1];
            
        }
        sort(x.begin(),x.end());
        vector<float> b;
        float sumb = 0;
        for(int i = 0; i < x.size()-1; i++){
            float s = x[i+1] - x[i];
            sumb += s;
            b.push_back(s);
        }
        float a = -1 ;
        int k = 0;
        for(int i = 0; i < b.size(); i++){
            if (a < b[i]){
                a = b[i];
                k = i;
            }
        }
        cout << k << " " << k+1 << " " << "사이" << endl;
        
        if (a < sumb/b.size()*1.7 && hulls.size() == 7){
            cout << "21 4" << endl;
        }else if(a > sumb/b.size()*1.7 && hulls.size() == 6){
            cout << "2x 4" << endl;
//            vector<float> j;
//            vector<float> l;
            float sumj = 0;
            float suml = 0;
            for(int i = 0; i < hulls.size(); i++){
//                j.push_back(hulls[i][2]);
//                l.push_back(hulls[i][3]);
                sumj +=hulls[i][2];
                suml += hulls[i][3];
            }
            extra = {x[k]+(sumb-a)/(b.size()-1), sumy/y.size(), sumj/y.size(), suml/y.size()};
            
        } else if(a > sumb/b.size()*1.7 && hulls.size() == 7){
            cout << "3x 4" << endl;
//            vector<float> j;
//            vector<float> l;
            float sumj = 0.0;
            float suml = 0;
            for(int i = 0; i < hulls.size(); i++){
//                j.push_back(hulls[i][2]);
//                l.push_back(hulls[i][3]);
                sumj +=hulls[i][2];
                suml += hulls[i][3];
            }
            extra = {x[k]+(sumb-a)/(b.size()-1), sumy/y.size(), sumj/y.size(), suml/y.size()};
            
        } else {
            cout << "?" << endl;
        }
        
    }
    if (!extra.empty()){
        vector<Point> a;
        a.push_back(Point_<float>(extra[0], extra[1]));
        a.push_back(Point_<float>(extra[0], extra[1]+extra[3]));
        a.push_back(Point_<float>(extra[0]+extra[2], extra[1]));
        a.push_back(Point_<float>(extra[0]+extra[2], extra[1]+extra[3]));
        
//        a.push_back(Point_<float>(extra[0], (extra[1]+extra[3])/2));
//        a.push_back(Point_<float>(extra[0]+extra[2], (extra[1]+extra[3])/2));
//
//        a.push_back(Point_<float>((extra[0]+extra[2])/2, extra[1]));
//         a.push_back(Point_<float>((extra[0]+extra[2])/2, extra[1]+extra[3]));
        vector<Point> hu;
        convexHull(a, hu);
        hullz.push_back(hu);
        drawContours(charCandidates,hullz, int(hullz.size())-1, Scalar(255,255,255),-1);
    }
    
    /*
    extra = 0
    if len(hulls) >= 6 and len(hulls) < 8:  # 31 4 (제외)
        # print(hulls)
        x = [a[0] for a in hulls]
        y = [a[1] for a in hulls]
        x.sort()
        b = []
        for i in range(len(x)-1):
            b.append(x[i+1] - x[i])
        a = max(b)
        k = 0
        for i, v in enumerate(b):
            if v == a:
                print(i, i+1, "사이")
                k = i
                break
        # print(x, b)

        if (a < sum(b)/len(b)*1.7 and len(hulls) == 7):  # 21 4 ()
            print('21 4')
        elif(a > sum(b)/len(b)*1.7 and len(hulls) == 6): # 2x 4
            print('2x 4')
            # print(hulls)
            j = [a[2] for a in hulls]
            l = [a[3] for a in hulls]
            extra = (x[i]+(sum(b)-a)//(len(b)-1), sum(y)//len(y), sum(j)//len(j), sum(l)//len(l))
            #(136, 10, 33, 53)
        elif(a > sum(b)/len(b)*1.7 and len(hulls) == 7): # 3x 4 ()
            print('3x 4')
            j = [a[2] for a in hulls]
            l = [a[3] for a in hulls]
            extra = (x[i]+(sum(b)-a)//(len(b)-1), sum(y)//len(y), sum(j)//len(j), sum(l)//len(l))
        else:
            print('?')

    self.extra = extra
     */
    //bitwise_and(threshc, threshc, threshc, mask=charCandidates);
    bitwise_and(threshc, charCandidates, threshc);
    PlateDetector::hull = hullz;
    return threshc;
}

vector<Mat> PlateDetector::scissor(Mat threshc){
    vector<Mat> chars;
    vector<Rect> box;
    
    for (int i = 0 ; i < hull.size(); i++){
        Rect rect = boundingRect(hull[i]);
//        float boxX = rect.x;
//        float boxY = rect.y;
//        float boxW = rect.width;
//        float boxH = rect.height;
//        float dX = (40 > 40 - boxW) ? 40 - boxW : 40;
//        boxX -= dX;
//        boxW += (dX * 2);
//        rect.x = boxX;
//        rect.width = boxW;
//        vector<float> temp = {boxX,boxY,boxW+boxX ,boxH+boxY};
        box.push_back(rect);
    }
    
    sort(box.begin(), box.end(),compareRectX);
    //(startX, startY, endX, endY)
    //chars.append(lp.thresh[startY:endY, startX:endX])
     for (int i = 0 ; i < box.size(); i++){
         //startY:endY, startX:endX
//         chars.push_back(threshc(cv::Range(box[i][1],box[i][3]),cv::Range(box[i][0], box[i][2])));
         chars.push_back(threshc(box[i]));
//         cout << box[i]  << endl;
//         chars.push_back(threshc);
     }
    /*
     cnts = cv2.findContours(lp.candidates.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
           cnts = imutils.grab_contours(cnts)
           boxes = []
           chars = []

           # loop over the contours
           for c in cnts:
               # compute the bounding box for the contour while maintaining the minimum width
               (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
               dX = min(self.minCharW, self.minCharW - boxW) // 2
               boxX -= dX
               boxW += (dX * 2)

               # update the list of bounding boxes
               boxes.append((boxX, boxY, boxX + boxW, boxY + boxH))
           if self.extra != 0:
               (boxX, boxY, boxW, boxH) = self.extra
               dX = min(self.minCharW, self.minCharW - boxW) // 2
               boxX -= dX
               boxW += (dX * 2)
               boxes.append((boxX, boxY, boxX + boxW, boxY + boxH))
               # print((boxX, boxY, boxX + boxW, boxY + boxH))
           # sort the bounding boxes from left to right
           boxes = sorted(boxes, key=lambda b: b[0])
           print(boxes)
           # loop over the started bounding boxes
           for (startX, startY, endX, endY) in boxes:
               # extract the ROI from the thresholded license plate and update the characters
               # list
               chars.append(lp.thresh[startY:endY, startX:endX])

           # return the list of characters
           return chars
     */
    
    return chars;
}


vector<Mat> PlateDetector::detect_plate(Mat image){
    Mat rectKernel = getStructuringElement(MORPH_RECT,Size(13,5));
//    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    Mat squareKernel = getStructuringElement(MORPH_RECT,Size(3,3));
//    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    Mat squareKernel2 = getStructuringElement(MORPH_RECT,Size(5,5));
//    squareKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    Mat rectKernel2 = getStructuringElement(MORPH_RECT,Size(20,2));
//    rectKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
    vector<Mat> regions;
//    regions = []
//
    Mat gray;
    cvtColor(image, gray, COLOR_RGB2GRAY);
    Mat tophat;
    morphologyEx(gray, tophat, MORPH_TOPHAT, rectKernel);
    Mat blackhat;
    morphologyEx(gray, blackhat, MORPH_BLACKHAT, rectKernel);
    Mat add1;
    add(gray, tophat,add1);
    Mat subtract1;
    subtract(add1, blackhat, subtract1);
    
    
    
//    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
//           tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
//           blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
//
//           add = cv2.add(gray, tophat)
//           subtract = cv2.subtract(add, blackhat)
//
//           blackhat = cv2.GaussianBlur(blackhat, (5, 5), 0)
//           blackhat2 = cv2.GaussianBlur(subtract, (5, 5), 0)
//
    GaussianBlur(blackhat, blackhat, Size(5,5), 0);
    Mat blackhat2;
    GaussianBlur(subtract1, blackhat2, Size(5,5), 0);
    
    Mat thresh,thresh1, thresh2, thresh3;
    threshold(blackhat, thresh1, 0, 255, THRESH_BINARY + THRESH_OTSU);
    threshold(blackhat2, thresh2, 0, 255, THRESH_BINARY + THRESH_OTSU);
    adaptiveThreshold(blackhat2, thresh3, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 19, 9);
    dilate(thresh3, thresh3, rectKernel);
    dilate(thresh3, thresh3, rectKernel);
    adaptiveThreshold(blackhat2,thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 19, 20);
    subtract(thresh2, thresh, thresh);
    bitwise_and(thresh, thresh3, thresh);
    
    
//           thresh1 = cv2.threshold(
//               blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
//           thresh2 = cv2.threshold(
//               blackhat2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
//
//
//           thresh3 = cv2.adaptiveThreshold(blackhat, 255,
//                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
//                                           cv2.THRESH_BINARY_INV,
//                                           19,
//                                           9)
//
//           cv2.imshow("adaptiveThreshold", thresh3)
//           thresh3 = cv2.dilate(thresh3, None, iterations=2)
//
//
//           thresh = cv2.adaptiveThreshold(blackhat2, 255,
//                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
//                                          cv2.THRESH_BINARY_INV,
//                                          19,
//                                           20)
//           cv2.imshow("adaptiveThreshold2", thresh)
//           thresh = cv2.subtract(thresh2, thresh)
//           cv2.imshow("adaptiveThreshold-1", thresh3)
//           thresh = cv2.bitwise_and(thresh, thresh3)
//
//           cv2.imshow("result", thresh)
//
    vector<vector<Point>> cnts;
    vector<Vec4i> hierarchy;
    Mat dst = Mat::zeros(thresh.rows, thresh.cols, CV_8UC3);

    findContours(thresh, cnts,hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    sort(cnts.begin(), cnts.end(), compareContourAreas);
    if (cnts.size() > 0){
    vector<cv::Point> biggestContour = cnts[cnts.size()-1];
    vector<cv::Point> smallestContour = cnts[0];
    }

//
//           cnts = cv2.findContours(
//               thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
//
//           cnts = imutils.grab_contours(cnts)
////
//
//     vector<cv::Point> regions;
    Rect rect;
    int idx = 0;
    for(int i=0; i< cnts.size();i++) {
        if(0 < cnts[i].size()){
            idx = i;
            rect = boundingRect(cnts[idx]);
            Point pt1, pt2;
            pt1.x = rect.x;
            pt1.y = rect.y;
            pt2.x = rect.x + rect.width;
            pt2.y = rect.y + rect.height;
            float w =abs(pt1.x - pt2.x);
            float h = abs(pt1.y - pt2.y);
            float aspectRatio = w / h;
            RotatedRect rect = minAreaRect(cnts[idx]);
            Mat box;
            boxPoints(rect, box);
            // rect = cv2.minAreaRect(c)
            // box = np.int0(cv2.cv.BoxPoints(rect)) if imutils.is_cv2() else cv2.boxPoints(rect)
            if (w  > 400 && aspectRatio > 2 && aspectRatio < 5){
//         Scalar color( rand()&255, rand()&255, rand()&255 );
                regions.push_back(box);
             /////   rectangle(image, pt1, pt2, CV_RGB(0,0,255), 4);
            }
//      }
//         drawContours( dst, cnts, idx, color, FILLED, 8,hierarchy);
        }}
//           for c in cnts:
//               # grab the bounding box associated with the contour and compute the area and
//               # aspect ratio
//               (w, h) = cv2.boundingRect(c)[2:]
//               aspectRatio = w / float(h)
//
//               # compute the rotated bounding box of the region
//               rect = cv2.minAreaRect(c)
//               box = np.int0(cv2.cv.BoxPoints(
//                   rect)) if imutils.is_cv2() else cv2.boxPoints(rect)
//               # regions.append(box)
//
//               # ensure the aspect ratio, width, and height of the bounding box fall within
//               # tolerable limits, then update the list of license plate regions
//               # if (aspectRatio > 2 and aspectRatio < 6) and h > self.minPlateH and w > self.minPlateW:
//               if w * h > 300:
//                       regions.append(box)
//
//
//
//           return regions
    
    return regions;
}



