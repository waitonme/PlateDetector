//
//  PlateDetectorBridge.m
//  PlateDetector
//
//  Created by near on 2020/05/12.
//  Copyright © 2020 near. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <Foundation/Foundation.h>
#import "PlateDetectorBridge.h"
#include "PlateDetector.hpp"


#import "koreanModel.h"
#import "charModel.h"
#import "digitModel.h"
#import "SimpleMnist.h"


@implementation PlateDetectorBridge


- (UIImage *) detectPlateIn: (UIImage *) image {

    // convert uiimage to mat
    cv::Mat opencvImage;
    UIImageToMat(image, opencvImage, true);
    
    // convert colorspace to the one expected by the lane detector algorithm (RGB)
    cv::Mat convertedColorSpaceImage;
    cv::cvtColor(opencvImage, convertedColorSpaceImage, COLOR_RGBA2RGB);
    
    // Run lane detection
    PlateDetector plateDetector;
//    vector<vector<cv::Mat>> chars = plateDetector.detect(convertedColorSpaceImage);
    vector<Mat> T = plateDetector.detect(convertedColorSpaceImage);;
    if (T.size()>0)
        return MatToUIImage(T[0]);
    
//       NSArray *shape = @[@1, @45];
//       MLMultiArrayDataType dataType = MLMultiArrayDataTypeDouble;
//       NSError * error = nil;
//       MLMultiArray *theMultiArray =  [[MLMultiArray alloc] initWithShape:(NSArray*)shape dataType:(MLMultiArrayDataType)dataType error:&error] ;
//
    
    
    
    
  
//
//    MLModel *cm = [[[charModel alloc] init] model];
//    MLModel *dm = [[[digitModel alloc] init] model];
//
//
//
//
//    VNCoreMLModel *cmm = [VNCoreMLModel modelForMLModel: cm error:nil];
//    VNCoreMLModel *dmm = [VNCoreMLModel modelForMLModel: dm error:nil];
//
//    VNCoreMLRequest *cmm_req = [[VNCoreMLRequest alloc] initWithModel: cmm completionHandler: (VNRequestCompletionHandler) ^(VNRequest *request, NSError *error){
//        dispatch_async(dispatch_get_main_queue(), ^{
//            unsigned long resultsCount = request.results.count;
//            NSArray *results = [request.results copy];
//            VNClassificationObservation *topResult = ((VNClassificationObservation *)(results[0]));
//            float percent = topResult.confidence * 100;
//            NSString *VNCoreMLRequestresultLabel = [NSString stringWithFormat: @"Confidence: %.f%@ %@", percent,@"%", topResult.identifier];
//            if (resultsCount > 0)
//            NSLog(@"%@\n", VNCoreMLRequestresultLabel);
//            if (error)
//                NSError(*error);
//        });
//    }];
//
//    VNCoreMLRequest *dmm_req = [[VNCoreMLRequest alloc] initWithModel: dmm completionHandler: (VNRequestCompletionHandler) ^(VNRequest *request, NSError *error){
//           dispatch_async(dispatch_get_main_queue(), ^{
//               unsigned long resultsCount = request.results.count;
//                          NSArray *results = [request.results copy];
//                          VNClassificationObservation *topResult = ((VNClassificationObservation *)(results[0]));
//                          float percent = topResult.confidence * 100;
//               NSString *VNCoreMLRequestresultLabel = [NSString stringWithFormat: @"Confidence: %.f%@ %@", percent,@"%", topResult.identifier];
//               if (resultsCount > 0)
//                   NSLog(@"%lu\n", resultsCount);
//
//           });
//    }];
//
//
//    NSDictionary *options = [[NSDictionary alloc] init];
//    NSArray *cmm_reqArray = @[cmm_req];
//    NSArray *dmm_reqArray = @[dmm_req];
//    CIImage *a = [[CIImage alloc] initWithImage: image ];
    
    
//    VNImageRequestHandler *cmm_handler = [[VNImageRequestHandler alloc] initWithCIImage:a options:options];
//    dispatch_async(dispatch_get_main_queue(), ^{
//        [cmm_handler performRequests:cmm_reqArray error:nil];
//    });
//
//    VNImageRequestHandler *dmm_handler = [[VNImageRequestHandler alloc] initWithCIImage:a options:options];
//    dispatch_async(dispatch_get_main_queue(), ^{
//        [dmm_handler performRequests:dmm_reqArray error:nil];
//    });

    
//    for (int i = 0 ; i < chars.size(); i++){
//         for (int j = 0 ; j < chars[i].size(); j++){
//
//        if ((chars[i].size() == 8 && j == 3) || (chars[i].size() == 7 && j == 2)){
//
//                //    private var inputArray: MLMultiArray!
//                //      private let tensorShape: [NSNumber] = [32, 32, 3]
//                //
//                //      // Init CoreML Array
//                //      public func predict(pixel: CVPixelBuffer?) -> String? {
//                //          inputArray = try? MLMultiArray(shape: tensorShape, dataType: .float32)
//                //          guard let pixelBuffer: CVPixelBuffer = pixel else {
//                //              return nil
//                //          }
//                //          // CoreML Model Initialization and Predict
//                //          let model = hand_written_korean_classification()
//                //          guard let output: hand_written_korean_classificationOutput = try? model.prediction(image: pixelBuffer) else {
//                //              return nil
//                //          }
//                //          return output.classLabel
//                //      }
//            Mat b;
//            cv::cvtColor(chars[i][j], b, COLOR_GRAY2RGB);
//            cv::resize(b, b, cv::Size(32,32));
////            cout << b.size() << "b size \n" ;
//
//            CIImage *a = [[CIImage alloc] initWithImage: MatToUIImage(b) ];
//            CIContext *mcontext = [CIContext contextWithOptions:nil];
//            CGImageRef myImage = [mcontext createCGImage:a fromRect:CGRectMake(0, 0, 32, 32)];
//
//
//            CGSize frameSize = CGSizeMake(32, 32);
//
//            CVPixelBufferRef cvBuf;
//            CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, frameSize.width, frameSize.height, kCVPixelFormatType_32BGRA, nil, &cvBuf);
//            if (status != kCVReturnSuccess) {
//                 return NULL;
//             }
//
//            CVPixelBufferLockBaseAddress(cvBuf, 0);
//            void *data = CVPixelBufferGetBaseAddress(cvBuf);
//
//            CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
//            CGContextRef context = CGBitmapContextCreate(data, frameSize.width, frameSize.height, 8, CVPixelBufferGetBytesPerRow(cvBuf), rgbColorSpace, (CGBitmapInfo) kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
//            CGContextDrawImage(context, CGRectMake(0, 0, 32, 32), myImage);
//
//            CGColorSpaceRelease(rgbColorSpace);
//            CGContextRelease(context);
//            CVPixelBufferUnlockBaseAddress(cvBuf, 0);
//
//
//            NSError *error = nil;
//
//
//            koreanModel *km = [[koreanModel alloc] init];
//            koreanModelOutput *kmo =  [(koreanModel *)km predictionFromImage:cvBuf error:&error];
//
//            NSLog(@"SVM Model output = %@ ", kmo.classLabel);
//
//
//            CIImage *ciImage = [CIImage imageWithCVPixelBuffer:cvBuf];
//
//            CIContext *temporaryContext = [CIContext contextWithOptions:nil];
//            CGImageRef videoImage = [temporaryContext
//                               createCGImage:ciImage
//                               fromRect:CGRectMake(0, 0,
//                                      CVPixelBufferGetWidth(cvBuf),
//                                      CVPixelBufferGetHeight(cvBuf))];
//
//            UIImage *uiImage = [UIImage imageWithCGImage:videoImage];
//            CGImageRelease(videoImage);
//
//
//            return uiImage;
//            //MatToUIImage(chars[i][j]);
//
//        }
//
//                              else{
//                                      //    private var inputArray: MLMultiArray!
//                                                  //      private let tensorShape: [NSNumber] = [32, 32, 3]
//                                                  //
//                                                  //      // Init CoreML Array
//                                                  //      public func predict(pixel: CVPixelBuffer?) -> String? {
//                                                  //          inputArray = try? MLMultiArray(shape: tensorShape, dataType: .float32)
//                                                  //          guard let pixelBuffer: CVPixelBuffer = pixel else {
//                                                  //              return nil
//                                                  //          }
//                                                  //          // CoreML Model Initialization and Predict
//                                                  //          let model = hand_written_korean_classification()
//                                                  //          guard let output: hand_written_korean_classificationOutput = try? model.prediction(image: pixelBuffer) else {
//                                                  //              return nil
//                                                  //          }
//                                                  //          return output.classLabel
//                                                  //      }
////                                              Mat b;
//////                                              cv::cvtColor(chars[i][j], b, GRAY);
////                                              cv::resize(chars[i][j], b, cv::Size(28,28));
//////                                            Mat tempImage;
////                                                normalize(b,b,0,1,NORM_MINMAX);
//////                                                b.convertTo(b, CV_32FC3, 1.f/255);
////
////                                  //            cout << b.size() << "b size \n" ;
////
////                                              CIImage *a = [[CIImage alloc] initWithImage: MatToUIImage(b) ];
////                                              CIContext *mcontext = [CIContext contextWithOptions:nil];
////                                              CGImageRef myImage = [mcontext createCGImage:a fromRect:CGRectMake(0, 0, 28, 28)];
////
////
////                                              CGSize frameSize = CGSizeMake(28, 28);
////
////                                              CVPixelBufferRef cvBuf;
////                                              CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, frameSize.width, frameSize.height, kCVPixelFormatType_16Gray, nil, &cvBuf);
////                                              if (status != kCVReturnSuccess) {
////                                                   return NULL;
////                                               }
////
////                                              CVPixelBufferLockBaseAddress(cvBuf, 0);
////                                              void *data = CVPixelBufferGetBaseAddress(cvBuf);
////
////                                              CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
////                                              CGContextRef context = CGBitmapContextCreate(data, frameSize.width, frameSize.height, 8, CVPixelBufferGetBytesPerRow(cvBuf), rgbColorSpace, (CGBitmapInfo) kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
////                                              CGContextDrawImage(context, CGRectMake(0, 0, 28, 28), myImage);
////
////                                              CGColorSpaceRelease(rgbColorSpace);
////                                              CGContextRelease(context);
////                                              CVPixelBufferUnlockBaseAddress(cvBuf, 0);
////
////
////                                              NSError *error = nil;
////
////
////                                              SimpleMnist *km = [[SimpleMnist alloc] init];
////                                  SimpleMnistOutput *kmo = [(SimpleMnist*)km predictionFromInput:cvBuf error:&error];
//////                                  [(SimpleMnist *)km predictionFromImage:cvBuf error:&error];
////
////                                  NSLog(@"SVM Model output = %@ ", kmo.predictedNumber);
////
////
////                                              CIImage *ciImage = [CIImage imageWithCVPixelBuffer:cvBuf];
////
////                                              CIContext *temporaryContext = [CIContext contextWithOptions:nil];
////                                              CGImageRef videoImage = [temporaryContext
////                                                                 createCGImage:ciImage
////                                                                 fromRect:CGRectMake(0, 0,
////                                                                        CVPixelBufferGetWidth(cvBuf),
////                                                                        CVPixelBufferGetHeight(cvBuf))];
////
////                                              UIImage *uiImage = [UIImage imageWithCGImage:videoImage];
////                                              CGImageRelease(videoImage);
////
////
////                                              return uiImage;
//                                              //MatToUIImage(chars[i][j]);
//
//
//                }
//
//

//        }
//    }
        
    
//       for (int i = 0; i < 45; i++) {
//                  [theMultiArray setObject:[NSNumber numberWithDouble:1.0] atIndexedSubscript:(NSInteger)i];
//       }
       

   
//    # loop over the detected plages
//       for (lpBox, chars) in plates:
//           # restructure lpBox
//           lpBox = np.array(lpBox).reshape((-1, 1, 2)).astype(np.int32)
//
//           # initialize the text containing the recognized characters
//           text = ""
//
//           # loop over each character
//           for (i, char) in enumerate(chars):
//               # preprocess the character and describe it
//               # char = LicensePlateDetector.preprocessChar(char)
//               if char is None:
//                   continue
//               features = desc.describe(char).reshape(1, -1)
//
//               # if this is the first 3 characters, then use the character classifier
//               if (len(chars) == 8 and i == 3) or (len(chars) == 7 and i == 2):
//                   cv2.imshow("Character {}".format(i), char)
//                   prediction = charModel.predict(features)[0]
//
//               # otherwise, use the digit classifier
//               else:
//                   cv2.imshow("Character {}".format(i), char)
//                   prediction = digitModel.predict(features)[0]
//
//               # update the text of recognized characters
//               text += prediction
//
//           # compute the center of the license plate bounding box
//           M = cv2.moments(lpBox)
//           cX = int(M["m10"] / M["m00"])
//           cY = int(M["m01"] / M["m00"])
//
//           cv2.drawContours(image, [lpBox], -1, (0, 255, 0), 2)
//
//           fontpath = "/Users/near/Documents/venv/NanumBarunGothic.ttf"
//           font = ImageFont.truetype(fontpath, 40)
//           img_pil = Image.fromarray(image)
//           draw = ImageDraw.Draw(img_pil)
//           text = unicodedata.normalize('NFC', text)
//
//           draw.text((cX - (cX // 5), cY - 30),  text, font=font, fill=(0, 0, 255, 2))
//           image = np.array(img_pil)
//
//           # draw the license plate region and license plate text on the image
//
//           # cv2.putText(image, text, (cX - (cX // 5), cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
//           #     (0, 0, 255), 2)
//
//       # display the output image
//       cv2.imshow("image", image)
//       cv2.waitKey(0)
//
        return MatToUIImage(convertedColorSpaceImage);
}


@end
