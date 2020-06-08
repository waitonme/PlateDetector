//
//  PlateDetectorBridge.h
//  PlateDetector
//
//  Created by near on 2020/05/12.
//  Copyright © 2020 near. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <CoreML/CoreML.h>
#import <Vision/Vision.h>


@interface PlateDetectorBridge : NSObject

- (UIImage *) detectPlateIn: (UIImage *) image;
@end

