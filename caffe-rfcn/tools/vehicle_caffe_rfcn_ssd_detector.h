#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <caffe/caffe.hpp>
#include <cassert>
#include <ctime>
#include <ctype.h>
#include <fcntl.h>
#include <fstream>
#include <gflags/gflags.h>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <vector>

using namespace caffe;
using namespace cv;


typedef struct Detection {
    int id = -1;
    bool deleted = false;
	cv::Rect box;
    float confidence = 0;
    float col_ratio = 1.0;
    float row_ratio = 1.0;
    void Rescale(float scale) {
	        box.x = box.x / scale;
	        box.y = box.y / scale;
	        box.width = box.width / scale;
	        box.height = box.height / scale;
	}
} Detection;

typedef struct {
    int gpu_id = 0;
    bool use_gpu = true;
    int batch_size = 1;
    float scale=255.0;
    string deploy_file="deploy.txt";
    string model_file="model.dat";
    int max_width=711;
    int resize_height=400;
    float base_threshold=0.5;
    float S2_threshold=0.8;
    float nms_threshold=0.35;
} VehicleCaffeRFCN_SSD_DetectorConfig;

class VehicleCaffeRFCN_SSD_Detector{
    
    typedef struct {
        Mat image;
        float cols_ratio = 0.0;
        float rows_ratio = 0.0;
    } MAT;

public:
    VehicleCaffeRFCN_SSD_Detector(const VehicleCaffeRFCN_SSD_DetectorConfig& config);
    virtual ~VehicleCaffeRFCN_SSD_Detector();
    virtual void BatchProcess(const vector<Mat>& imgs, vector<vector<Detection> >& detect_results);

private:
    vector<vector<boost::shared_ptr<Blob<float>>>> PredictBatch(const vector<Mat>& imgs, float red, float green, float blue);
    void EachFullfil(const vector<cv::Mat> &img,
                 const vector<boost::shared_ptr<Blob<float>>>& outputs,
                 vector<vector<Detection>>& detect_results, float, float, vector<float>&, vector<float>&);
    void Fullfil(const vector<cv::Mat> &img,
                 const vector<vector<boost::shared_ptr<Blob<float>>>>& outputs,
                 vector<vector<Detection>>& detect_results, vector<float>&, vector<float>&);

    void vis(const vector<Mat>& batch_imgs, vector<vector<Detection>>& detections);

private:
    boost::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    int batch_size_;
    float scale_;
    bool useGPU_;
    int max_width;
    int resize_height;
    float base_threshold;
    float S2_threshold;
    float nms_threshold;
};
