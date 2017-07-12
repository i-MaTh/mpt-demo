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
#include <yaml-cpp/yaml.h>

using namespace std;
using namespace cv;
using namespace caffe;

template <typename T> void parse_vector(YAML::Node node, vector<T> &vec);

struct Bbox {
  float confidence;
  int cls;
  Rect rect;
  bool deleted;
};
bool mycmp(struct Bbox b1, struct Bbox b2) {
  return b1.confidence > b2.confidence;
}

class RFCN_detector {
public:
  RFCN_detector(const string &model_file, const string &trained_file,
               const bool use_GPU, const int batch_size, const int device_id,
               YAML::Node node);

  vector<boost::shared_ptr<Blob<float>>> forward(vector<Mat> imgs);
  void nms(vector<struct Bbox> &p, float threshold);
  void get_detection(vector<boost::shared_ptr<Blob<float>>> &outputs, vector<struct Bbox> &bbs, float rfcn_thres, float nms_thres, float resize_height, float resize_width, float enlarge_ratio);
  void get_input_size(int &batch_size, int &num_channels, int &height,
                      int &width);
  void set_input_geometry(int width, int height);

private:
  boost::shared_ptr<Net<float>> net_;
  int batch_size_;
  int num_channels_;
  cv::Size input_geometry_;
  bool useGPU_;

  int sliding_window_stride_;
  int anchor_number_;
  vector<float> mean_values_;
  float scale_;
};
