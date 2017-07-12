// ------------------------------------------------------------------
// R-FCN
// Written by Yi Li
// ------------------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/box_annotator_ohem_layer.hpp"

using std::max;
using std::min;

//step1: sort loss vector by descending
//step2: keep each image has at most roi_per_img num rois
//when BATCH_SIZE=128 and roi_per_img=128, this layer doesn't work
namespace caffe {
  template <typename Dtype>
  void BoxAnnotatorOHEMLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	//shape: (N,5,1,1), each row: (img_index=0,x1,y1,x2,y2)
    const Dtype* bottom_rois = bottom[0]->cpu_data();
	//bottom_loss shape: (N,1,1,1)
    const Dtype* bottom_loss = bottom[1]->cpu_data();
	//bottom_labels shape: (N,1,1,1)
    const Dtype* bottom_labels = bottom[2]->cpu_data();
	//shape: (N,8,1,1)
    const Dtype* bottom_bbox_loss_weights = bottom[3]->cpu_data();
	//shape: (N,1,1,1)
    Dtype* top_labels = top[0]->mutable_cpu_data();
	//shape: (N,8,1,1)
	//chosen the top min(per_img_roi, BATCH_SIZE) ROIs to compute loss, labels not chosen will be set ignore_label and bbox_loss_weights not chosen will be set 0
    Dtype* top_bbox_loss_weights = top[1]->mutable_cpu_data();
    caffe_set(top[0]->count(), Dtype(ignore_label_), top_labels);
    caffe_set(top[1]->count(), Dtype(0), top_bbox_loss_weights);

    int num_rois_ = bottom[1]->count();

    int num_imgs = -1;
    for (int n = 0; n < num_rois_; n++) {
      for (int s = 0; s < spatial_dim_; s++) {
        num_imgs = bottom_rois[0] > num_imgs ? bottom_rois[0] : num_imgs;
        bottom_rois++;
      }
      bottom_rois += (5-1)*spatial_dim_;
    }
    num_imgs++;
    CHECK_GT(num_imgs, 0)
      << "number of images must be greater than 0 at BoxAnnotatorOHEMLayer";
    bottom_rois = bottom[0]->cpu_data();

    // Find rois with max loss
    vector<int> sorted_idx(num_rois_);
    for (int i = 0; i < num_rois_; i++) {
      sorted_idx[i] = i;
    }
    std::sort(sorted_idx.begin(), sorted_idx.end(),
      [bottom_loss](int i1, int i2) {
        return bottom_loss[i1] > bottom_loss[i2];
    });

    // Generate output labels for scoring and loss_weights for bbox regression
    vector<int> number_left(num_imgs, roi_per_img_);
    for (int i = 0; i < num_rois_; i++) {
      int index = sorted_idx[i];
      int s = index % (width_*height_);
      int n = index / (width_*height_);
      int batch_ind = bottom_rois[n*5*spatial_dim_+s];
      if (number_left[batch_ind] > 0) {
        number_left[batch_ind]--;
        top_labels[index] = bottom_labels[index];
        for (int j = 0; j < bbox_channels_; j++) {
          int bbox_index = (n*bbox_channels_+j)*spatial_dim_+s;
          top_bbox_loss_weights[bbox_index] =
            bottom_bbox_loss_weights[bbox_index];
        }
      }
    }
  }

  template <typename Dtype>
  void BoxAnnotatorOHEMLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    return;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(BoxAnnotatorOHEMLayer);

}  // namespace caffe
