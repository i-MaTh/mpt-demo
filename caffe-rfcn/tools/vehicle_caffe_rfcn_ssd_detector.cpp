#include "vehicle_caffe_rfcn_ssd_detector.h"
#include <algorithm>
#define SHOW_VIS

using namespace std;
using namespace cv;


VehicleCaffeRFCN_SSD_Detector::VehicleCaffeRFCN_SSD_Detector(const VehicleCaffeRFCN_SSD_DetectorConfig& config) {
    useGPU_ = config.use_gpu;

    Caffe::set_mode(Caffe::CPU);
    if (useGPU_) {
		Caffe::SetDevice(config.gpu_id);
        Caffe::set_mode(Caffe::GPU);
    } else {
        Caffe::set_mode(Caffe::CPU);
    }

    batch_size_ = config.batch_size;
	net_.reset(new Net<float>(config.deploy_file, TEST));
	net_->CopyTrainedLayersFrom(config.model_file);

    Blob<float> *input_layer = net_->input_blobs()[0];
    Blob<float>* input_layer_info = net_->input_blobs()[1];
    num_channels_ = input_layer->channels();
    max_width = config.max_width;
    resize_height = config.resize_height;
    input_geometry_ = cv::Size(max_width, resize_height);
    base_threshold = config.base_threshold;
    S2_threshold = config.S2_threshold;
    nms_threshold = config.nms_threshold;
    scale_ = config.scale;

    input_layer->Reshape(batch_size_, num_channels_, input_geometry_.height, input_geometry_.width);
    input_layer_info->Reshape(1,2,1,1);

    float* input_data_info = input_layer_info->mutable_cpu_data();
    input_data_info[0] = resize_height;
    input_data_info[1] = max_width;
    net_->Reshape();
}

VehicleCaffeRFCN_SSD_Detector::~VehicleCaffeRFCN_SSD_Detector() {
}


bool detectionCmp(Detection b1, Detection b2) {
    return b1.confidence > b2.confidence;
}

void detectionNMS_hyy(vector<struct Detection> &p, float threshold) {								
  sort(p.begin(), p.end(), detectionCmp);
  for (int i = 0; i < p.size(); i++) {
      if (p[i].deleted)
        continue;
      for (int j = i + 1; j < p.size(); j++) {
	        if (!p[j].deleted) {
			        cv::Rect intersect = p[i].box & p[j].box;
			        float iou = intersect.area() * 1.0 /
			                    (p[i].box.area() + p[j].box.area() - intersect.area());
			        if (iou > threshold) {
					          p[j].deleted = true;
					        }
			      }   
	      }   
    }
}

void VehicleCaffeRFCN_SSD_Detector::Fullfil(const vector<cv::Mat>& images,
                                   const vector<vector<boost::shared_ptr<Blob<float>>>>& outputs,
                                   vector<vector<Detection> >& detect_results, vector<float>& enlarge_ratio_w_vec, vector<float>& enlarge_ratio_h_vec) {
    for (int i=0;i< detect_results.size();++i)
        detect_results[i].clear();

    vector<vector<Detection>> base_detection_results;
    EachFullfil(images, outputs[0], base_detection_results, base_threshold, nms_threshold, enlarge_ratio_w_vec, enlarge_ratio_h_vec);
    vector<vector<Detection>> S2_detection_results;
    EachFullfil(images, outputs[1], S2_detection_results, S2_threshold, nms_threshold, enlarge_ratio_w_vec, enlarge_ratio_h_vec);

    assert(base_detection_results.size()== S2_detection_results.size());
    int imgs_num= base_detection_results.size();
    for (int i=0;i< imgs_num;++i){
        vector<Detection> det;
        for (int j=0;j< base_detection_results[i].size();++j)
            det.push_back(base_detection_results[i][j]);
        for (int j=0;j< S2_detection_results[i].size();++j)
            det.push_back(S2_detection_results[i][j]);
        detectionNMS_hyy(det, nms_threshold);
        vector<Detection> after_nms_det;
        for(int k = 0; k < det.size(); k++) {                                                           
            if(!det[k].deleted)  after_nms_det.push_back(det[k]);
        }   
        detect_results.push_back(after_nms_det);
    }
}

void VehicleCaffeRFCN_SSD_Detector::EachFullfil(const vector<cv::Mat>& images,
                                   const vector<boost::shared_ptr<Blob<float>>>& outputs,
                                   vector<vector<Detection> >& detect_results, float score_thres, float nms_thres, 
                                   vector<float>& enlarge_ratio_w_vec, vector<float>& enlarge_ratio_h_vec) {

    detect_results.clear();

    const float* rois = outputs[0]->cpu_data();
    const float* cls_prob = outputs[1]->cpu_data();
    const float* pred_bbox = outputs[2]->cpu_data();
    float resize_width = images[0].cols;
    vector<vector<struct Detection> > batch_bbs;
    for(int i=0;i< images.size();++i){
        vector<struct Detection> bbs;
        batch_bbs.push_back(bbs);
    }
    int box_num= outputs[0]->num();
    for(int i=0;i<box_num;++i){
        int max_cls = -1;
        float max_conf=-1;
        for(int j = 1; j < outputs[1]->channels(); j++) {
            if (max_conf < cls_prob[i*5+j]) {
                max_conf = cls_prob[i*5+j];
                max_cls = j;
            }
        }
        float score= max_conf;
        if (score < score_thres) continue;
        int batch_idx= rois[5*i+0];
        float x1=rois[5*i+1];   
        float y1=rois[5*i+2];   
        float x2=rois[5*i+3];   
        float y2=rois[5*i+4];   
        float width= x2-x1+1;
        float height= y2-y1+1;
        float cx= x1+ 0.5*width;
        float cy= y1+ 0.5*height;
        float dx= pred_bbox[i*8+4];
        float dy= pred_bbox[i*8+5];
        float dw= pred_bbox[i*8+6];
        float dh= pred_bbox[i*8+7];
        float lx = cx+dx*width - 0.5* exp(dw)*width;
        float ly = cy+dy*height - 0.5* exp(dh)*height;
        float rx = cx+dx*width + 0.5* exp(dw)*width;
        float ry = cy+dy*height + 0.5* exp(dh)*height;
        float enlarge_ratio_h= enlarge_ratio_h_vec[batch_idx];
        float enlarge_ratio_w= enlarge_ratio_w_vec[batch_idx];
        int xmin = std::max(0.0f, lx)*enlarge_ratio_w;
        int xmax = std::min((float)resize_width, rx)*enlarge_ratio_w;
        int ymin = std::max(0.0f, ly)*enlarge_ratio_h;                                                                                                                                                         
        int ymax = std::min((float)resize_height, ry)*enlarge_ratio_h;
        if ((ymax - ymin) * (xmax - xmin) < 3*3) continue;
        struct Detection bb;
        bb.box= Rect(xmin, ymin, xmax-xmin+1, ymax-ymin+1);
        bb.confidence = score;
        bb.deleted= false;
        bb.id= max_cls;
        batch_bbs[batch_idx].push_back(bb);
    }
    for (int i=0;i< images.size();++i){
            if(batch_bbs[i].size() != 0) detectionNMS_hyy(batch_bbs[i], nms_thres);
    }

    for (int batch_idx=0;batch_idx< images.size();++batch_idx){
        vector<Detection> bbs= batch_bbs[batch_idx]; 
        vector<Detection> det;
        for(int i = 0; i < bbs.size(); i++) {                                                           
            if(!bbs[i].deleted)  det.push_back(bbs[i]);
        }   
        detect_results.push_back(det);
    }
}

void VehicleCaffeRFCN_SSD_Detector::vis(const vector<cv::Mat>& imgs, vector<vector<Detection>>& detect_results) {

    assert(imgs.size() == detect_results.size());
#ifdef SHOW_VIS
    vector<Scalar> color;
    color.push_back(Scalar(255,0,0));
    color.push_back(Scalar(0,255,0));
    color.push_back(Scalar(0,0,255));
    color.push_back(Scalar(255,255,0));
    color.push_back(Scalar(0,255,255));
    color.push_back(Scalar(255,0,255));
    vector<string> tags;
    tags.push_back("bg");
    tags.push_back("person");
    tags.push_back("bicycle");
    tags.push_back("tricycle");
    tags.push_back("car");

    for (int i = 0; i < imgs.size(); i++) {
        vector<Detection>& detections = detect_results[i];
        Mat image = imgs[i];
        for (int j = 0; j < detections.size(); j++) {
            int xmin= max(0, detections[j].box.x);
            int ymin= max(0, detections[j].box.y);
            int xmax= min(detections[j].box.x+detections[j].box.width, image.cols);
            int ymax= min(detections[j].box.y+detections[j].box.height, image.rows);
            detections[j].box.x= max(0, detections[j].box.x);
            detections[j].box.x= max(0, detections[j].box.x);
            detections[j].box.width= xmax- xmin;
            detections[j].box.height= ymax- ymin;
            char score_text[100];
            sprintf(score_text, "%.3f", detections[j].confidence);
            rectangle(image, detections[j].box, color[detections[j].id]);
            putText(image, tags[detections[j].id] + "_" + string(score_text), Point(detections[j].box.x, detections[j].box.y), 
                CV_FONT_HERSHEY_COMPLEX, 0.5, color[detections[j].id]);
        }
        imshow("debug", image);
        int key=cv::waitKey();
        if (key==-1 || key==27) exit(0);
    }
#endif
}

void add_black_edge(vector<cv::Mat>& imgs_raw, vector<cv::Mat>& imgs){
    imgs.clear();
    int max_w=0;
    for (int i=0;i< imgs_raw.size();++i){
        if (imgs_raw[i].cols > max_w) max_w = imgs_raw[i].cols;
    }

    for(int i=0;i< imgs_raw.size();++i){
        cv::Mat image= imgs_raw[i];   
        if (image.cols < max_w){
            Mat new_image = Mat::zeros(image.rows, max_w, CV_8UC3);
            image.copyTo(new_image.colRange(0,image.cols));
            imgs.push_back(new_image);
        }else{
            imgs.push_back(image);
        }
    }
}

void VehicleCaffeRFCN_SSD_Detector::BatchProcess(const vector<Mat> &imgs, vector<vector<Detection>> &detect_results) {

	detect_results.clear();
	vector<vector<Detection>> detections;
	vector<cv::Mat> toPredict_raw;
	vector<cv::Mat> toPredict;
	vector<float> enlarge_ratio_w_vec;
	vector<float> enlarge_ratio_h_vec;
	vector<cv::Mat> batch_imgs;

	for (int i = 0; i < imgs.size(); ++i) {
		cv::Mat image = imgs[i];
		int resize_width = (int)image.cols * resize_height / image.rows;
		if (resize_width > max_width) resize_width = max_width;

		enlarge_ratio_w_vec.push_back((float) image.cols / (float) resize_width);
		enlarge_ratio_h_vec.push_back((float) image.rows / (float) resize_height);
		batch_imgs.push_back(image.clone());

		resize(image, image, Size(resize_width, resize_height));
		toPredict_raw.push_back(image);
		if (toPredict_raw.size() == batch_size_) {
			add_black_edge(toPredict_raw, toPredict);

			vector<vector<boost::shared_ptr<Blob<float>>>> outputs = PredictBatch(toPredict, 102.98, 115.95, 122.77);
			Fullfil(toPredict, outputs, detections, enlarge_ratio_w_vec, enlarge_ratio_h_vec);
			vis(batch_imgs, detections);
			detect_results.insert(detect_results.end(), detections.begin(), detections.end());
			toPredict_raw.clear();
			toPredict.clear();
			batch_imgs.clear();
			enlarge_ratio_w_vec.clear();
			enlarge_ratio_h_vec.clear();
		}
	}

	if (toPredict.size() > 0) {
		add_black_edge(toPredict_raw, toPredict);
		vector<vector<boost::shared_ptr<Blob<float>>>> outputs = PredictBatch(toPredict, 102.98, 115.95, 122.77);
		Fullfil(toPredict, outputs, detections, enlarge_ratio_w_vec, enlarge_ratio_h_vec);
		vis(batch_imgs, detections);
		detect_results.insert(detect_results.end(), detections.begin(), detections.end());
	}
}

vector<std::vector<boost::shared_ptr<Blob<float>>>> VehicleCaffeRFCN_SSD_Detector::PredictBatch(const vector<Mat> &imgs, float red, float green, float blue) {
    Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(imgs.size(), num_channels_, imgs[0].rows, imgs[0].cols);
    float *input_data = input_layer->mutable_cpu_data();

    Blob<float>* input_layer_info = net_->input_blobs()[1];
    float* input_data_info = input_layer_info->mutable_cpu_data();
    input_data_info[0] = imgs[0].rows;
    input_data_info[1] = imgs[0].cols;

    int dim = imgs[0].cols*imgs[0].rows;
    for (int i = 0; i < imgs.size(); i++) {
        Mat sample = imgs[i];
        sample.convertTo(sample, CV_32FC3);
        sample = (sample - Scalar(102.9801, 115.9465, 122.7717)) / (float)scale_;

        vector<cv::Mat> input_channels;
        cv::split(sample, input_channels);
        float *input_imgdata = NULL;
        for (int j = 0; j < num_channels_; j++) {
            input_imgdata = (float *)input_channels[j].data;
            memcpy(input_data, input_imgdata, sizeof(float) * dim);
            input_data += dim;
        }
    }

    net_->ForwardPrefilled();
    if(useGPU_) {
        cudaDeviceSynchronize();
    }

    /* Copy the output layer to a std::vector */
    vector<vector<boost::shared_ptr<Blob<float>>>> outputs;
    vector<boost::shared_ptr<Blob<float>>> output;
    output.push_back(net_->blob_by_name("rois"));
    output.push_back(net_->blob_by_name("cls_prob"));
    output.push_back(net_->blob_by_name("bbox_pred"));
    vector<boost::shared_ptr<Blob<float>>> S2_output;
    S2_output.push_back(net_->blob_by_name("S2_rois"));
    S2_output.push_back(net_->blob_by_name("S2_cls_prob"));
    S2_output.push_back(net_->blob_by_name("S2_bbox_pred"));

    outputs.push_back(output);
    outputs.push_back(S2_output);
    return outputs;
}

int main(int argc, char** argv) {
	cout<<".. img_list batch_size"<<endl;
	string img_list=argv[1];
	int batch_size= atoi(argv[2]);

	VehicleCaffeRFCN_SSD_DetectorConfig config;
	config.batch_size= batch_size;
	VehicleCaffeRFCN_SSD_Detector* rfcn_model=new VehicleCaffeRFCN_SSD_Detector(config);

    ifstream fin(img_list);
    string img_path;
    vector<Mat> images;
    while (fin >> img_path){
        Mat image = imread(img_path);
        images.push_back(image);
        if (images.size()== batch_size){
            vector<vector<Detection>> detections;
			rfcn_model->BatchProcess(images, detections);
            images.clear();
        }
    }
    fin.close();
    delete rfcn_model;
    return 0;
}
