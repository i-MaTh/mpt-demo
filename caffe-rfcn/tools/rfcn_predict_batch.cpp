#include "rfcn_predict_batch.h"
#include <algorithm>

template <typename T> void parse_vector(YAML::Node node, vector<T> &vec) {
  for (std::size_t i = 0; i < node.size(); i++) {
    vec.push_back(node[i].as<T>());
  }
}

string basename(const string path) {
  vector<string> strs;
  boost::split(strs, path, boost::is_any_of("/"));
  return strs[strs.size() - 1];
}

string remove_ent(const string s) {
  vector<string> strs;
  boost::split(strs, s, boost::is_any_of("."));
  string str_new=strs[0];
  for (int i=1;i<strs.size()-1;++i){
	str_new= str_new+"."+ strs[i]; 
  }
  return str_new;
}

RFCN_detector::RFCN_detector(const string &model_file, const string &trained_file,
                           const bool use_GPU, const int devide_id, YAML::Node config) {
  if (use_GPU) {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(devide_id);
    useGPU_ = true;
  } else {
    Caffe::set_mode(Caffe::CPU);
    useGPU_ = false;
  }

  /* Load the network. */
  cout << "loading " << model_file << endl;
  net_.reset(new Net<float>(model_file, TEST));
  cout << "loading " << trained_file << endl;
  net_->CopyTrainedLayersFrom(trained_file);

  Blob<float> *input_layer = net_->input_blobs()[0];
  batch_size_ = input_layer->num();
  num_channels_ = input_layer->channels();
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  sliding_window_stride_ = config["FEAT_STRIDE"].as<int>();
  mean_values_.push_back(102.9801);
  mean_values_.push_back(115.9465);
  mean_values_.push_back(122.7717);
  scale_ = config["TRAIN"]["DATA_AUG"]["NORMALIZER"].as<float>();
}

void RFCN_detector::set_input_geometry(int height, int width) {
  input_geometry_.height = height;
  input_geometry_.width = width;
}
// predict single frame forward function
vector<boost::shared_ptr<Blob<float>>>
RFCN_detector::forward(vector<cv::Mat> imgs) {
  Blob<float> *input_layer = net_->input_blobs()[0];
  Blob<float> *input_layer_info = net_->input_blobs()[1];
  input_geometry_.height = imgs[0].rows;
  input_geometry_.width = imgs[0].cols;
  input_layer->Reshape(batch_size_, num_channels_, input_geometry_.height,
                       input_geometry_.width);
  input_layer_info->Reshape(1,2,1,1);

  int dim = input_geometry_.height * input_geometry_.width;
  float *input_data = input_layer->mutable_cpu_data();
  float* input_data_info = input_layer_info->mutable_cpu_data();
  for (int i = 0; i < imgs.size(); i++) {

    Mat sample = imgs[i];

    if ((sample.rows != input_geometry_.height) ||
        (sample.cols != input_geometry_.width)) {
      resize(sample, sample,
             Size(input_geometry_.width, input_geometry_.height));
    }

    sample.convertTo(sample, CV_32FC3);
    sample = (sample - Scalar(102.9801, 115.9465, 122.7717)) / (float)scale_;

    vector<cv::Mat> input_channels;
    cv::split(sample, input_channels);
    float *input_imgdata = NULL;
    for (int i = 0; i < num_channels_; i++) {
      input_imgdata = (float *)input_channels[i].data;
      memcpy(input_data, input_imgdata, sizeof(float) * dim);
      input_data += dim;
    }
  }
  input_data_info[0] = input_geometry_.height;
  input_data_info[1] = input_geometry_.width;

  net_->Reshape();

  net_->ForwardPrefilled();
  if (useGPU_) {
    cudaDeviceSynchronize();
  }

  vector<boost::shared_ptr<Blob<float>>> outputs;
  boost::shared_ptr<Blob<float>> rois =
      net_->blob_by_name(string("rois"));
  boost::shared_ptr<Blob<float>> cls_prob =
      net_->blob_by_name(string("cls_prob"));
  boost::shared_ptr<Blob<float>> bbox_pred =
      net_->blob_by_name(string("bbox_pred"));
  outputs.push_back(rois);
  outputs.push_back(cls_prob);
  outputs.push_back(bbox_pred);

  return outputs;
}

void RFCN_detector::nms(vector<struct Bbox> &p, float threshold) {

  sort(p.begin(), p.end(), mycmp);
  for (int i = 0; i < p.size(); i++) {
    if (p[i].deleted)
      continue;
    for (int j = i + 1; j < p.size(); j++) {

      if (!p[j].deleted) {
        cv::Rect intersect = p[i].rect & p[j].rect;
        float iou = intersect.area() * 1.0 /
                    (p[i].rect.area() + p[j].rect.area() - intersect.area());
		//if (
			//(iou > 0.3 && p[i].cls==p[j].cls && p[i].cls==1) // person
			//|| (iou > 0.3 && p[i].cls==p[j].cls && p[i].cls == 2) // bicycle
			//|| (iou > 0.3 && p[i].cls==p[j].cls && p[i].cls == 3) // tricycle
			//|| (iou > 0.45 && p[i].cls==p[j].cls && p[i].cls == 4) // car
			//|| (iou > 0.45 && p[i].cls!=p[j].cls)
		   //)
       if (iou > threshold)
	   {
          p[j].deleted = true;
	   }
      }
    }
  }
}

void RFCN_detector::get_input_size(int &batch_size, int &num_channels,
                                  int &height, int &width) {
  batch_size = batch_size_;
  num_channels = num_channels_;
  height = input_geometry_.height;
  width = input_geometry_.width;
}

void RFCN_detector::get_detection(vector<boost::shared_ptr<Blob<float> > > &outputs, vector<vector<struct Bbox>> &dets, float score_thres, float nms_thres, float resize_height, float resize_width, vector<float>& enlarge_ratio_vec, int images_num){
	const float* rois = outputs[0]->cpu_data();
	const float* cls_prob = outputs[1]->cpu_data();
	const float* pred_bbox = outputs[2]->cpu_data();
	vector<vector<struct Bbox> > batch_bbs;
	for(int i=0;i< images_num;++i){
		vector<struct Bbox> bbs;
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
		float enlarge_ratio= enlarge_ratio_vec[batch_idx];
		int xmin = std::max(0.0f, lx)*enlarge_ratio;
		int xmax = std::min(resize_width, rx)*enlarge_ratio;
		int ymin = std::max(0.0f, ly)*enlarge_ratio;
		int ymax = std::min(resize_height, ry)*enlarge_ratio;
		if ((ymax - ymin) * (xmax - xmin) < 3*3) continue;
		struct Bbox bb;
		bb.rect= Rect(xmin, ymin, xmax-xmin+1, ymax-ymin+1);
		bb.confidence = score;
		bb.deleted= false;
		bb.cls= max_cls;
		batch_bbs[batch_idx].push_back(bb);
	}
  for (int i=0;i< images_num;++i){
		if(batch_bbs[i].size() != 0) nms(batch_bbs[i], nms_thres);
  }

  dets.clear();
  for (int batch_idx=0;batch_idx< images_num;++batch_idx){
		vector<Bbox> bbs= batch_bbs[batch_idx]; 
		vector<Bbox> det;
	  for(int i = 0; i < bbs.size(); i++) {                                                           
		  if(!bbs[i].deleted)  det.push_back(bbs[i]);
	  }   
	  dets.push_back(det);
  }
}

void DetectionForVideo(string &model_file, string &trained_file, YAML::Node &config, string &img_file, bool x_show, const string& save_dir) {

  vector<int> sliding_window_width;
  vector<int> sliding_window_height;
  float score_thres;
  float nms_thres;

  parse_vector<int>(config["ANCHOR_GENERATOR"]["SLIDING_WINDOW_WIDTH"],
                    sliding_window_width);
  parse_vector<int>(config["ANCHOR_GENERATOR"]["SLIDING_WINDOW_HEIGHT"],
                    sliding_window_height);
  score_thres = config["TEST"]["THRESH"].as<float>();
  nms_thres = config["TEST"]["NMS"].as<float>();
  bool x_save = true;
  if(save_dir.compare("") == 0)
	  x_save = false;

  std::ifstream infile(img_file.c_str());
  std::string img_path;

  cout << "loading model..." << endl;
  RFCN_detector rfcn_det(model_file, trained_file, true, 0, config);

  const string preds_path=save_dir+"/preds";
  mkdir(save_dir.c_str(), 0755);
  mkdir(preds_path.c_str(), 0755);

  struct timeval start, end;

  int batch_size = 0, num_channels = 0, resize_width = 0, resize_height = 0;
  rfcn_det.get_input_size(batch_size, num_channels, resize_height,
					   resize_width);
  while (true){
		cv::Mat frame;
		int batch_ind=0;
		vector<cv::Mat> images;
		vector<float> enlarge_ratio_vec;
		int max_width=0;
		vector<Mat> batch_frame;
		vector<string> batch_frame_names;
		while(batch_ind < batch_size && infile >> img_path){
			frame = imread(img_path);
			if (frame.empty()) {
			  cout << img_path<<" Wrong Image" << endl;
			  continue;
			}
			batch_frame.push_back(frame);
			batch_frame_names.push_back(img_path);
			int output_width = frame.cols;
			int output_height = frame.rows;

			resize_width = (int)output_width * resize_height / output_height;
			if (max_width < resize_width) max_width=resize_width;
			float enlarge_ratio = output_height * 1.0 / resize_height;
			enlarge_ratio_vec.push_back(enlarge_ratio);

			Mat norm_img;
			cv::resize(frame, norm_img, cv::Size(resize_width, resize_height));
			images.push_back(norm_img);
			batch_ind++;
		}
		for(int i=0;i< images.size();++i){
			cv::Mat image= images[i];	

			Mat new_image;
			if (image.cols < max_width){
				new_image = Mat::zeros(resize_height, max_width, CV_8UC3);
				image.copyTo(new_image.colRange(0,image.cols));
			}else{
				new_image= image;		
			}
			images[i]=new_image;
		}
		
		gettimeofday(&start, NULL);
		vector<boost::shared_ptr<Blob<float>>> outputs = rfcn_det.forward(images);
		gettimeofday(&end, NULL);
		cout<<"forward time: "<< double(end.tv_sec - start.tv_sec)*1000.0 + double(end.tv_usec - start.tv_usec) / 1.0e3 <<endl;                                                                         

		gettimeofday(&start, NULL);
		vector<vector<struct Bbox>> batch_result;
		rfcn_det.get_detection(outputs, batch_result, score_thres, nms_thres, resize_height, resize_width, enlarge_ratio_vec, batch_frame.size());
		gettimeofday(&end, NULL);
		cout<<"get_detection time: "<< double(end.tv_sec - start.tv_sec)*1000.0 + double(end.tv_usec - start.tv_usec) / 1.0e3 <<endl;

	  vector<int> objs_count(4, 0);
	  vector<string> tags;
	  tags.push_back("person");
	  tags.push_back("bicycle");
	  tags.push_back("tricycle");
	  tags.push_back("car");
	  for(int batch_idx=0; batch_idx< batch_result.size();++batch_idx){
		  vector<Bbox> result= batch_result[batch_idx];
		  Mat frame= batch_frame[batch_idx];
		  for (int bbox_id = 0; bbox_id < result.size(); bbox_id++) {
			objs_count[result[bbox_id].cls-1] +=1;
		  }
		cout<<batch_frame_names[batch_idx]<<" ";
		  for(int i=0;i< tags.size();++i){
			int num= objs_count[i]; 
			if (i == tags.size()-1) cout<<tags[i]<<"_"<<num<<endl;
			else   cout<<tags[i]<<"_"<<num<<"_";
		  }
	  }

    if (x_show) {
		  char str_info[100];
		  vector<cv::Scalar> color;
		  color.push_back(cv::Scalar(255,0,0));
		  color.push_back(cv::Scalar(0,255,0));
		  color.push_back(cv::Scalar(0,0,255));
		  color.push_back(cv::Scalar(255,255,0));
		  vector<string> tags;
		  tags.push_back("person");
		  tags.push_back("bicycle");
		  tags.push_back("tricycle");
		  tags.push_back("car");

		  for(int batch_idx=0; batch_idx< batch_result.size();++batch_idx){
			  vector<Bbox> result= batch_result[batch_idx];
			  Mat frame= batch_frame[batch_idx];
			  for (int bbox_id = 0; bbox_id < result.size(); bbox_id++) {
				rectangle(frame, result[bbox_id].rect, color[result[bbox_id].cls-1],
						  2.0);
				sprintf(str_info, "%s_%.2f", tags[result[bbox_id].cls-1].c_str(), result[bbox_id].confidence);
				string prob_info(str_info);
				putText(frame, prob_info, Point(result[bbox_id].rect.x, result[bbox_id].rect.y),
						CV_FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255));
			  }

			 cv::namedWindow("Cam");
			 cv::imshow("Cam", frame);
			 int key=cv::waitKey();
			 if (key==-1 || key== 27) exit(0);
		  }
    }

	//if (x_save)
	//{
		//string img_name= basename(img_path);
		//string pred_path=preds_path+"/"+remove_ent(img_name)+".txt";

		//ofstream outfile(pred_path.c_str());
		//double x,y,width,height,score;
		//int cls;
		//for(int bbox_id = 0; bbox_id < result.size(); bbox_id ++) {
			 //x=result[bbox_id].rect.x;
			 //y=result[bbox_id].rect.y;
			 //width=result[bbox_id].rect.width;
			 //height=result[bbox_id].rect.height;
			 //score=result[bbox_id].confidence;
			 //cls= result[bbox_id].cls;
			 //char s[100];
			 //sprintf(s,"%.3f",score);
			 //outfile<<cls<<" "<<x<<" "<<y<<" "<<width<<" "<<height<<" "<<s<<"\n";
		//}
		//outfile.close();
	//}
  }
}

DEFINE_bool(show, false, "whether show");
DEFINE_string(save_dir,"", "predict result save dir");

int main(int argc, char **argv) {
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("rfcn prediction\n"
                          "Usage:\n"
                          "    prediction [-show] deploy "
                          "caffemodel img_file config_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], " ");
    return 1;
  }

  const bool is_show = FLAGS_show;
  const string save_dir=FLAGS_save_dir;

  string model_file = argv[1];
  string trained_file = argv[2];
  string src = argv[3];
  string config_file = argv[4];

  YAML::Node config = YAML::LoadFile(config_file);

  google::InitGoogleLogging(argv[0]);

  DetectionForVideo(model_file, trained_file, config, src, is_show, save_dir);
}
