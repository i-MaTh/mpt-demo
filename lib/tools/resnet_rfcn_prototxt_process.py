import sys

sys.path.append('/home/huyangyang/rfcn/caffe/python')
sys.path.append('/home/huyangyang/tools')
from prototxt_process import process_net_pt
import hus_yaml


def assign_list(var_to_assign, lst):
	del var_to_assign[:]
	var_to_assign.extend(lst)


def modify(net, cfg, net_type, state):
	state = state.upper()
	if cfg.ANCHOR_GENERATOR.POLICY == "SCALE_RATIO":
		anchors_num = len(cfg.ANCHOR_GENERATOR.SCALES) * len(cfg.ANCHOR_GENERATOR.RATIOS)
	if cfg.ANCHOR_GENERATOR.POLICY == "SLIDING_WINDOW":
		anchors_num = len(cfg.ANCHOR_GENERATOR.SLIDING_WINDOW_WIDTH)
	num_classes = cfg.NUM_CLASSES
	position_num = cfg.POSITION_NUM
	feat_stride = cfg.FEAT_STRIDE
	if "AGNOSTIC" not in cfg.keys() or cfg.AGNOSTIC is True:
		rfcn_loc_num = 8
	else:
		rfcn_loc_num = 4*num_classes
	for layer in net.layer:
		if layer.name == "input-data":
			layer.python_param.param_str = str(dict(num_classes=num_classes))

		if layer.name == "rpn_cls_score":
			layer.convolution_param.num_output = anchors_num * 2
		if layer.name == "rpn_bbox_pred":
			layer.convolution_param.num_output = anchors_num * 4
		if layer.name == "rpn_cls_prob_reshape":
			layer.reshape_param.shape.dim[1] = anchors_num * 2
		if layer.name == "proposal":
			p = layer.proposal_param
			p.feat_stride = feat_stride
			p.base_size = feat_stride
			p.min_size = cfg.TRAIN.DATA_AUG.MIN_SIZE
			assign_list(p.ratio, cfg.ANCHOR_GENERATOR.RATIOS)
			assign_list(p.scale, cfg.ANCHOR_GENERATOR.SCALES)
			p.pre_nms_topn = cfg[state].RPN_PRE_NMS_TOP_N
			p.post_nms_topn = cfg[state].RPN_POST_NMS_TOP_N
			p.nms_thresh = cfg[state].RPN_NMS_THRESH

		if layer.name == "rfcn_cls":
			layer.convolution_param.num_output = position_num ** 2 * num_classes
		if layer.name == "rfcn_bbox":
			layer.convolution_param.num_output = position_num ** 2 * rfcn_loc_num
		if layer.name == "psroipooled_cls_rois":
			layer.psroi_pooling_param.spatial_scale = 1.0 / feat_stride
			layer.psroi_pooling_param.output_dim = num_classes
			layer.psroi_pooling_param.group_size = position_num
		if layer.name == "psroipooled_loc_rois":
			layer.psroi_pooling_param.spatial_scale = 1.0 / feat_stride
			layer.psroi_pooling_param.output_dim = rfcn_loc_num
			layer.psroi_pooling_param.group_size = position_num
		if layer.name == "ave_cls_score_rois":
			layer.pooling_param.kernel_size = position_num
			layer.pooling_param.stride = position_num
		if layer.name == "ave_bbox_pred_rois":
			layer.pooling_param.kernel_size = position_num
			layer.pooling_param.stride = position_num
		if layer.name == "annotator_detector":
			layer.box_annotator_ohem_param.roi_per_img = cfg.TRAIN.ROI_PER_IMG

		if layer.name == "cls_prob_reshape":
			layer.reshape_param.shape.dim[1] = num_classes
		if layer.name == "bbox_pred_reshape":
			layer.reshape_param.shape.dim[1] = rfcn_loc_num

                if layer.name=="conv_new_1":
                        layer.convolution_param.num_output= 1024
                if layer.name in ["rpn_output", "rpn_cls_score", "rpn_bbox_pred", "conv_new_1", "rfcn_cls", "rfcn_bbox"]:
                        layer.param[0].lr_mult=1.0
                        layer.param[1].lr_mult=2.0
                        layer.param[1].decay_mult=0.0
                        layer.convolution_param.weight_filler.type= "gaussian"
                        layer.convolution_param.weight_filler.std=0.01
                if layer.name == "rpn_output":
                        layer.convolution_param.num_output= 512
                if layer.name in ["rpn_cls_score", "rpn_bbox_pred", "rfcn_cls", "rfcn_bbox"]:
                        layer.name="hyy/"+layer.name

                if layer.name in ["res5a_branch1", "res5a_branch2a"]:
                        if net_type==101:
                                layer.convolution_param.stride[0]=1
                        
	return net


if __name__ == '__main__':
	if len(sys.argv) == 1:
		print "python .. config_file net_file  net_type[50/101] state[train/test] ..."
		sys.exit(0)

	config_file = sys.argv[1]
	net_file = sys.argv[2]
        net_type = int(sys.argv[3])
	state = sys.argv[4]
	net_file = [net_file]

	cfg = hus_yaml.cfg_from_file(config_file)
	ppt = process_net_pt(modify_fun=modify)
	ppt.process_all(net_file, cfg, net_type, state)
