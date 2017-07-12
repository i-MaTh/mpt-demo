import sys

sys.path.append("/home/huyangyang/rfcn/caffe/python")
sys.path.append("/home/huyangyang/tools")
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import caffe
import net_zoom
import hus_yaml

cfg = hus_yaml.cfg_from_file("config.yaml")
if cfg.ANCHOR_GENERATOR.POLICY == "SCALE_RATIO":
	anchors_num = len(cfg.ANCHOR_GENERATOR.SCALES) * \
				  len(cfg.ANCHOR_GENERATOR.RATIOS)
if cfg.ANCHOR_GENERATOR.POLICY == "SLIDING_WINDOW":
	anchors_num = len(cfg.ANCHOR_GENERATOR.SLIDING_WINDOW_WIDTH)
num_classes = cfg.NUM_CLASSES
position_num = cfg.POSITION_NUM
feat_stride = cfg.FEAT_STRIDE


def setupDataLayers(net_type):
	net = caffe.NetSpec()

	multilabel_gt_box_str = ','.join(["net['{}_gt_boxes']".format(class_label) for class_label in cfg.CLASSES[1:]])
	if net_type == "train":
		exec ('''net['data'], net['im_info'], {} = L.Data(name="input-data", type="Python",
															 python_param=dict(
																 module="roi_data_layer.layer",
																 layer="RoIDataLayer",
																 param_str=str(dict(
																	 num_classes=cfg.NUM_CLASSES))),
															 ntop={})'''.format(multilabel_gt_box_str, cfg.NUM_CLASSES+1))
	if net_type == "deploy":
		net['data'] = L.Input(name="data", input_param={'shape': {'dim': [1, 3, 224, 224]}})
		net['im_info'] = L.Input(name="im_info", input_param={'shape': {'dim': [1, 2]}})
	return net


def add_train_rfcn_layers(net, split_to_rpn_layer, end_body_layer=None, prefix=""):
	# ensure no same output
	kwargs = {
		'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		'weight_filler': dict(type='msra'),
		'bias_filler': dict(type='constant', value=0.0)}
	proposal_param = {'ratio': cfg.ANCHOR_GENERATOR.RATIOS, 'scale': cfg.ANCHOR_GENERATOR.SCALES,
					  'base_size': cfg.FEAT_STRIDE, 'feat_stride': cfg.FEAT_STRIDE,
					  'pre_nms_topn': cfg.TRAIN.RPN_PRE_NMS_TOP_N, 'post_nms_topn': cfg.TRAIN.RPN_POST_NMS_TOP_N,
					  'nms_thresh': cfg.TRAIN.RPN_NMS_THRESH, 'min_size': cfg.TRAIN.DATA_AUG.MIN_SIZE}

	net[prefix + 'rpn_output'] = L.Convolution(split_to_rpn_layer, num_output=256, pad=1, kernel_size=3,
											   stride=1, **kwargs)
	net[prefix + 'rpn_output_relu'] = L.ReLU(net[prefix + 'rpn_output'], in_place=True)
	net[prefix + 'rpn_cls_score'] = L.Convolution(net[prefix + 'rpn_output_relu'], num_output=anchors_num * 2, pad=0,
												  kernel_size=1, stride=1,
												  **kwargs)
	net[prefix + 'rpn_bbox_pred'] = L.Convolution(net[prefix + 'rpn_output_relu'], num_output=anchors_num * 4, pad=0,
												  kernel_size=1, stride=1,
												  **kwargs)
	net[prefix + 'rpn_cls_score_reshape'] = L.Reshape(net[prefix + 'rpn_cls_score'], reshape_param={
		'shape': {'dim': [0, 2, -1, 0]}})
	net[prefix + 'rpn_labels'], net[prefix + 'rpn_bbox_targets'], net[prefix + 'rpn_bbox_inside_weights'], net[
		prefix + 'rpn_bbox_outside_weights'] = L.Python(
		net[prefix + 'rpn_cls_score'], net[prefix + 'gt_boxes'], net['im_info'], net['data'],
		name=prefix+"rpn-data",
		python_param={'module': "rpn.anchor_target_layer", 'layer': "AnchorTargetLayer"}, ntop=4)
	net[prefix + 'rpn_loss_cls'] = L.SoftmaxWithLoss(net[prefix + 'rpn_cls_score_reshape'], net[prefix + 'rpn_labels'],
													 loss_weight=1.0,
													 propagate_down=[True, False],
													 loss_param={"ignore_label": -1, "normalize": True})
	net[prefix + 'rpn_loss_bbox'] = L.SmoothL1Loss(net[prefix + 'rpn_bbox_pred'], net[prefix + 'rpn_bbox_targets'],
												   net[prefix + 'rpn_bbox_inside_weights'],
												   net[prefix + 'rpn_bbox_outside_weights'], loss_weight=1.0,
												   smooth_l1_loss_param={'sigma': 3.0})
	net[prefix + 'rpn_cls_prob'] = L.Softmax(net[prefix + 'rpn_cls_score_reshape'])
	net[prefix + 'rpn_cls_prob_reshape'] = L.Reshape(net[prefix + 'rpn_cls_prob'],
													 reshape_param={'shape': {'dim': [0, 2 * anchors_num, -1, 0]}})
	# net[prefix+'rpn_rois'], net[prefix+'rpn_scores'] = L.Python(net[prefix+'rpn_cls_prob_reshape'], net[prefix+'rpn_bbox_pred'], net['im_info'], name=prefix+"proposal",
	# python_param={'module': "rpn.proposal_layer", 'layer': "ProposalLayer"},
	# ntop=2)
	net[prefix + 'rpn_rois'], net[prefix + 'rpn_scores'] = L.Proposal(net[prefix + 'rpn_cls_prob_reshape'],
																	  net[prefix + 'rpn_bbox_pred'],
																	  net['im_info'], name=prefix+"proposal",
																	  proposal_param=proposal_param, ntop=2)
	net[prefix + 'rpn_scores_silence'] = L.Silence(net[prefix + 'rpn_scores'], ntop=0)
	net[prefix + 'rois'], net[prefix + 'labels'], net[prefix + 'bbox_targets'], net[prefix + 'bbox_inside_weights'], \
	net[prefix + 'bbox_outside_weights'], net[prefix + 'pos_num'] = L.Python(
		net[prefix + 'rpn_rois'], net[prefix + 'gt_boxes'], net['data'], name=prefix+"roi-data",
		python_param={'module': "rpn.proposal_target_layer", 'layer': "ProposalTargetLayer"}, ntop=6)
	net[prefix + 'conv_new_1'] = L.Convolution(
		end_body_layer, num_output=256, pad=0, kernel_size=1, stride=1, **kwargs)
	net[prefix + 'conv_new_1_relu'] = L.ReLU(net[prefix + 'conv_new_1'], in_place=True)
	net[prefix + 'rfcn_cls'] = L.Convolution(net[prefix + 'conv_new_1_relu'],
											 num_output=position_num ** 2 * num_classes,
											 pad=0, kernel_size=1, stride=1,
											 **kwargs)
	net[prefix + 'rfcn_bbox'] = L.Convolution(net[prefix + 'conv_new_1_relu'], num_output=position_num ** 2 * 8, pad=0,
											  kernel_size=1, stride=1,
											  **kwargs)
	net[prefix + 'psroipooled_cls_rois'] = L.PSROIPooling(net[prefix + 'rfcn_cls'], net[prefix + 'rois'],
														  psroi_pooling_param={'spatial_scale': 1.0 / feat_stride,
																			   'output_dim': num_classes,
																			   'group_size': position_num})
	net[prefix + 'cls_score'] = L.Pooling(net[prefix + 'psroipooled_cls_rois'], name=prefix+"ave_cls_score_rois",
										  pool=P.Pooling.AVE,
										  kernel_size=position_num, stride=position_num)
	net[prefix + 'psroipooled_loc_rois'] = L.PSROIPooling(net[prefix + 'rfcn_bbox'], net[prefix + 'rois'],
														  psroi_pooling_param={'spatial_scale': 1.0 / feat_stride,
																			   'output_dim': 8,
																			   'group_size': position_num})
	net[prefix + 'bbox_pred'] = L.Pooling(net[prefix + 'psroipooled_loc_rois'], name=prefix+"ave_bbox_pred_rois",
										  pool=P.Pooling.AVE,
										  kernel_size=position_num, stride=position_num)
	net[prefix + 'temp_loss_cls'], net[prefix + 'temp_prob_cls'], net[
		prefix + 'per_roi_loss_cls'] = L.SoftmaxWithLossOHEM(net[prefix + 'cls_score'], net[prefix + 'labels'],
															 name=prefix+"per_roi_loss_cls",
															 loss_weight=[
																 0.0, 0.0, 0.0],
															 propagate_down=[
																 False, False],
															 ntop=3)
	net[prefix + 'temp_loss_bbox'], net[prefix + 'per_roi_loss_bbox'] = L.SmoothL1LossOHEM(net[prefix + 'bbox_pred'],
																						   net[prefix + 'bbox_targets'],
																						   net[
																							   prefix + 'bbox_inside_weights'],
																						   name=prefix+"per_roi_loss_bbox",
																						   loss_weight=[
																							   0.0, 0.0],
																						   propagate_down=[False, False,
																										   False],
																						   ntop=2)
	net[prefix + 'per_roi_loss'] = L.Eltwise(
		net[prefix + 'per_roi_loss_cls'], net[prefix + 'per_roi_loss_bbox'], propagate_down=[False, False])
	net[prefix + 'labels_ohem'], net[prefix + 'bbox_loss_weights_ohem'] = L.BoxAnnotatorOHEM(net[prefix + 'rois'], net[
		prefix + 'per_roi_loss'], net[prefix + 'labels'],
																							 net[
																								 prefix + 'bbox_inside_weights'],
																							 name=prefix+"annotator_detector",
																							 propagate_down=[
																								 False, False, False,
																								 False],
																							 box_annotator_ohem_param={
																								 'roi_per_img': cfg.TRAIN.ROI_PER_IMG,
																								 'ignore_label': -1},
																							 ntop=2)
	net[prefix + 'silence'] = L.Silence(net[prefix + 'bbox_outside_weights'], net[prefix + 'temp_loss_cls'],
										net[prefix + 'temp_prob_cls'], net[prefix + 'temp_loss_bbox'], ntop=0)
	net[prefix + 'loss_cls'] = L.SoftmaxWithLoss(net[prefix + 'cls_score'], net[prefix + 'labels_ohem'], name=prefix+"loss",
												 loss_weight=1.0,
												 propagate_down=[True, False], loss_param={'ignore_label': -1})
	net[prefix + 'accuracy'] = L.Accuracy(net[prefix + 'cls_score'], net[prefix + 'labels_ohem'],
										  propagate_down=[False, False],
										  accuracy_param={'ignore_label': -1})
	net[prefix + 'loss_bbox'] = L.Loss(net[prefix + 'bbox_pred'], net[prefix + 'bbox_targets'],
									   net[prefix + 'bbox_loss_weights_ohem'], net[prefix + 'pos_num'],
									   type="SmoothL1LossOHEM", loss_weight=1.0,
									   propagate_down=[True, False, False, False],
									   loss_param={'normalization': P.Loss.POS_NUM})
	return net


def add_deploy_rfcn_layers(net, split_to_rpn_layer, end_body_layer=None, prefix=""):
	# ensure no same output
	kwargs = {
		'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		'weight_filler': dict(type='msra'),
		'bias_filler': dict(type='constant', value=0.0)}
	proposal_param = {'ratio': cfg.ANCHOR_GENERATOR.RATIOS, 'scale': cfg.ANCHOR_GENERATOR.SCALES,
					  'base_size': cfg.FEAT_STRIDE, 'feat_stride': cfg.FEAT_STRIDE,
					  'pre_nms_topn': cfg.TEST.RPN_PRE_NMS_TOP_N, 'post_nms_topn': cfg.TEST.RPN_POST_NMS_TOP_N,
					  'nms_thresh': cfg.TEST.RPN_NMS_THRESH, 'min_size': cfg.TRAIN.DATA_AUG.MIN_SIZE}

	net[prefix + 'rpn_output'] = L.Convolution(split_to_rpn_layer, num_output=256, pad=1, kernel_size=3,
											   stride=1, **kwargs)
	net[prefix + 'rpn_output_relu'] = L.ReLU(net[prefix + 'rpn_output'], in_place=True)
	net[prefix + 'rpn_cls_score'] = L.Convolution(net[prefix + 'rpn_output_relu'], num_output=anchors_num * 2, pad=0,
												  kernel_size=1, stride=1,
												  **kwargs)
	net[prefix + 'rpn_bbox_pred'] = L.Convolution(net[prefix + 'rpn_output_relu'], num_output=anchors_num * 4, pad=0,
												  kernel_size=1, stride=1,
												  **kwargs)
	net[prefix + 'rpn_cls_score_reshape'] = L.Reshape(net[prefix + 'rpn_cls_score'], reshape_param={
		'shape': {'dim': [0, 2, -1, 0]}})
	net[prefix + 'rpn_cls_prob'] = L.Softmax(net[prefix + 'rpn_cls_score_reshape'])
	net[prefix + 'rpn_cls_prob_reshape'] = L.Reshape(net[prefix + 'rpn_cls_prob'],
													 reshape_param={'shape': {'dim': [0, 2 * anchors_num, -1, 0]}})
	net[prefix + 'rois'], net[prefix + 'rpn_score'] = L.Proposal(net[prefix + 'rpn_cls_prob_reshape'],
																 net[prefix + 'rpn_bbox_pred'], net['im_info'],
																 name=prefix+"proposal",
																 proposal_param=proposal_param, ntop=2)
	net[prefix + 'conv_new_1'] = L.Convolution(
		end_body_layer, num_output=256, pad=0, kernel_size=1, stride=1, **kwargs)
	net[prefix + 'conv_new_1_relu'] = L.ReLU(net[prefix + 'conv_new_1'], in_place=True)
	net[prefix + 'rfcn_cls'] = L.Convolution(net[prefix + 'conv_new_1_relu'],
											 num_output=position_num ** 2 * num_classes,
											 pad=0, kernel_size=1, stride=1,
											 **kwargs)
	net[prefix + 'rfcn_bbox'] = L.Convolution(net[prefix + 'conv_new_1_relu'], num_output=position_num ** 2 * 8, pad=0,
											  kernel_size=1, stride=1,
											  **kwargs)
	net[prefix + 'psroipooled_cls_rois'] = L.PSROIPooling(net[prefix + 'rfcn_cls'], net[prefix + 'rois'],
														  psroi_pooling_param={'spatial_scale': 1.0 / feat_stride,
																			   'output_dim': num_classes,
																			   'group_size': position_num})
	net[prefix + 'cls_score'] = L.Pooling(net[prefix + 'psroipooled_cls_rois'], name=prefix+"ave_cls_score_rois",
										  pool=P.Pooling.AVE,

										  kernel_size=position_num, stride=position_num)

	net[prefix + 'cls_prob_pred'] = L.Softmax(net[prefix + 'cls_score'])
	net[prefix + 'cls_prob'] = L.Reshape(net[prefix + 'cls_prob_pred'], name=prefix+"cls_prob_reshape",
										 reshape_param={'shape': {'dim': [-1, num_classes]}})
	net[prefix + 'psroipooled_loc_rois'] = L.PSROIPooling(net[prefix + 'rfcn_bbox'], net[prefix + 'rois'],
														  psroi_pooling_param={'spatial_scale': 1.0 / feat_stride,
																			   'output_dim': 8,
																			   'group_size': position_num})
	net[prefix + 'bbox_pred_pred'] = L.Pooling(net[prefix + 'psroipooled_loc_rois'], name=prefix+"ave_bbox_pred_rois",
											   pool=P.Pooling.AVE,
											   kernel_size=position_num, stride=position_num)
	net[prefix + 'bbox_pred'] = L.Reshape(net[prefix + 'bbox_pred_pred'], name=prefix+"bbox_pred_reshape", reshape_param={
		'shape': {'dim': [-1, 8]}})
	return net


def get_solver_pt(train_data_file):
	solver_config = cfg.SOLVER
	max_epoch = solver_config.max_epoch
	with open(train_data_file, 'r') as f:
		train_data_num = len(f.readlines())
	batch_size = cfg.TRAIN.IMS_PER_BATCH
	max_iter = int(max_epoch * train_data_num / batch_size)
	stepsize = int(solver_config.step_epoch * train_data_num / batch_size)

	solver_param = {
		# Train parameters
		'train_net': "train.prototxt",
		'base_lr': solver_config.base_lr,
		'weight_decay': 0.0005,
		'lr_policy': "step",
		'stepsize': stepsize,
		'gamma': 0.1,
		'momentum': 0.9,
		'iter_size': 1,
		'max_iter': max_iter,
		'snapshot': solver_config.snapshot,
		'display': 20,
		'average_loss': 10,
		'type': "SGD",
		'snapshot_prefix': "models_solverstates/",
		'snapshot_after_train': True,
	}
	solver = caffe_pb2.SolverParameter(**solver_param)

	with open("solver.prototxt", 'w') as f:
		f.write(str(solver))


def get_pt(net_type, net_name, split_to_rpn_layer_name, end_body_layer_name):
	net = setupDataLayers(net_type)
	exec ('''net=net_zoom.{}(net, "{}")'''.format(net_name, "net.data"))
	for class_name in cfg.CLASSES[1:]:
		exec (
			'''net=add_{}_rfcn_layers(net, net['{}'], net['{}'], prefix='{}')'''.format(net_type,
																						split_to_rpn_layer_name,
																						end_body_layer_name,
																						class_name + "_"))
	pt_name = "{}.prototxt".format(net_type)

	with open(pt_name, 'w') as f:
		f.write("name: \"rfcn_{}\"\n".format(net_name))
		f.write(str(net.to_proto()))


if __name__ == '__main__':
	if len(sys.argv) == 1:
		print "python .. train_data_path net_type[train/deploy/solver/all] net_name[pz_L_net,...] split_to_rpn_layer_name[b6_sum] end_body_layer_name[b8_sum]"
		sys.exit(0)

	train_data_path = sys.argv[1]
	net_type = sys.argv[2]
	net_name = sys.argv[3]
	split_to_rpn_layer_name = sys.argv[4]
	end_body_layer_name = sys.argv[5]

	if net_type == "all":
		get_pt("train", net_name, split_to_rpn_layer_name, end_body_layer_name)
	get_pt("deploy", net_name, split_to_rpn_layer_name, end_body_layer_name)
	get_solver_pt(train_data_path)
	if net_type == "train" or net_type == "deploy":
		get_pt(net_type, net_name, split_to_rpn_layer_name, end_body_layer_name)
	if net_type == "solver":
		get_solver_pt(train_data_path)
