import torch
import argparse
import datetime

from tqdm import tqdm

from utils.loading import load_config
from utils.setup import *

from modules.routing import ConfidenceRouting

from skimage import io
import numpy as np

def arg_parser():

	parser = argparse.ArgumentParser()

	parser.add_argument('--config', required=False)

	args = parser.parse_args()

	return vars(args)

def prepare_input_data(batch, config, device):
	if config.DATA.input == 'multidepth':
		if config.DATA.intensity_grad:
			depth_tof = batch['tof_depth'].unsqueeze_(1)
			# depth_mono = batch['mono_depth'].unsqueeze_(1)
			depth_stereo = batch['stereo_depth'].unsqueeze_(1)
			intensity = batch['intensity'].unsqueeze_(1)
			grad = batch['gradient'].unsqueeze_(1)
			inputs = torch.cat((intensity, grad, depth_tof, depth_stereo), 1)
			inputs = inputs.to(device)
		else:
			depth_tof = batch['tof_depth'].unsqueeze_(1)
			# depth_mono = batch['mono_depth'].unsqueeze_(1)
			depth_stereo = batch['stereo_depth'].unsqueeze_(1)
			inputs = torch.cat((depth_tof, depth_stereo), 1)
			inputs = inputs.to(device)
	else:
		if config.DATA.intensity_grad:
			depth = batch[config.DATA.input].unsqueeze_(1)
			intensity = batch['intensity'].unsqueeze_(1)
			grad = batch['gradient'].unsqueeze_(1)
			inputs = torch.cat((intensity, grad, depth), 1)
			inputs = inputs.to(device)
		else:
			inputs = batch[config.DATA.input] 
			inputs = inputs.unsqueeze_(1)
			inputs = inputs.to(device)

	target = batch[config.DATA.target] # 2, 512, 512 (batch size, height, width)
	target = target.to(device)
	target = target.unsqueeze_(1) # 2, 1, 512, 512 (batch size, channels, height, width)
	return inputs, target

def test(config):

	if config.SETTINGS.gpu:
		device = torch.device('cuda:0')
	else: 
		device = torch.device("cpu")

	# get test dataset
	test_data_config = get_data_config(config, mode='test')
	test_dataset = get_data(config.DATA.dataset, test_data_config)
	test_loader = torch.utils.data.DataLoader(test_dataset, config.TESTING.test_batch_size, config.TESTING.test_shuffle)

	# define model
	Cin = 0
	if config.DATA.input == 'multidepth':
		Cin += 2
	else:
		Cin += 1
	if config.DATA.intensity_grad:
		Cin += 2


	model = ConfidenceRouting(Cin=Cin,
							  F=config.MODEL.contraction,
							  Cout=config.MODEL.n_output_channels,
							  depth=config.MODEL.depth,
							  batchnorms=config.MODEL.normalization)
	# load model
	checkpoint = torch.load(config.TESTING.model_path)
	model.load_state_dict(checkpoint['state_dict'])

	model = model.to(device)

	n_test_batches = int(len(test_dataset) / config.TESTING.test_batch_size)

	for i, batch in enumerate(tqdm(test_loader, total=n_test_batches)):
		inputs, target = prepare_input_data(batch, config, device)
		
		output = model.forward(inputs)

		est = output[:, 0, :, :].unsqueeze_(1)
		unc = output[:, 1, :, :].unsqueeze_(1)

		est = est.detach().cpu().numpy()
		est = est.squeeze()
		estplot = est
		est = est*1000
		est = est.astype('uint16')

		unc = unc.detach().cpu().numpy()
		unc = unc.squeeze() # there is a relu activation function as the last step of the confidence decoder s.t. we always get non-negative numbers
		confidence = np.exp(-1. * unc)
		confidence *= 10000
		confidence = confidence.astype('uint16')

		output_dir_refined = config.DATA.root_dir + '/' + batch['frame_id'][0].split('/')[0] + '/' + batch['frame_id'][0].split('/')[1] + '/left_routing_refined_' + config.TESTING.model_path.split('/')[10]
		output_dir_confidence = config.DATA.root_dir + '/' + batch['frame_id'][0].split('/')[0] + '/' + batch['frame_id'][0].split('/')[1] + '/left_routing_confidence_' + config.TESTING.model_path.split('/')[10]

		if not os.path.exists(output_dir_refined):
			os.makedirs(output_dir_refined)

		if not os.path.exists(output_dir_confidence):
			os.makedirs(output_dir_confidence)

		io.imsave(output_dir_refined + '/' + batch['frame_id'][0].split('/')[-1] + '.png', est)
		io.imsave(output_dir_confidence + '/' + batch['frame_id'][0].split('/')[-1] + '.png', confidence)

		# input_ = input_.detach().numpy()
		# input_ = input_.squeeze()

		# target = target.detach().numpy()
		# target = target.squeeze()

		# final = np.concatenate((input_, target),axis=1)
		# final = np.concatenate((final, estplot), axis=1)
		# final = np.concatenate((final, unc), axis=1)

		# io.imshow(final, vmin=0, vmax=7.5)
		# io.show()

		# break


if __name__ == '__main__':

	# get arguments
	args = arg_parser()

	# get configs
	# load config
	if args['config']:
		config = load_config(args['config'])
	else:
		raise ValueError('Missing configuration: Please either specify config or experiment.')

	# train
	test(config)

