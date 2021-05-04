import torch
import argparse
import datetime
import numpy as np
import random

from tqdm import tqdm

from utils.setup import *
from utils.loading import *
from utils.loss import *

from modules.model import FusionNet
from modules.routing import ConfidenceRouting
from modules.pipeline import Pipeline


def arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config')

    args = parser.parse_args()
    return vars(args)

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# @profile
def train_fusion(args):

    config = load_config_from_yaml(args['config'])

    assert not (config.LOSS.gt_loss and config.FILTERING_MODEL.w_features), "You can only use gt loss when not using features"

    config.TIMESTAMP = datetime.datetime.now().strftime('%y%m%d-%H%M%S')

    # set seed for reproducibility
    if config.SETTINGS.seed:
        random.seed(config.SETTINGS.seed)
        np.random.seed(config.SETTINGS.seed)
        torch.manual_seed(config.SETTINGS.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # get workspace
    workspace = get_workspace(config)

    # save config before training
    workspace.save_config(config)

    if config.SETTINGS.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    config.FUSION_MODEL.device = device


    # get datasets
    # get train dataset
    train_data_config = get_data_config(config, mode='train')
    train_dataset = get_data(config.DATA.dataset, train_data_config)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               config.TRAINING.train_batch_size, config.TRAINING.train_shuffle)

    # get val dataset
    val_data_config = get_data_config(config, mode='val')
    val_dataset = get_data(config.DATA.dataset, val_data_config)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             config.TRAINING.val_batch_size, config.TRAINING.val_shuffle)

    # get database
    # get train database
    train_database = get_database(train_dataset, config, mode='train')
    val_database = get_database(val_dataset, config, mode='val')

    # setup pipeline
    pipeline = Pipeline(config)
    pipeline = pipeline.to(device) # put the networks on the gpu


    print('Fusion Net ToF: ', count_parameters(pipeline._fusion_network_tof))
    print('Fusion Net stereo: ', count_parameters(pipeline._fusion_network_stereo))
    print('Feature Net ToF: ', count_parameters(pipeline._feature_network_tof))
    print('Feature Net Stereo: ', count_parameters(pipeline._feature_network_stereo))
    print('Filtering Net: ', count_parameters(pipeline._filtering_network))
    print('Fusion and Filtering: ', count_parameters(pipeline))

    # optimization
    if config.FILTERING_MODEL.fuse_sensors:
        criterion = Fusion_TranslationFuseSensorsLoss(config)
    else:
        criterion = Fusion_TranslationLoss(config)

    # load pretrained routing model into parameters
    if config.ROUTING.do:
        if config.DATA.fusion_strategy == 'routingNet':
            routing_checkpoint = torch.load(config.TESTING.routing_model_path)
            # print(routing_checkpoint)
            # load_model(config.TESTING.routing_model_path, pipeline._routing_network)
            # Keep line below until I see that the new loading function works.
            pipeline._routing_network.load_state_dict(routing_checkpoint['state_dict'])
        elif config.DATA.fusion_strategy == 'fusionNet' or config.DATA.fusion_strategy == 'two_fusionNet' or config.DATA.fusion_strategy == 'fusionNet_conditioned':
            # routing_mono_checkpoint = torch.load(config.TESTING.routing_mono_model_path)
            routing_stereo_checkpoint = torch.load(config.TRAINING.routing_stereo_model_path)
            routing_tof_checkpoint = torch.load(config.TRAINING.routing_tof_model_path)

            # pipeline._routing_network_mono.load_state_dict(routing_mono_checkpoint['state_dict'])
            pipeline._routing_network_stereo.load_state_dict(routing_stereo_checkpoint['pipeline_state_dict'])
            pipeline._routing_network_tof.load_state_dict(routing_tof_checkpoint['pipeline_state_dict'])

    if config.TESTING.pretrain_filtering_net:
        load_filtering(config.TESTING.fusion_model_path, pipeline) # this is the filtering loading

    if config.TRAINING.pretraining:
        load_pipeline(config.TRAINING.pretraining_fusion_tof_model_path, pipeline, 'tof')
        # load_pipeline_stereo(config.TRAINING.pretraining_fusion_stereo_model_path, pipeline, 'stereo') # only for the routedfusion_only_fusion loading
        load_pipeline(config.TRAINING.pretraining_fusion_stereo_model_path, pipeline, 'stereo') # for loading from late_fusion


    if config.FILTERING_MODEL.w_features:
        feature_params = list(pipeline._feature_network_tof.parameters()) + list(pipeline._feature_network_stereo.parameters())
        optimizer_feature = torch.optim.RMSprop(feature_params,
                                                config.OPTIMIZATION.lr_fusion,
                                                config.OPTIMIZATION.rho,
                                                config.OPTIMIZATION.eps,
                                                momentum=config.OPTIMIZATION.momentum,
                                                weight_decay=config.OPTIMIZATION.weight_decay)

        scheduler_feature = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_feature,
                                                            step_size=config.OPTIMIZATION.scheduler.step_size_fusion,
                                                            gamma=config.OPTIMIZATION.scheduler.gamma_fusion)


    if not config.FILTERING_MODEL.fixed:
        optimizer_filt = torch.optim.RMSprop(pipeline._filtering_network.parameters(),
                                            config.OPTIMIZATION.lr_filtering,
                                            config.OPTIMIZATION.rho,
                                            config.OPTIMIZATION.eps,
                                            momentum=config.OPTIMIZATION.momentum,
                                            weight_decay=config.OPTIMIZATION.weight_decay)

        scheduler_filt = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_filt,
                                                        step_size=config.OPTIMIZATION.scheduler.step_size_filtering,
                                                        gamma=config.OPTIMIZATION.scheduler.gamma_filtering)
        
    if not config.FUSION_MODEL.fixed:
        fusion_params = list(pipeline._fusion_network_tof.parameters()) + list(pipeline._fusion_network_stereo.parameters())
        optimizer_fusion = torch.optim.RMSprop(fusion_params,
                                            config.OPTIMIZATION.lr_fusion,
                                            config.OPTIMIZATION.rho,
                                            config.OPTIMIZATION.eps,
                                            momentum=config.OPTIMIZATION.momentum,
                                            weight_decay=config.OPTIMIZATION.weight_decay)

        scheduler_fusion = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_fusion,
                                                        step_size=config.OPTIMIZATION.scheduler.step_size_fusion,
                                                        gamma=config.OPTIMIZATION.scheduler.gamma_fusion)


    # define some parameters
    n_batches = float(len(train_dataset)/config.TRAINING.train_batch_size)

    # evaluation metrics
    best_iou = 0. # best filtered 
    best_iou_tof = 0.
    best_iou_stereo = 0.

    is_best=False
    is_best_tof=False
    is_best_stereo=False

    # when using fuse sensors true make sure we use both sensors here in variable sensors
    if config.FILTERING_MODEL.fuse_sensors:
        sensors = ['tof', 'stereo'] # make sure thi only used when we have input: multidepth and fusion_strategy: fusionNet and derivatives
    else:
        sensors = ['tof']
    sensor_opposite = {'tof': 'stereo', 'stereo': 'tof'}

    for epoch in range(0, config.TRAINING.n_epochs):

        workspace.log('Training epoch {}/{}'.format(epoch, config.TRAINING.n_epochs),
                                  mode='train')

        pipeline.train()
        if config.DATA.fusion_strategy == 'two_fusionNet':
            if config.ROUTING.do:
                pipeline._routing_network_tof.eval()
                pipeline._routing_network_stereo.eval()
            if config.FUSION_MODEL.fixed:
                pipeline._fusion_network_tof.eval()
                pipeline._fusion_network_stereo.eval()
            if config.FILTERING_MODEL.fixed:
                pipeline._filtering_network.eval()
        else:
            pipeline._routing_network.eval()
            # pipeline._fusion_network.eval()
            # pipeline._filtering_network.eval()

        # resetting databases before each epoch starts
        train_database.reset()
        val_database.reset()

        train_loss = 0
        grad_norm = 0
        val_norm = 0
        l1_interm = 0
        l1_grid = 0
        l1_grid_tof = 0
        l1_grid_stereo = 0
        l1_gt_grid = 0
        l_feat = 0

        for i, batch in tqdm(enumerate(train_loader), total=len(train_dataset)):
            # reset the database for every new trajectory (if using hybrid loading strategy)
            # if batch['frame_id'][0].split('/')[-1] == '0' and config.DATA.data_load_strategy == 'hybrid':
            #     workspace.log('Starting new trajectory {} at step {}'.format(batch['frame_id'][0][:-2], i), mode='train')
            #     workspace.log('Resetting grid for scene {} at step {}'.format(batch['frame_id'][0].split('/')[0], i),
            #                       mode='train')
            #     train_database.reset(batch['frame_id'][0].split('/')[0])

            if config.TRAINING.reset_strategy:
                if np.random.random_sample() <= config.TRAINING.reset_prob:
                    workspace.log('Resetting randomly trajectory {} at step {}'.format(batch['frame_id'][0][:-2], i), mode='train')
                    workspace.log('Resetting grid for scene {} at step {}'.format(batch['frame_id'][0].split('/')[0], i),
                                  mode='train')
                    train_database.reset(batch['frame_id'][0].split('/')[0])

            # take care of the fusion strategy here i.e. loop through the 3 integrations randomly by adding the "mask" and depth 
            # as keys in the batch. But I also need knowledge of sensor label for the routing network. I create the 'routing_net'
            # and 'depth' keys and pass that to the fuse_training function in three steps. Also pass the routing threshold as a
            # key.
            # fusion pipeline
            if config.DATA.input == 'multidepth' and (config.DATA.fusion_strategy == 'fusionNet' or config.DATA.fusion_strategy == 'two_fusionNet' or config.DATA.fusion_strategy == 'fusionNet_conditioned'):
                # randomly integrate the three sensors
                random.shuffle(sensors)
                for sensor in sensors:
                    batch['depth'] = batch[sensor + '_depth']
                    batch['confidence_threshold'] = eval('config.ROUTING.threshold_' + sensor) 
                    batch['routing_net'] = 'self._routing_network_' + sensor
                    if config.DATA.fusion_strategy == 'two_fusionNet':
                        batch['fusion_net'] = 'self._fusion_network_' + sensor
                        batch['feature_net'] = 'self._feature_network_' + sensor
                    batch['mask'] = batch[sensor + '_mask']
                    batch['sensor'] = sensor
                    batch['sensor_opposite'] = sensor_opposite[sensor]
                    output = pipeline.fuse_training(batch, train_database, device)

                    # optimization
                    if output is None: # output is None when no valid indices were found for the filtering net within the random
                    # bbox within the bounding volume of the integrated indices
                        continue

                    output = criterion(output)

                    # loss = criterion(output['tsdf_filtered_grid'], output['tsdf_target_grid'])
                    # if loss.grad_fn: # this is needed because when the mono mask filters out all pixels, this results in a failure
                    # print('bef backward: ', torch.cuda.memory_allocated(device))
                    train_loss += output['loss'].item() # note that this loss is a moving average over the training window of log_freq steps
                    if output['l1_interm'] is not None:
                        l1_interm += output['l1_interm'].item() # note that this loss is a moving average over the training window of log_freq steps
                    l1_grid += output['l1_grid'].item() 
                    if output['l1_gt_grid'] is not None:
                        l1_gt_grid += output['l1_gt_grid'].item() 

                    if config.FILTERING_MODEL.fuse_sensors:
                        if output['l1_grid_tof'] is not None:
                            l1_grid_tof += output['l1_grid_tof'].item() 
                        if output['l1_grid_stereo'] is not None:
                            l1_grid_stereo += output['l1_grid_stereo'].item() 
                    # l_occ += l_occ_tmp.item() 
                    # l_feat += l_feat_tmp.item() 


                    output['loss'].backward()


            else:
                output = pipeline.fuse_training(batch, train_database, device)

                tsdf_fused = output['tsdf_fused']
                tsdf_target = output['tsdf_target']

                # optimization
                loss, l1_fused_tmp, l_translation_tmp = criterion(output['tsdf_filtered_grid'], output['tsdf_target_grid'], 
                    output['tsdf_fused'], output['tsdf_target'], 4)
                # loss = criterion(output['tsdf_filtered_grid'], output['tsdf_target_grid'])
                # if loss.grad_fn: # this is needed because when the mono mask filters out all pixels, this results in a failure
                # print('bef backward: ', torch.cuda.memory_allocated(device))
                # print(loss)
                loss.backward()
                    # print([param.grad for name, param in pipeline._fusion_network.named_parameters()])
                    # print([name for name, param in pipeline._fusion_network.named_parameters()])

                train_loss += loss.item() # note that this loss is a moving average over the training window of log_freq steps
                l1_fused += l1_fused_tmp.item() # note that this loss is a moving average over the training window of log_freq steps
                l_translation += l_translation_tmp.item() 
                # std += output['std']

            del batch

            for name, param in pipeline.named_parameters():
                if param.grad is not None:
                    if (i + 1) % config.OPTIMIZATION.accumulation_steps == 0 or i == n_batches - 1:
                        grad_norm += torch.norm(
                            param.grad)
                        # print(torch.norm(param.grad))
                    # optimizer.zero_grad() # REMOVE LATER!
                    # val_norm += torch.norm(param)
                    # print('grad norm: ', torch.norm(param.grad))
                    # print('val norm: ' , torch.norm(param))
                # if name.startswith('_feature'):
                #     print(name, param.grad)

            if (i + 1) % config.SETTINGS.log_freq == 0:

                # if config.DATA.fusion_strategy == 'two_fusionNet': # TODO: split plotting into tof and stereo
                #     divide = 2*config.SETTINGS.log_freq
                # else:
                divide = config.SETTINGS.log_freq
                train_loss /= divide
                grad_norm /= divide
                val_norm /= divide
                # print('averaged grad norm: ', grad_norm)
                # print('averaged val norm: ', val_norm)
                l1_interm /= divide
                l1_grid /= divide
                l1_gt_grid /= divide
                l1_grid_tof /= divide
                l1_grid_stereo /= divide

                # l_occ /= divide
                # l_feat /= divide
                workspace.writer.add_scalar('Train/loss', train_loss, global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/grad_norm', grad_norm, global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/val_norm', val_norm, global_step=i + 1 + epoch*n_batches)
                # workspace.writer.add_scalar('Train/lr_filt', get_lr(optimizer_filt), global_step=i + 1 + epoch*n_batches)
                # workspace.writer.add_scalar('Train/lr_fusion', get_lr(optimizer_fusion), global_step=i + 1 + epoch*n_batches)
                # workspace.writer.add_scalar('Train/loss_coeff', loss_coeff(i), global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/l1_interm', l1_interm, global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/l1_translation', l1_grid, global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/l1_gt_translation', l1_gt_grid, global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/l1_gt_tof', l1_gt_grid_tof, global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/l1_gt_stereo', l1_gt_grid_stereo, global_step=i + 1 + epoch*n_batches)
                # workspace.writer.add_scalar('Train/l1_gt_translation', l1_gt_grid_tof, global_step=i + 1 + epoch*n_batches)
                # workspace.writer.add_scalar('Train/l1_gt_translation', l1_gt_grid_stereo, global_step=i + 1 + epoch*n_batches)
                # workspace.writer.add_scalar('Train/occ_loss', l_occ, global_step=i + 1 + epoch*n_batches)
                # workspace.writer.add_scalar('Train/feat_loss', l_feat, global_step=i + 1 + epoch*n_batches)
                train_loss = 0
                grad_norm = 0
                val_norm = 0
                l1_interm = 0
                l1_grid = 0
                l1_grid_tof = 0
                l1_grid_stereo = 0
                l1_gt_grid = 0
                l_feat = 0

            if config.TRAINING.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(pipeline.parameters(),
                                                   max_norm=1.,
                                                   norm_type=2)

            if (i + 1) % config.OPTIMIZATION.accumulation_steps == 0 or i == n_batches - 1:
                if config.FILTERING_MODEL.w_features:
                    optimizer_feature.step()
                    optimizer_feature.zero_grad()
                    scheduler_feature.step()

                if not config.FILTERING_MODEL.fixed:
                    optimizer_filt.step()
                    optimizer_filt.zero_grad()
                    scheduler_filt.step()
                if not config.FUSION_MODEL.fixed:
                    optimizer_fusion.step()
                    optimizer_fusion.zero_grad()
                    scheduler_fusion.step()

            if (i + 1) % config.SETTINGS.eval_freq == 0 or i == n_batches - 1 or i == 20:  # evaluate after 20 steps wince then we have integrated at least one frame for each scene
            # if epoch % 2 == 0 and i == 0:
                # print(i)
                val_database.reset()     
                # zero out all grads
                if config.FILTERING_MODEL.w_features:
                    optimizer_feature.zero_grad()
                if not config.FILTERING_MODEL.fixed:
                    optimizer_filt.zero_grad()
                if not config.FUSION_MODEL.fixed:    
                    optimizer_fusion.zero_grad()

                # train_database.filter(value=1.) # when should I do the filtering? Since I do random resetting, it does not
                # make sense to do outlier filtering during training
                # doesn't make sense to evaluate the training grids if we only train the translation network now
                # train_eval = train_database.evaluate(mode='train', workspace=workspace)
                # # # print(train_eval)
                # workspace.writer.add_scalar('Train/mse', train_eval['mse'], global_step=i + 1 + epoch*n_batches)
                # workspace.writer.add_scalar('Train/acc', train_eval['acc'], global_step=i + 1 + epoch*n_batches)
                # workspace.writer.add_scalar('Train/iou', train_eval['iou'], global_step=i + 1 + epoch*n_batches)
                # workspace.writer.add_scalar('Train/mad', train_eval['mad'], global_step=i + 1 + epoch*n_batches)

                pipeline.eval()

                # validation step - fusion
                # The following loop only integrates around 500 frames so it will not be complete for the validation scene
                for k, batch in tqdm(enumerate(val_loader), total=len(val_dataset)):
                    # fusion pipeline
                    if config.DATA.input == 'multidepth' and (config.DATA.fusion_strategy == 'fusionNet' or config.DATA.fusion_strategy == 'two_fusionNet' or config.DATA.fusion_strategy == 'fusionNet_conditioned'):
                        # randomly integrate the three sensors
                        random.shuffle(sensors)
                        for sensor in sensors:
                            batch['depth'] = batch[sensor + '_depth']
                            batch['confidence_threshold'] = eval('config.ROUTING.threshold_' + sensor) 
                            batch['routing_net'] = 'self._routing_network_' + sensor
                            if config.DATA.fusion_strategy == 'two_fusionNet':
                                batch['fusion_net'] = 'self._fusion_network_' + sensor
                                batch['feature_net'] = 'self._feature_network_' + sensor
                            batch['mask'] = batch[sensor + '_mask']
                            batch['sensor'] = sensor
                            batch['sensor_opposite'] = sensor_opposite[sensor]
                            batch = transform.to_device(batch, device)
                            output = pipeline.fuse(batch, val_database, device)
 
                    else:
                        batch = transform.to_device(batch, device)

                        pipeline.fuse(batch, val_database, device)
                        # break
                    # if k == 1: # for debugging
                    #     break

                # run translation network on all voxels which have a non-zero weight
                for scene in val_database.filtered.keys():   
                    pipeline.sensor_fusion(scene, val_database, device)

                # val_database.filter(value=1.) # the more frames you integrate, the higher can the value be
                val_eval_tof, val_eval_stereo, val_eval_fused = val_database.evaluate(mode='val', workspace=workspace)

                workspace.writer.add_scalar('Val/mse_tof', val_eval_tof['mse'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/acc_tof', val_eval_tof['acc'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/iou_tof', val_eval_tof['iou'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/mad_tof', val_eval_tof['mad'], global_step=i + 1 + epoch*n_batches)

                workspace.writer.add_scalar('Val/mse_stereo', val_eval_stereo['mse'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/acc_stereo', val_eval_stereo['acc'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/iou_stereo', val_eval_stereo['iou'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/mad_stereo', val_eval_stereo['mad'], global_step=i + 1 + epoch*n_batches)

                workspace.writer.add_scalar('Val/mse_fused', val_eval_fused['mse'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/acc_fused', val_eval_fused['acc'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/iou_fused', val_eval_fused['iou'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/mad_fused', val_eval_fused['mad'], global_step=i + 1 + epoch*n_batches)



                # check if current checkpoint is best
                if val_eval_fused['iou'] >= best_iou:
                    is_best = True
                    best_iou = val_eval_fused['iou']
                    workspace.log('found new best model overall with iou {} at epoch {}'.format(best_iou, epoch),
                                  mode='val')
       
                else:
                    is_best = False

                if val_eval_tof['iou'] >= best_iou_tof:
                    is_best_tof = True
                    best_iou_tof = val_eval_tof['iou']
                    workspace.log('found new best tof model with iou {} at epoch {}'.format(best_iou_tof, epoch),
                                  mode='val')
 
                else:
                    is_best_tof = False

                if val_eval_stereo['iou'] >= best_iou_stereo:
                    is_best_stereo = True
                    best_iou_stereo = val_eval_stereo['iou']
                    workspace.log('found new best stereo model with iou {} at epoch {}'.format(best_iou_stereo, epoch),
                                  mode='val')
   
                else:
                    is_best_stereo = False
                # save models
                # train_database.save_to_workspace(workspace, mode='latest_train', save_mode=config.SETTINGS.save_mode)
                # val_database.save_to_workspace(workspace, is_best, is_best_tof, is_best_stereo, save_mode=config.SETTINGS.save_mode)

                # save checkpoint
                workspace.save_model_state({'pipeline_state_dict': pipeline.state_dict(),
                                                'optimizer_state_dict': optimizer_feature.state_dict(),
                                                'epoch': epoch},
                                               is_best=is_best, is_best_tof=is_best_tof, is_best_stereo=is_best_stereo)

                pipeline.train()
                if config.DATA.fusion_strategy == 'two_fusionNet':
                    if config.ROUTING.do:
                        pipeline._routing_network_tof.eval()
                        pipeline._routing_network_stereo.eval()
                    if config.FUSION_MODEL.fixed:
                        pipeline._fusion_network_tof.eval()
                        pipeline._fusion_network_stereo.eval()
                    if config.FILTERING_MODEL.fixed:
                        pipeline._filtering_network.eval()

                else:
                    pipeline._routing_network.eval()


            # if i == 6: # for debugging
                # break
        

if __name__ == '__main__':
    args = arg_parser()
    train_fusion(args)
