import torch
import argparse
import datetime
import numpy as np
import random

from tqdm import tqdm

from utils.setup import *
from utils.loading import *
from utils.loss import *

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

    # assert not (config.LOSS.gt_loss and config.FILTERING_MODEL.w_features), "You can only use gt loss when not using features"

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


    # specify number of features
    if config.FEATURE_MODEL.learned_features:
        config.FEATURE_MODEL.n_features = config.FEATURE_MODEL.n_features + config.FEATURE_MODEL.append_depth
    else:
        config.FEATURE_MODEL.n_features = 1 + config.FEATURE_MODEL.append_depth # 1 for label encoding of noise in gaussian threshold data

    # get database
    # get train database
    train_database = get_database(train_dataset, config, mode='train')
    val_database = get_database(val_dataset, config, mode='val')

    # setup pipeline
    pipeline = Pipeline(config)
    pipeline = pipeline.to(device) # put the networks on the gpu

    for sensor in config.DATA.input:
        print('Fusion Net ', sensor, ': ', count_parameters(pipeline.fuse_pipeline._fusion_network[sensor]))
        print('Feature Net ', sensor, ': ', count_parameters(pipeline.fuse_pipeline._feature_network[sensor]))

    print('Filtering Net: ', count_parameters(pipeline.filter_pipeline._filtering_network))
    print('Fusion and Filtering: ', count_parameters(pipeline))

    # optimization
    criterion = Fusion_TranslationLoss(config)

    # load pretrained routing model into parameters
    if config.ROUTING.do:
        raise NotImplementedError
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
        load_pipeline(config.TESTING.fusion_model_path, pipeline) # this is the filtering loading

    if config.TRAINING.pretraining:
        for sensor in config.DATA.input:
            if sensor == 'tof' or sensor == 'stereo':
                load_net_old(eval('config.TRAINING.pretraining_fusion_' + sensor +  '_model_path'), pipeline.fuse_pipeline._fusion_network[sensor], sensor)
            else:
                load_net(eval('config.TRAINING.pretraining_fusion_' + sensor +  '_model_path'), pipeline.fuse_pipeline._fusion_network[sensor], sensor)
        
            # load_net(eval('config.TRAINING.pretraining_fusion_' + sensor +  '_model_path'), pipeline.fuse_pipeline._fusion_network[sensor], sensor)
            # loading gt depth model fusion net
            # load_net('/cluster/work/cvl/esandstroem/src/late_fusion_3dconvnet/workspace/fusion/210507-093251/model/best.pth.tar', pipeline.fuse_pipeline._fusion_network[sensor], 'left_depth_gt_2')
        

    if config.FILTERING_MODEL.features_to_sdf_enc or config.FILTERING_MODEL.features_to_weight_head:
        feature_params = []
        for sensor in config.DATA.input:
            feature_params += list(pipeline.fuse_pipeline._feature_network[sensor].parameters())

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
        optimizer_filt = torch.optim.RMSprop(pipeline.filter_pipeline._filtering_network.parameters(),
                                            config.OPTIMIZATION.lr_filtering,
                                            config.OPTIMIZATION.rho,
                                            config.OPTIMIZATION.eps,
                                            momentum=config.OPTIMIZATION.momentum,
                                            weight_decay=config.OPTIMIZATION.weight_decay)

        scheduler_filt = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_filt,
                                                        step_size=config.OPTIMIZATION.scheduler.step_size_filtering,
                                                        gamma=config.OPTIMIZATION.scheduler.gamma_filtering)
        
    if not config.FUSION_MODEL.fixed:
        fusion_params = []
        for sensor in config.DATA.input:
            fusion_params += list(pipeline.fuse_pipeline._fusion_network[sensor].parameters())
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
    best_iou_filt = 0. # best filtered 
    is_best_filt = False

    best_iou = dict()
    is_best = dict()
    for sensor in config.DATA.input:
        best_iou[sensor] = 0.
        is_best[sensor] = False

    # when using fuse sensors true make sure we use both sensors here in variable sensors
    sensors = config.DATA.input

    for epoch in range(0, config.TRAINING.n_epochs):

        workspace.log('Training epoch {}/{}'.format(epoch, config.TRAINING.n_epochs),
                                  mode='train')

        pipeline.train() # need to change! Check so that gradients can pass!

        if config.ROUTING.do:
            raise NotImplementedError
        if config.FUSION_MODEL.fixed:
            pipeline.fuse_pipeline._fusion_network.eval()
        if config.FILTERING_MODEL.fixed:
            pipeline.filter_pipeline._filtering_network.eval()


        # resetting databases before each epoch starts
        train_database.reset()
        val_database.reset()

        # I want to handle the plotting of training data in a more elegant way - probably best to include this
        # functionality in the workspace so that it can do the writing of the appropriate properties
        train_loss = 0
        grad_norm = 0
        grad_norm_feature = 0
        val_norm = 0
        l1_interm = 0
        l1_grid = 0
        l1_grid_dict = dict()
        l_occ_dict = dict()
        for sensor_ in config.DATA.input:
            l1_grid_dict[sensor_] = 0
            l_occ_dict[sensor_] = 0

        l1_gt_grid = 0
        l_feat = 0
        
        l_occ = 0 # single sensor training

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
            # randomly integrate the selected sensors
            random.shuffle(sensors)
            for sensor in sensors:
                # print(sensor)
                batch['depth'] = batch[sensor + '_depth']
                # batch['confidence_threshold'] = eval('config.ROUTING.threshold_' + sensor) # not relevant to use anymore
                batch['mask'] = batch[sensor + '_mask']
                batch['sensor'] = sensor
                output = pipeline(batch, train_database, device)

                # optimization
                if output is None: # output is None when no valid indices were found for the filtering net within the random
                # bbox within the bounding volume of the integrated indices
                    print('output None from pipeline')
                    # break
                    continue

                if output == 'save_and_exit':
                    print('Found alpha nan. Save and exit')
                    workspace.save_model_state({'pipeline_state_dict': pipeline.state_dict(),
                                'epoch': epoch},
                               is_best_filt=is_best_filt, is_best=is_best)
                    return

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

                if len(config.DATA.input) > 0:
                    for sensor_ in config.DATA.input:
                        if output['l1_grid_' + sensor_] is not None:
                            l1_grid_dict[sensor_] += output['l1_grid_' + sensor_].item() 
                if config.FILTERING_MODEL.model == 'mlp' and config.FILTERING_MODEL.setting == 'translate'  \
                            and config.FILTERING_MODEL.MLP_MODEL.occ_head:
                    for sensor_ in config.DATA.input:
                        if output['l_occ_' + sensor_] is not None:
                            l_occ_dict[sensor_] += output['l_occ_' + sensor_].item() 
                    if output['l_occ'] is not None:
                        l_occ += output['l_occ'].item() 

                output['loss'].backward()
                # break

            del batch

            for name, param in pipeline.named_parameters():
                if param.grad is not None:
                    if (i + 1) % config.OPTIMIZATION.accumulation_steps == 0 or i == n_batches - 1:
                        if name.startswith('fuse_pipeline._feature'):
                            grad_norm_feature += torch.norm(
                            param.grad)
                        else:
                            grad_norm += torch.norm(
                            param.grad)
                        # print(torch.norm(param.grad))
                    # optimizer.zero_grad() # REMOVE LATER!
                    val_norm += torch.norm(param)
                    # print('grad norm: ', torch.norm(param.grad))
                    # print('val norm: ' , torch.norm(param))
                # if name.startswith('fuse_pipeline._feature'):
                #     # print(name)
                #     if param.isnan().sum() > 0:
                #         print(name)
                #         print(param)
                #     # print('isnan sum: ', param.isnan().sum())
                #     if param.grad is None:
                #         print('None')   

                    # Note, gradients that have been not None at one time, will never
                    # be none again since the zero_Grad option just makes them zero again.
                    # In pytorch 1.7.1 there is the option to set the gradients to none again


            # optimizer_feature.zero_grad(set_to_none=True) 
            
                # print(name, param.grad)

            if (i + 1) % config.SETTINGS.log_freq == 0:
# 
                # if config.DATA.fusion_strategy == 'two_fusionNet': # TODO: split plotting into tof and stereo
                #     divide = 2*config.SETTINGS.log_freq
                # else:
                divide = config.SETTINGS.log_freq
                train_loss /= divide
                grad_norm /= divide
                grad_norm_feature /= divide
                val_norm /= divide
                # print('averaged grad norm: ', grad_norm)
                # print('averaged val norm: ', val_norm)
                l1_interm /= divide
                l1_grid /= divide
                l1_gt_grid /= divide
                for sensor_ in config.DATA.input:
                    l1_grid_dict[sensor_] /= divide
                    l_occ_dict[sensor_] /= divide

                l_occ /= divide

                # l_occ /= divide
                # l_feat /= divide
                workspace.writer.add_scalar('Train/loss', train_loss, global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/grad_norm', grad_norm, global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/grad_norm_feature', grad_norm_feature, global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/val_norm', val_norm, global_step=i + 1 + epoch*n_batches)
                # workspace.writer.add_scalar('Train/lr_filt', get_lr(optimizer_filt), global_step=i + 1 + epoch*n_batches)
                # workspace.writer.add_scalar('Train/lr_fusion', get_lr(optimizer_fusion), global_step=i + 1 + epoch*n_batches)
                # workspace.writer.add_scalar('Train/loss_coeff', loss_coeff(i), global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/l1_interm', l1_interm, global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/l1_translation', l1_grid, global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/l1_gt_translation', l1_gt_grid, global_step=i + 1 + epoch*n_batches)
                for sensor_ in config.DATA.input:
                    workspace.writer.add_scalar('Train/l1_' + sensor_, l1_grid_dict[sensor_], global_step=i + 1 + epoch*n_batches)
                    workspace.writer.add_scalar('Train/occ_loss_' + sensor_, l_occ_dict[sensor_], global_step=i + 1 + epoch*n_batches)

                workspace.writer.add_scalar('Train/occ_loss', l_occ, global_step=i + 1 + epoch*n_batches)
                # workspace.writer.add_scalar('Train/feat_loss', l_feat, global_step=i + 1 + epoch*n_batches)
                train_loss = 0
                grad_norm = 0
                grad_norm_feature = 0
                val_norm = 0
                l1_interm = 0
                l1_grid = 0
                l1_grid_dict = dict()
                l_occ_dict = dict()
                for sensor_ in config.DATA.input:
                    l1_grid_dict[sensor_] = 0
                    l_occ_dict[sensor_] = 0


                l1_gt_grid = 0
                l_feat = 0
                l_occ = 0 # single sensor training

            if config.TRAINING.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(pipeline.parameters(),
                                                   max_norm=1.,
                                                   norm_type=2)

            if (i + 1) % config.OPTIMIZATION.accumulation_steps == 0 or i == n_batches - 1:
                if config.FILTERING_MODEL.features_to_sdf_enc or config.FILTERING_MODEL.features_to_weight_head:
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
                if config.FILTERING_MODEL.features_to_sdf_enc or config.FILTERING_MODEL.features_to_weight_head:
                    optimizer_feature.zero_grad()
                if not config.FILTERING_MODEL.fixed:
                    optimizer_filt.zero_grad()
                if not config.FUSION_MODEL.fixed:    
                    optimizer_fusion.zero_grad()

                pipeline.eval()


                pipeline.test(val_loader, val_dataset, val_database, sensors, device)

                # val_database.filter(value=1.) # the more frames you integrate, the higher can the value be
                val_eval, val_eval_fused = val_database.evaluate(mode='val', workspace=workspace)

                for sensor in config.DATA.input:
                    workspace.writer.add_scalar('Val/mse_' + sensor, val_eval[sensor]['mse'], global_step=i + 1 + epoch*n_batches)
                    workspace.writer.add_scalar('Val/acc_' + sensor, val_eval[sensor]['acc'], global_step=i + 1 + epoch*n_batches)
                    workspace.writer.add_scalar('Val/iou_' + sensor, val_eval[sensor]['iou'], global_step=i + 1 + epoch*n_batches)
                    workspace.writer.add_scalar('Val/mad_' + sensor, val_eval[sensor]['mad'], global_step=i + 1 + epoch*n_batches)

                workspace.writer.add_scalar('Val/mse_fused', val_eval_fused['mse'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/acc_fused', val_eval_fused['acc'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/iou_fused', val_eval_fused['iou'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/mad_fused', val_eval_fused['mad'], global_step=i + 1 + epoch*n_batches)



                # check if current checkpoint is best
                if val_eval_fused['iou'] >= best_iou_filt:
                    is_best_filt = True
                    best_iou_filt = val_eval_fused['iou']
                    workspace.log('found new best model overall with iou {} at epoch {}'.format(best_iou_filt, epoch),
                                  mode='val')
       
                else:
                    is_best_filt = False

                for sensor in config.DATA.input:
                    if val_eval[sensor]['iou'] >= best_iou[sensor]:
                        is_best[sensor] = True
                        best_iou[sensor] = val_eval[sensor]['iou']
                        workspace.log('found new best ' + sensor + ' model with iou {} at epoch {}'.format(best_iou[sensor], epoch),
                                      mode='val')
     
                    else:
                        is_best[sensor] = False

                # save models
                # train_database.save_to_workspace(workspace, mode='latest_train', save_mode=config.SETTINGS.save_mode)
                # val_database.save_to_workspace(workspace, is_best, is_best_tof, is_best_stereo, save_mode=config.SETTINGS.save_mode)

                # save checkpoint
                workspace.save_model_state({'pipeline_state_dict': pipeline.state_dict(),
                                                'epoch': epoch},
                                               is_best_filt=is_best_filt, is_best=is_best)

                pipeline.train() # CHANGE CHECK
                if config.ROUTING.do:
                    pipeline.fuse_pipeline._routing_network_tof.eval()
                    pipeline.fuse_pipeline._routing_network_stereo.eval()
                if config.FUSION_MODEL.fixed:
                    pipeline.fuse_pipeline._fusion_network.eval()
                if config.FILTERING_MODEL.fixed:
                    pipeline.filter_pipeline._filtering_network.eval()



            # if i == 6: # for debugging
                # break
        

if __name__ == '__main__':

    args = arg_parser()
    train_fusion(args)
