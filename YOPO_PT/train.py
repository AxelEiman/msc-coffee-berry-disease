from ultralytics.yolo.v8.detect import DetectionTrainer, DetectionValidator
from ultralytics.yolo.utils import LOGGER, RANK, TQDM_BAR_FORMAT, colorstr, ops
from ultralytics.yolo.utils.torch_utils import de_parallel

from dataset import build_point_yolo_dataset
from tasks import DetectionPointModel
#import metrics# import box_iou    # TODO check if any changes to this function, otherwise import from ultralytics
from val import DetectionPointValidator

import time
import torch
import numpy as np
from copy import copy
from copy import deepcopy
from tqdm import tqdm
from torch import distributed as dist
from point_match_pgcp import tensor_delete, point_matcher, point_guided_copy_paste
from custom_nms import non_max_suppression
from plotting import plot_results

class DetectionPointTrainer(DetectionTrainer): 
    
    def build_dataset(self, img_path, mode='train', batch=None):
        """Build point YOLO dataset
        
        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): size of batches, this is for 'rect'. Defaults to None.
        """
        #print(f'building dataset. mode: {mode}')
        #print(f'args: {type(self.args)}')
        #with open(f'{mode}_args.txt', 'w') as f:
        #    for item in self.args:
        #        f.write('='.join([str(word) for word in item]) + '\n')

        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)    # Sets a stride? 32 or larger
        return build_point_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode=='val', stride=gs) #mode=='val'
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionPointModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    
    def preprocess_batch(self, batch):
            """Preprocesses a batch of images by scaling and converting to float."""
            batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
            # TODO standardize, (img-mean)/std
            return batch    
    
    def get_validator(self):
        """Returns a DetectionPointValidator for YOLO model validation."""

        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'pw_MIL', 'iw_MIL'
        return DetectionPointValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    
    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (5 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'm/p[%]', 'Instances', 'Size')

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)

        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs *
                       nb), 100) if self.args.warmup_epochs > 0 else -1  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        print('\n Creating a teacher model... ')
        self.teacher = deepcopy(self.model)   #TODO: fix to implement earlier with model maybe.
        self.teacher.model._modules['22'].training = False  # see the if-statement in point_head.py
        gt_labels_library = torch.from_numpy(np.loadtxt(self.data['train'] + '/gt_labels.txt')) # loads the gt_labels once
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with torch.cuda.amp.autocast(self.amp):     
                    batch = self.preprocess_batch(batch)
                    batch['points'] = torch.tensor([]).view(-1, 4)
                    batch['batch_idx_points'] = torch.tensor([])
                    batch['cls_points'] = torch.tensor([]).view(-1, 1)
                    
                    # Teacher
                    
                    if i > 0 or epoch > 0:
                        #self.teacher.to(gpu_device)
                        self.teacher.state_dict = self.ema.ema.state_dict   # TODO: is this correct!?
                        #self.model.to(cpu_device)
                    with torch.no_grad():
                        preds, _ , sP_preds = self.teacher(batch['img'])
                    preds = preds.to('cpu')
                    sP_preds = sP_preds.to('cpu')
                    s_preds = preds[:, -self.data['nc']:, :]

                    gts = deepcopy(gt_labels_library)
                    #NOTE: this is a custom NMS which also outputs a mask.
                    preds, mask = non_max_suppression(preds,
                                    0.25,   # TODO: this is the confidence threshold. This one should be dyna,ic and optimised with e.g. F1-score.
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    classes=self.args.classes)
                    batched_matched_points = 0.
                    #TODO: Change back and forth between CPU and GPU? Tried with the comments, but did not help.
                    #mask = mask.to(cpu_device)
                    for image_idx, image_preds in enumerate(preds):
                        #image_preds = image_preds.to(cpu_device)
                        matched_points = 0.
                        image_gts = torch.concatenate((batch['cls'][batch['batch_idx'] == image_idx].view(-1, 1), batch['bboxes'][batch['batch_idx'] == image_idx]), 1)  # check that these are indeed in the order box_gts, point_gts
                        image_gts = torch.concatenate((image_gts[image_gts[:, 3] != 0], image_gts[image_gts[:, 3] == 0]), 0)
                        
                        if image_preds.size(0) > 0 and image_gts[image_gts[:, 3] == 0].size(0) > 0: # true if there are predictions and gts points in image
                            if s_preds.shape[1] > 1:
                                s = s_preds[image_idx, :, mask[image_idx]].float().softmax(-2)
                            else:
                                s = torch.ones((self.data['nc'], mask[image_idx].sum()))
                            matched_points -= image_gts.size(0)
                            image_gts = point_matcher(torch.concatenate((image_preds[:, 5].unsqueeze(1), image_preds[:, :4] / batch['resized_shape'][0][0]), 1).unsqueeze(0), # divide with imagesize since one wants everything in the span [0,1]
                                                     image_gts, s, sP_preds[image_idx, :, mask[image_idx]].float().softmax(-2))
                            matched_points += image_gts.size(0)
                        batched_matched_points += matched_points

                        #else:
                        #    print(f'\nNo point_matching... \n   Predictions: {image_preds.size(0)} \n   GT points: {image_gts[image_gts[:, 3] == 0].size(0)}\n')
                        if image_gts[image_gts[:, 3] == 0].size(0) > 0: # true if there are gts points in image
                            image_gts, gts = point_guided_copy_paste(image_gts, gts)
                        #else:
                        #    print(f'\nNo point-guided copy-paste... Predictions: {image_preds.size(0)} \n GT points: {image_gts[image_gts[:, 3] == 0].size(0)}\n')
                        #NOTE: this mess separetes the gt boxes and pts. Allows the model to train on only the boxes while the pts can be reached in the loss as well.
                        batch['bboxes'] = torch.concatenate((batch['bboxes'][batch['batch_idx'] < image_idx].view(-1,4), image_gts[image_gts[:, 3] != 0][:, 1:], batch['bboxes'][batch['batch_idx'] > image_idx].view(-1,4)), 0)
                        batch['cls'] = torch.concatenate((batch['cls'][batch['batch_idx'] < image_idx].view(-1,1), image_gts[image_gts[:, 3] != 0][:, 0].view(-1,1), batch['cls'][batch['batch_idx'] > image_idx].view(-1,1)), 0)
                        batch['batch_idx'] = torch.concatenate((batch['batch_idx'][batch['batch_idx'] < image_idx], torch.zeros(image_gts[image_gts[:, 3] != 0].size(0)) + image_idx, batch['batch_idx'][batch['batch_idx'] > image_idx]), 0)
                        batch['points'] = torch.concatenate((batch['points'], image_gts[image_gts[:, 3] == 0][:, 1:]), 0)
                        batch['cls_points'] = torch.concatenate((batch['cls_points'], image_gts[image_gts[:, 3] == 0][:, 0].view(-1,1)), 0)
                        batch['batch_idx_points'] = torch.concatenate((batch['batch_idx_points'], torch.zeros(image_gts[image_gts[:, 3] == 0].size(0)) + image_idx, ), 0)
                    
                    # Student
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (3 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batched_matched_points/mask.sum(), batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')
    
    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png




