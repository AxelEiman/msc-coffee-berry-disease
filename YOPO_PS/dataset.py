from ultralytics.yolo.data.dataset import YOLODataset
from ultralytics.yolo.data.utils import HELP_URL, get_hash, img2label_paths, verify_image_label
from ultralytics.yolo.data.augment import Compose, Format, LetterBox
from ultralytics.yolo.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, colorstr, is_dir_writeable

from augment import PointInstances, v8_point_transforms

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tqdm import tqdm
import numpy as np

DATASET_CACHE_VERSION = '1.0.3'

def build_point_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    """
        Builds dataset for points
    """
    return PointYoloDataset(
        img_path=img_path,              # path to image folder (str?)
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',
        hyp=cfg,                        # Hyperparameters? ("probably add a get_hyps_from_cfg function")
        rect=cfg.rect or rect,          # Rectangular batches (?)
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        use_segments=cfg.task == 'segment',
        use_keypoints=cfg.task == 'pose',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0
    )


def save_dataset_cache_file(prefix, path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x['version'] = DATASET_CACHE_VERSION  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
        LOGGER.info(f'{prefix}New cache created: {path}')
    else:
        LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')

class PointYoloDataset(YOLODataset):
    #def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
    #    super().__init__(*args, data=data, use_segments=use_segments, use_keypoints=use_keypoints, **kwargs)

    
        
    def cache_labels(self, path=Path('./labels.cache')):     # Outdated as of 20/10/2023 apparently. self.cache_version is nonexistent
        """Cache dataset labels, check images and read shapes.
        Saves a labels.cache file with label info as in the dictionary 'x' below, also returns that dictionary
        Args:
            path (Path): path where to save the cache file
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, [] # number missing, found, empty, corrupt messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}' # Think this is just a string describing the search through 
        total = len(self.im_files)
        nkpt, ndim = self.data.get('kpt_shape', (0,0)) # Gets number of keypoints and dim? TODO points in data(set?).get and here
        
        # Assert keypoints in correct dimension if using kpts.
        if self.use_keypoints and (nkpt<=0 or ndim not in (2,3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        
        with ThreadPool(NUM_THREADS) as pool:
            # I think this parrallelly maps func to files, verifying that all files are correct format etc.
            # TODO Think we may need to add points here and in verify_image_label. Or just rely on choosing the correct files might be fine
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.prefix),
                                             repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim)))
            
            pbar = tqdm(results, desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
            # Assume it returns somehting like files

            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                # increment number missing, found, empty, corrupt messages increment, as found in pbar? 
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    # Add labels for the image to x:
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:,0:1],      # n, 1  - All rows, and first element? why the slice though
                            bboxes=lb[:,1:],    # n, 4  - Boxes are the other four coordinates
                            segments=segments,  # TODO Not sure if these are just T/F or somehow contain labels?
                            keypoints=keypoint, # TODO add point labels or include in bboxes? if len==3? wh==0?
                            normalized=True,
                            bbox_format='xywh'
                        )
                    )

                if msg:
                    msgs.append(msg)

                pbar.desc = f'{desc} {nf} images, {nm+ne} backgrounds, {nc} corrupt'
            pbar.close()

            if msgs:
                LOGGER.info('\n'.join(msgs))
            if nf == 0:
                LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
            
            x['hash'] = get_hash(self.label_files + self.im_files)
            x['results'] = nf, nm, ne, nc, len(self.im_files)
            x['msgs'] = msgs # warnings
            x['version'] = self.cache_version

            if is_dir_writeable(path.parent):
                if path.exists():
                    path.unlink()   #remove *.cache file if exists
                np.save(str(path), x)   # save cache for next time
                path.with_suffix('.cache.npy').rename(path) # remove .npy suffix
                LOGGER.info(f'{self.prefix}New cache created: {path}')
            else:
                LOGGER.warning(f'{self.prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')
            return x
        
    

        
    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)   # Finds label path, presumably right next to im_files
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')

        # Try-except loads cache or creates one
        try:
            import gc
            gc.disable() # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
            cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True    # load dict
            gc.enable()
            assert cache['version'] == self.cache_version   # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False    # run cache ops 

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')    # found, missing, empty, corrupt, total

        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm+ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=self.prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)    # dislpay cache results
            
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))   # display warnings
        
        if nf == 0:
            raise FileNotFoundError(f'{self.prefix}No labels found in {cache_path}, can not start training. {HELP_URL}')
        
        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')] # remove items
        labels = cache['labels']
        self.im_files = [lb['im_file'] for lb in labels]    # update im_files (not sure what the point is here? guess it may update in some case, but the value comes from itself)

        # Check if the dataset is all boxes or all segments
        # TODO likely adapt for boxes+pts below. NOTE zeros seem to be found in lb['bboxes']
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)   # Number of instances (class, boxes)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))

        if len_segments and len_boxes != len_segments:
            # Labels of different types found
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        
        if len_cls == 0:
            raise ValueError(f'All labels empty in {cache_path}, can not start training without labels. {HELP_URL}')
        
        return labels
    
    def update_labels_info(self, label):
        """customize your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        keypoints = label.pop('keypoints')
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = PointInstances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label
    
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_point_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms
