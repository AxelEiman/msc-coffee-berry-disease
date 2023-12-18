from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, yaml_model_load
from ultralytics.yolo.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, ROOT, is_git_dir, yaml_load
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_pip_update_available, check_yaml
from ultralytics.yolo.utils.torch_utils import smart_inference_mode
from ultralytics.yolo.cfg import get_cfg

from tasks import DetectionPointModel
from train import DetectionPointTrainer
from val import DetectionPointValidator
#from predict import DetectionPointPredictor
from ultralytics.yolo.v8.detect import DetectionPredictor


from typing import Union
from pathlib import Path
import sys


TASK_MAP = {
    'detect': [
        DetectionPointModel, DetectionPointTrainer, DetectionPointValidator,
        DetectionPredictor],
    }

class YOPO_PT(YOLO):
    """YOPO_PT (You Only Point Once) detection model"""

    def __init__(self, model: Union[str, Path] = 'yolov8n.pt', task='detect') -> None:
        super().__init__(model, task)
        
    def _new(self, cfg: str, task=None, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            task (str | None): model task
            verbose (bool): display model info on load
        """
        
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = TASK_MAP[self.task][0](cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides['model'] = self.cfg

        # Below added to allow export from yamls
        args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine model and default args, preferring model args
        self.model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
        self.model.task = self.task
        print(f'New YOPO object initialized')

    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """
        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args['task']
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides['model'] = weights
        self.overrides['task'] = self.task

    def train(self, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        self._check_is_pytorch_model()
        if self.session:  # Ultralytics HUB session
            if any(kwargs):
                LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
            kwargs = self.session.train_args
        check_pip_update_available()
        overrides = self.overrides.copy()
        if kwargs.get('cfg'):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs['cfg']))
        overrides.update(kwargs)
        overrides['mode'] = 'train'
        if not overrides.get('data'):
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        if overrides.get('resume'):
            overrides['resume'] = self.ckpt_path
        self.task = overrides.get('task') or self.task
        self.trainer = TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)
        if not overrides.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in (-1, 0):
            self.model, _ = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP

    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        """
        Validate a model on a given dataset.

        Args:
            data (str): The dataset to validate on. Accepts all formats accepted by yolo
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        overrides = self.overrides.copy()
        overrides['rect'] = True  # rect batches as default
        overrides.update(kwargs)
        overrides['mode'] = 'val'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        if 'task' in overrides:
            self.task = args.task
        else:
            args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(self.model, (str, Path)):
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)

        validator = TASK_MAP[self.task][2](args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics

        return validator.metrics

    @smart_inference_mode()
    def predict(self, source=None, stream=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The prediction results.
        """
        if source is None:
            source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
        is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')) and any(
            x in sys.argv for x in ('predict', 'track', 'mode=predict', 'mode=track'))
        overrides = self.overrides.copy()
        overrides['conf'] = 0.25   # 0.25 in original
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        if not is_cli:
            overrides['save'] = kwargs.get('save', False)  # do not save by default if called in Python
        if not self.predictor:
            self.task = overrides.get('task') or self.task
            self.predictor = TASK_MAP[self.task][3](overrides=overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
            if 'project' in overrides or 'name' in overrides:
                self.predictor.save_dir = self.predictor.get_save_dir()
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)
