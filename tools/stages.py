import os
import re
from argparse import Namespace
from pathlib import Path

import torch
import transformers
from dlhpcstarter.trainer import trainer_instance
from dlhpcstarter.utils import (
    get_test_ckpt_path,
    importer,
    resume_from_ckpt_path,
    write_test_ckpt_path,
)
from lightning.pytorch import seed_everything


def stages(args: Namespace):
    """
    Handles the training and testing stages for the task. This is the stages() function
        defined in the task's stages.py.

    Argument/s:
        args - an object containing the configuration for the job.
    """
    print(args.dataset_dir)
    
    args.warm_start_modules = False

    # Set seed number (using the trial number) for deterministic training
    seed_everything(args.trial, workers=True)

    if torch.cuda.is_available():
        print(f'Device capability: {torch.cuda.get_device_capability()}')
        if args.float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(args.float32_matmul_precision)

    # Model definition:
    TaskModel = importer(definition=args.definition, module=args.module)

    # Trainer:
    trainer = trainer_instance(**vars(args))

    # Train:
    if args.train:

        # Resume from checkpoint:
        ckpt_path = resume_from_ckpt_path(args.exp_dir_trial, args.resume_last, args.resume_epoch, args.resume_ckpt_path) if not args.fast_dev_run else None

        # Warm start from checkpoint if not resuming:
        if args.warm_start_ckpt_path and ckpt_path is None and not args.fast_dev_run:
            print('Warm-starting using: {}.'.format(args.warm_start_ckpt_path))

            if args.warm_start_optimiser:
                model = TaskModel(**vars(args))
                trainer.fit(model, ckpt_path=args.warm_start_ckpt_path)                
            else:                    
                strict = args.warm_start_ckpt_path_strict if args.warm_start_ckpt_path_strict is not None else True
                model = TaskModel.load_from_checkpoint(checkpoint_path=args.warm_start_ckpt_path, strict=strict, **vars(args))
                if args.allow_warm_start_optimiser_partial:
                    assert not strict
                    model.warm_start_optimiser_partial = True
                trainer.fit(model)

        # Resume training from ckpt_path:
        elif ckpt_path is not None:
            model = TaskModel(**vars(args))
            trainer.fit(model, ckpt_path=ckpt_path)

        # Let the module warm start itself if ckpt_path is None:
        else:
            args.warm_start_modules = True
            model = TaskModel(**vars(args))
            trainer.fit(model)

    # Test:
    if args.test:

        args.warm_start_modules = False
        model = TaskModel(**vars(args))
        
        ckpt_path = None
        if args.test_ckpt_name and not args.test_without_ckpt:
            assert 'model' not in locals(), 'if "test_ckpt_name" is defined in the config, it will overwrite the model checkpoint that has been trained.'
            hf_ckpt = transformers.AutoModel.from_pretrained(args.test_ckpt_name, trust_remote_code=True)
            model.encoder_decoder.load_state_dict(hf_ckpt.state_dict())
        elif not args.fast_dev_run and not args.test_without_ckpt:

            if args.other_exp_dir:

                # The experiment trial directory of the other configuration:
                other_exp_dir_trial = os.path.join(args.other_exp_dir, f'trial_{args.trial}')

                ckpt_path = get_test_ckpt_path(
                    other_exp_dir_trial, args.other_monitor, args.other_monitor_mode, 
                )
            
            else:

                # Get the path to the best performing checkpoint
                ckpt_path = get_test_ckpt_path(
                    args.exp_dir_trial, args.monitor, args.monitor_mode, args.test_epoch, args.test_ckpt_path,
                )

            print('Testing checkpoint: {}.'.format(ckpt_path))
            write_test_ckpt_path(ckpt_path, args.exp_dir_trial)

            # Work-around as trainer.current_epoch cannot be set:
            model.ckpt_epoch = int(re.search(r'epoch=(\d+)-step=', ckpt_path).group(1))
        
        trainer.test(model, ckpt_path=ckpt_path)
        