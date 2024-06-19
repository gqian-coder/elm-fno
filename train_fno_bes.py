# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
from omegaconf import DictConfig
from math import ceil

from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler

from modulus.models.fno import FNO
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger, initialize_mlflow
from modulus.models.mlp.fully_connected import FullyConnected

from validator import GridValidator
import h5py
from utils import *

@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def bes_trainer(cfg: DictConfig) -> None:
    """Training for the 1D ELM problem.

    This training script demonstrates how to set up a data-driven model for a time-series 
    BES-ELM signal prediction problem using Fourier Neural Operators (FNO). 
    Training data is loaded from d3d_bes_elm_dataset. The training uses the first [T_0,...,T_n] 
    timesteps of data to predict the next [T_n+1, T_n+m] timesteps, where m < n. 
    Then combines the [T_n+1, T_n+m] with the previous [T_m, T_n] data to form the new series of 
    data, using it to predict [T_n+m, T_n+2m] timesteps. 
    Each 1D data consists of 64 channels. 
    """
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="bes_fno")
    log.file_logging()
    initialize_mlflow(
        experiment_name=f"temporal_FNO",
        experiment_desc=f"training an FNO model for the BES-ELM problem",
        run_name=f"BES-ELM FNO training",
        run_desc=f"training FNO for BES signal prediction",
        user_name="Gretchen Ross",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)  # Modulus launch logger

    # define model, loss, optimiser, scheduler, data loader
    fno = FNO(
        in_channels=cfg.bes.observation_ts,
        out_channels=cfg.bes.prediction_ts,
        decoder_layers=cfg.arch.decoder.layers,
        decoder_layer_size=cfg.arch.decoder.layer_size,
        dimension=cfg.arch.fno.dimension,
        latent_channels=cfg.arch.fno.latent_channels,
        num_fno_layers=cfg.arch.fno.fno_layers,
        num_fno_modes=cfg.arch.fno.fno_modes,
        padding=cfg.arch.fno.padding,
    )
    #model     = pred_FNO(fno, in_features=cfg.bes.observation_ts, out_features=cfg.bes.prediction_ts).to(dist.device)
    model     = fno
    loss_fun  = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
    )
    
    obsv_ts     = cfg.bes.observation_ts 
    pred_ts     = cfg.bes.prediction_ts 
    n_channels  = cfg.bes.n_channels
    dataset     = BESDataset(cfg.bes.dataset, cfg.bes.eventsep, n_channels, obsv_ts, pred_ts, cfg.training.perc_data, "std")
    dataloader  = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=1)
    testset     = BESDataset(cfg.bes.dataset, cfg.bes.eventsep, n_channels, obsv_ts, pred_ts, cfg.training.perc_data-1.0, "std")
    testloader  = DataLoader(testset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=1)
    validset    = BESDataset(cfg.bes.dataset, cfg.bes.eventsep, n_channels, obsv_ts, pred_ts, cfg.validation.perc_data, "std")
    validloader = DataLoader(validset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=1)

    print("training data length: ", dataset.__len__())
    print("validation data length: ", validset.__len__())
    print("testing data length: ", testset.__len__())
    dlen_test = testset.__get_len__(cfg.training.perc_data-1.0)
    #print("dlen of testset = ", dlen_test)
    nx = ny = int(np.sqrt(n_channels))
    pred_test  = np.zeros([dlen_test, nx, ny])
    targ_test  = np.zeros([dlen_test, nx, ny])
    dlen_valid = validset.__get_len__(cfg.validation.perc_data)
    pred_valid = np.zeros([dlen_valid, nx, ny])
    targ_valid = np.zeros([dlen_valid, nx, ny])

    '''
    dataloader = Darcy2D(
        resolution=cfg.training.resolution,
        batch_size=cfg.training.batch_size,
        normaliser=normaliser,
    )
    '''
    validator = GridValidator(loss_fun=MSELoss(reduction="mean"))

    ckpt_args = {
        "path": f"./checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_pseudo_epoch = 0#load_checkpoint(device=dist.device, **ckpt_args)

    # calculate steps per pseudo epoch
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )
    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }
    if cfg.training.pseudo_epoch_sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased pseudo_epoch_sample_size to multiple of \
                      batch size: {steps_per_pseudo_epoch*cfg.training.batch_size}"
        )
    if cfg.validation.sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased validation sample size to multiple of \
                      batch size: {validation_iters*cfg.training.batch_size}"
        )

    # define forward passes for training and inference
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(model, invars, target):
        pred = model(invars)
        #print(pred.shape)
        loss = loss_fun(pred, target)
        target = target.cpu().numpy()[0, 0, :, :]
        pred   = pred.detach().cpu().numpy()[0, 0, :, :]
        print("predict: ", pred)
        print("target: ", target)
        print("loss: ", loss.detach().cpu().numpy())
        return loss

    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        return model(invars)

    if loaded_pseudo_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch+1}.")

    log.success("steps_per_pseudo_epoch = {}".format(steps_per_pseudo_epoch))
    for pseudo_epoch in range(
        max(1, loaded_pseudo_epoch + 1), cfg.training.max_pseudo_epochs + 1
    ):
        log.success("pseudo_epoch = {}, total epochs = {}".format(pseudo_epoch, cfg.training.max_pseudo_epochs))
        # Wrap epoch in launch logger for console / MLFlow logs
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            for _, batch in zip(range(steps_per_pseudo_epoch), dataloader):
                log.success("batch {}, total step {} ".format(_, steps_per_pseudo_epoch))
                #print("data shape: ", batch["data"].shape)
                #print("target shape: ", batch["target"].shape)
                loss = forward_train(model, batch["data"], batch["target"])
                logger.log_minibatch({"loss": loss.detach()})
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        print("epoch {}, loss = {}".format(pseudo_epoch, loss))
        # save checkpoint
        #if pseudo_epoch % cfg.training.rec_results_freq == 0:
        #    save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # validation step
        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                for _, batch in zip(range(validation_iters), dataloader):
                    val_loss = validator.compareBES(
                        batch["data"],
                        batch["target"],
                        forward_eval(batch["data"]),
                        pseudo_epoch,
                        logger,
                    )
                    total_loss += val_loss
                logger.log_epoch({"Validation error": total_loss / validation_iters})

            # update learning rate
        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()

        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            # validation 
            ts = 0
            for batch in validloader:
                pred = forward_eval(batch["data"])
                #print(pred.detach().cpu().numpy().shape, batch["target"].cpu().numpy().shape, ts, pred_ts)
                batch_len = pred.detach().cpu().numpy().shape[0]
                pred_valid[ts:ts+batch_len, :, :] = pred.detach().cpu().numpy()[:,0,:,:]
                targ_valid[ts:ts+batch_len, :, :] = batch["target"].cpu().numpy()[:,0,:,:]
                ts += batch_len 
                if (ts%256==0):
                    print(batch_len)
                    print("validation step {}/{}".format(ts, validset.__len__()))
            pred_valid.tofile("predicted_signals_validation_epoch_" + str(pseudo_epoch) + ".bin")
            targ_valid.tofile("real_signals_validation_epoch_" + str(pseudo_epoch) + ".bin")

            # testing
            ts = 0
            for batch in testloader:
                pred = forward_eval(batch["data"])
                #print(pred.detach().cpu().numpy().shape, batch["target"].cpu().numpy().shape, ts, pred_ts)
                batch_len = pred.detach().cpu().numpy().shape[0]
                pred_test[ts:ts+batch_len, :, :] = pred.detach().cpu().numpy()[:,0,:,:]
                targ_test[ts:ts+batch_len, :, :] = batch["target"].cpu().numpy()[:,0,:,:]
                ts += batch_len 
                if (ts%256==0):
                    print("testing step {}/{}".format(ts, testset.__len__()))
            pred_test.tofile("predicted_signals_test_epoch_" + str(pseudo_epoch) + ".bin")
            targ_test.tofile("real_signals_test_epoch_" + str(pseudo_epoch) + ".bin")

    #save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs)
    log.success("Training completed *yay*")

if __name__ == "__main__":
    bes_trainer()
