import os
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import gc

import transformers
import wandb

import gym
import wrappers as wrappers

import absl.app
import absl.flags
from flax.training.early_stopping import EarlyStopping
from flaxmodels.flaxmodels.lstm.lstm import LSTMRewardModel
from flaxmodels.flaxmodels.gpt2.trajectory_gpt2 import TransRewardModel

from JaxPref.sampler import TrajSampler
from JaxPref.jax_utils import batch_to_jax
import JaxPref.reward_transform as r_tf
from JaxPref.model import FullyConnectedQFunction
from viskit.logging import logger, setup_logger
from JaxPref.MR import MR
from JaxPref.replay_buffer import get_d4rl_dataset, index_batch
from JaxPref.NMR import NMR
from JaxPref.PrefTransformer import PrefTransformer
from JaxPref.utils import Timer, define_flags_with_default, set_random_seed, get_user_flags, prefix_metrics, \
    WandBLogger, save_pickle


os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS_DEF = define_flags_with_default(
    env='chess',
    model_type='PrefTransformer',
    max_traj_length=1000,
    seed=5,
    data_seed=5,
    save_model=True,
    batch_size=128,
    early_stop=True,
    min_delta=1e-3,
    patience=10,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    reward_arch='256-256',
    orthogonal_init=False,
    activations='relu',
    activation_final='none',
    training=True,

    n_epochs=1000,
    eval_period=5,

    data_dir='/home/hail/PreferenceTransformer/human_label/',
    num_query=2500,
    query_len=50,
    skip_flag=0,
    balance=False,
    topk=10,
    window=2,
    use_human_label=True,
    feedback_random=False,
    feedback_uniform=False,
    enable_bootstrap=False,

    comment='new_cel_2',

    robosuite=False,
    robosuite_dataset_type="ph",
    robosuite_dataset_path='./data',
    robosuite_max_episode_steps=500,

    reward=MR.get_default_config(),
    transformer=PrefTransformer.get_default_config(),
    lstm=NMR.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(_):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    save_dir = FLAGS.logging.output_dir + '/' + FLAGS.env
    save_dir += '/' + str(FLAGS.model_type) + '/'

    FLAGS.logging.group = f"{FLAGS.env}_{FLAGS.model_type}"
    assert FLAGS.comment, "You must leave your comment for logging experiment."
    FLAGS.logging.group += f"_{FLAGS.comment}"
    FLAGS.logging.experiment_id = FLAGS.logging.group + f"_s{FLAGS.seed}"
    save_dir += f"{FLAGS.comment}" + "/"
    save_dir += 's' + str(FLAGS.seed)

    setup_logger(
        variant=variant,
        seed=FLAGS.seed,
        base_log_dir=save_dir,
        include_exp_prefix_sub_dir=False
    )

    FLAGS.logging.output_dir = save_dir
    wb_logger = WandBLogger(FLAGS.logging, variant=variant)

    set_random_seed(FLAGS.seed)

    # env = chess_v6.env()
    # env.reset()
    # set_random_seed(FLAGS.data_seed)
    # set_random_seed(FLAGS.seed)

    observation_dim = 7616
    action_dim = 4672
    criteria_key = None

    data_size = 500
    interval = int(data_size / FLAGS.batch_size) + 1
    early_stop = EarlyStopping(min_delta=FLAGS.min_delta, patience=FLAGS.patience)

    if FLAGS.model_type == "MR":
        rf = FullyConnectedQFunction(observation_dim, action_dim, FLAGS.reward_arch, FLAGS.orthogonal_init,
                                     FLAGS.activations, FLAGS.activation_final)
        reward_model = MR(FLAGS.reward, rf)

    elif FLAGS.model_type == "PrefTransformer":
        total_epochs = FLAGS.n_epochs
        config = transformers.GPT2Config(
            **FLAGS.transformer
        )
        config.warmup_steps = int(total_epochs * 0.1 * interval)
        config.total_steps = total_epochs * interval

        trans = TransRewardModel(config=config, observation_dim=observation_dim, action_dim=action_dim,
                                 activation=FLAGS.activations, activation_final=FLAGS.activation_final)
        with open("/home/hail/PreferenceTransformer/JaxPref/reward_model/chess/PrefTransformer/new_cel/s5/model.pkl", 'rb') as f:
            reward_model_=  pickle.load(f)
            reward_model = reward_model_['reward_model']
        # reward_model = PrefTransformer(config, trans)

    elif FLAGS.model_type == "NMR":
        total_epochs = FLAGS.n_epochs
        config = transformers.GPT2Config(
            **FLAGS.lstm
        )
        config.warmup_steps = int(total_epochs * 0.1 * interval)
        config.total_steps = total_epochs * interval

        lstm = LSTMRewardModel(config=config, observation_dim=observation_dim, action_dim=action_dim,
                               activation=FLAGS.activations, activation_final=FLAGS.activation_final)
        reward_model = NMR(config, lstm)

    if FLAGS.model_type == "MR":
        train_loss = "reward/rf_loss"
    elif FLAGS.model_type == "NMR":
        train_loss = "reward/lstm_loss"
    elif FLAGS.model_type == "PrefTransformer":
        train_loss = "reward/trans_loss"

    with open("/media/hail/HDD/Chess_data/Classical/convert_train_eval/OO_CEL_batch_01.pkl", 'rb') as f:
        pref_dataset = pickle.load(f)

    with open("/media/hail/HDD/Chess_data/Classical/convert_train_eval/OE_CEL_batch_01.pkl", 'rb') as f:
         eval_dataset = pickle.load(f)

    eval_data_size = 250
    eval_interval = int(eval_data_size / FLAGS.batch_size) + 1

    for epoch in range(FLAGS.n_epochs + 1):
        metrics = defaultdict(list)
        metrics['epoch'] = epoch
        if epoch:
            shuffled_idx = np.random.permutation(500)
            for i in range(interval):
                start_pt = i * FLAGS.batch_size
                end_pt = min((i + 1) * FLAGS.batch_size,500)
                with Timer() as train_timer:
                    # train
                    batch = batch_to_jax(index_batch(pref_dataset, shuffled_idx[start_pt:end_pt]))
                    for key, val in prefix_metrics(reward_model.train(batch), 'reward').items():
                        metrics[key].append(val)
            metrics['train_time'] = train_timer()
        else:
            # for using early stopping with train loss.
            metrics[train_loss] = [float(FLAGS.query_len)]

            # eval phase
        if epoch % FLAGS.eval_period == 0:
            for j in range(eval_interval):
                eval_start_pt, eval_end_pt = j * FLAGS.batch_size, min((j + 1) * FLAGS.batch_size, 250)
                batch_eval = batch_to_jax(index_batch(eval_dataset, range(eval_start_pt, eval_end_pt)))
                for key, val in prefix_metrics(reward_model.evaluation(batch_eval), 'reward').items():
                    metrics[key].append(val)
            if not criteria_key:
                    criteria_key = key
            criteria = np.mean(metrics[criteria_key])
            early_stop = early_stop.update(criteria)
            has_improved = early_stop.has_improved

            if early_stop.should_stop and FLAGS.early_stop:
                for key, val in metrics.items():
                    if isinstance(val, list):
                        metrics[key] = np.mean(val)
                logger.record_dict(metrics)
                logger.dump_tabular(with_prefix=False, with_timestamp=False)
                wb_logger.log(metrics)
                print('Met early stopping criteria, breaking...')
                break
            elif epoch > 0 and has_improved:
                metrics["best_epoch"] = epoch
                metrics[f"{key}_best"] = criteria
                save_data = {"reward_model": reward_model, "variant": variant, "epoch": epoch}
                save_pickle(save_data, "best_model.pkl", save_dir)

        for key, val in metrics.items():
            if isinstance(val, list):
                metrics[key] = np.mean(val)
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        wb_logger.log(metrics)

    if FLAGS.save_model:
        save_data = {'reward_model': reward_model, 'variant': variant, 'epoch': epoch}
        save_pickle(save_data, 'model.pkl', save_dir)



if __name__ == '__main__':
    absl.app.run(main)


