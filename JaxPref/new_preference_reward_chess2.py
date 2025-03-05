import os
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
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
import gym
import copy
from gym.spaces import *
import chess
import chess.svg


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

WHITE = 1
BLACK = 0


def __deepcopy__(self, memo):
    new_instance = self.__class__.__new__(self.__class__)
    memo[id(self)] = new_instance

    new_instance.board = np.copy(self.board)
    new_instance.state = np.copy(self.state)
    new_instance.action_space = self.action_space

    return new_instance


def is_repetition(self, count: int = 3) -> bool:
    """
    Checks if the current position has repeated 3 (or a given number of)
    times.

    Unlike :func:`~chess.Board.can_claim_threefold_repetition()`,
    this does not consider a repetition that can be played on the next
    move.

    Note that checking this can be slow: In the worst case, the entire
    game has to be replayed because there is no incremental transposition
    table.
    """
    # Fast check, based on occupancy only.
    maybe_repetitions = 1
    for state in reversed(self._stack):
        if state.occupied == self.occupied:
            maybe_repetitions += 1
            if maybe_repetitions >= count:
                break
    if maybe_repetitions < count:
        return False

    # Check full replay.
    transposition_key = self._transposition_key()
    switchyard = []

    try:
        while True:
            if count <= 1:
                return True

            if len(self.move_stack) < count - 1:
                break

            move = self.pop()
            switchyard.append(move)

            if self.is_irreversible(move):
                break

            if self._transposition_key() == transposition_key:
                count -= 1
    finally:
        while switchyard:
            self.push(switchyard.pop())

    return False


chess.Board.is_repetition = is_repetition


class Chess(gym.Env):
    """AlphaGo Chess Environment"""
    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self):
        self.board = None

        self.T = 8
        self.M = 3
        self.L = 6

        self.size = (8, 8)

        # self.viewer = None

        # self.knight_move2plane[dCol][dRow]
        """
        [ ][5][ ][3][ ]
        [7][ ][ ][ ][1]
        [ ][ ][K][ ][ ]
        [6][ ][ ][ ][0]
        [ ][4][ ][2][ ]
        """
        self.knight_move2plane = {2: {1: 0, -1: 1}, 1: {2: 2, -2: 3}, -1: {2: 4, -2: 5}, -2: {1: 6, -1: 7}}

        self.observation_space = Dict(
            {"P1 piece": Tuple([MultiBinary((8, 8)) for t in range(self.T) for plane in range(6)]),
             "P2 piece": Tuple([MultiBinary((8, 8)) for t in range(self.T) for plane in range(6)]),
             "Repetitions": Tuple([MultiBinary((8, 8)) for t in range(self.T) for plane in range(2)]),
             "Color": MultiBinary((8, 8)),
             "Total move count": MultiBinary((8, 8)),
             "P1 castling": Tuple([MultiBinary((8, 8)) for rook in range(2)]),
             "P2 castling": Tuple([MultiBinary((8, 8)) for rook in range(2)]),
             "No-progress count": MultiBinary((8, 8))})

        self.action_space = Dict(
            {"Queen moves": Tuple([MultiBinary((8, 8)) for squares in range(7) for direction in range(8)]),
             "Knight moves": Tuple([MultiBinary((8, 8)) for move in range(8)]),
             "Underpromotions": Tuple(MultiBinary((8, 8)) for move in range(9))})

    def repetitions(self):
        count = 0
        for state in reversed(self.board.stack):
            if state.occupied == self.board.occupied:
                count += 1

        return count

    def get_direction(self, fromRow, fromCol, toRow, toCol):
        if fromCol == toCol:
            return 0 if toRow < fromRow else 4
        elif fromRow == toRow:
            return 6 if toCol < fromCol else 2
        else:
            if toCol > fromCol:
                return 1 if toRow < fromRow else 3
            else:
                return 7 if toRow < fromRow else 5

    def get_diagonal(self, fromRow, fromCol, toRow, toCol):
        return int(toRow < fromRow and toCol > fromCol or toRow > fromRow and toCol < fromCol)

    def move_type(self, move):
        return "Knight" if self.board.piece_type_at(move.from_square) == 2 else "Queen"

    def observe(self):
        self.P1_piece_planes = np.zeros((8, 8, 6))
        self.P2_piece_planes = np.zeros((8, 8, 6))

        for pos, piece in self.board.piece_map().items():
            row, col = divmod(pos, 8)

            if piece.color == WHITE:
                self.P1_piece_planes[row, col, piece.piece_type - 1] = 1
            else:
                self.P2_piece_planes[row, col, piece.piece_type - 1] = 1

        self.Repetitions_planes = np.concatenate(
            [np.full((8, 8, 1), int(self.board.is_repetition(repeats))) for repeats in range(1, 3)], axis=-1)
        self.Colour_plane = np.full((8, 8, 1), int(self.board.turn))
        self.Total_move_count_plane = np.full((8, 8, 1), self.board.fullmove_number)
        self.P1_castling_planes = np.concatenate((np.full((8, 8, 1), self.board.has_kingside_castling_rights(WHITE)),
                                                  np.full((8, 8, 1), self.board.has_queenside_castling_rights(WHITE))),
                                                 axis=-1)
        self.P2_castling_planes = np.concatenate((np.full((8, 8, 1), self.board.has_kingside_castling_rights(BLACK)),
                                                  np.full((8, 8, 1), self.board.has_queenside_castling_rights(BLACK))),
                                                 axis=-1)

        # The fifty-move rule in chess states that a player can claim a
        # draw if no capture has been made and no pawn has been moved in
        # the last fifty moves (https://en.wikipedia.org/wiki/Fifty-move_rule)
        self.No_progress_count_plane = np.full((8, 8, 1), self.board.halfmove_clock)

        self.binary_feature_planes = np.concatenate(
            (self.P1_piece_planes, self.P2_piece_planes, self.Repetitions_planes), axis=-1)
        self.constant_value_planes = np.concatenate((self.Colour_plane, self.Total_move_count_plane, \
                                                     self.P1_castling_planes, self.P2_castling_planes, \
                                                     self.No_progress_count_plane), axis=-1)

        self.state_history = self.state_history[:, :, 14:-7]
        self.state_history = np.concatenate(
            (self.state_history, self.binary_feature_planes, self.constant_value_planes), axis=-1)
        return self.state_history

    def reset(self):
        if self.board is None:
            self.board = chess.Board()

        self.board.reset()

        self.turn = WHITE

        self.reward = None
        self.terminal = False

        # Initialize states before timestep 1 to matrices containing all zeros
        self.state_history = np.zeros((8, 8, 14 * self.T + 7))
        return self.observe()

    def legal_move_mask(self):
        mask = np.zeros((8, 8, 73))

        for move in self.board.legal_moves:
            fromRow = 7 - move.from_square // 8
            fromCol = move.from_square % 8

            toRow = 7 - move.to_square // 8
            toCol = move.to_square % 8

            dRow = toRow - fromRow
            dCol = toCol - fromCol

            piece_type = self.board.piece_type_at(move.from_square)

            if piece_type == 2:  # Knight move
                # plane = knight_move2plane[dCol][dRow] + 56 # SH edit
                plane = self.knight_move2plane[dCol][dRow] + 56
            else:  # Queen move
                if move.promotion and move.promotion in [2, 3, 4]:  # Underpromotion move (to knight, biship, or rook)
                    if fromCol == toCol:  # Regular pawn promotion move
                        plane = 64 + move.promotion - 2
                    else:  # Simultaneous diagonal pawn capture from the 7th rank and subsequent promotion
                        diagonal = self.get_diagonal(fromRow, fromCol, toRow, toCol)
                        plane = 64 + (diagonal + 1) * 3 + move.promotion - 2
                else:  # Regular queen move
                    squares = max(abs(toRow - fromRow), abs(toCol - fromCol))
                    direction = self.get_direction(fromRow, fromCol, toRow, toCol)
                    plane = (squares - 1) * 8 + direction

            mask[fromRow, fromCol, plane] = 1

        return mask

    def step(self, p):
        mask = self.legal_move_mask()
        p = p * mask
        pMin, pMax = p.min(), p.max()
        p = (p - pMin) / (pMax - pMin)
        action = np.unravel_index(p.argmax(), p.shape)

        fromRow, fromCol, plane = action

        if plane < 56:  # Queen move
            squares, direction = divmod(plane, 8)
            squares += 1

            """
            7 0 1
            6   2
            5 4 3
            """
            if direction == 0:
                toRow = fromRow - squares
                toCol = fromCol
            elif direction == 1:
                toRow = fromRow - squares
                toCol = fromCol + squares
            elif direction == 2:
                toRow = fromRow
                toCol = fromCol + squares
            elif direction == 3:
                toRow = fromRow + squares
                toCol = fromCol + squares
            elif direction == 4:
                toRow = fromRow + squares
                toCol = fromCol
            elif direction == 5:
                toRow = fromRow + squares
                toCol = fromCol - squares
            elif direction == 6:
                toRow = fromRow
                toCol = fromCol - squares
            else:  # direction == 7
                toRow = fromRow - squares
                toCol = fromCol - squares

            fromSquare = (7 - fromRow) * 8 + fromCol
            toSquare = (7 - toRow) * 8 + toCol
            move = chess.Move(fromSquare, toSquare)
        elif plane < 64:  # Knight move
            """
            [ ][5][ ][3][ ]
            [7][ ][ ][ ][1]
            [ ][ ][K][ ][ ]
            [6][ ][ ][ ][0]
            [ ][4][ ][2][ ]
            """
            if plane == 56:
                toRow = fromRow + 1
                toCol = fromCol + 2
            elif plane == 57:
                toRow = fromRow - 1
                toCol = fromCol + 2
            elif plane == 58:
                toRow = fromRow + 2
                toCol = fromCol + 1
            elif plane == 59:
                toRow = fromRow - 2
                toCol = fromCol + 1
            elif plane == 60:
                toRow = fromRow + 2
                toCol = fromCol - 1
            elif plane == 61:
                toRow = fromRow - 2
                toCol = fromCol - 1
            elif plane == 62:
                toRow = fromRow + 1
                toCol = fromCol - 2
            else:  # plane == 63
                toRow = fromRow - 1
                toCol = fromCol - 2

            fromSquare = (7 - fromRow) * 8 + fromCol
            toSquare = (7 - toRow) * 8 + toCol
            move = chess.Move(fromSquare, toSquare)
        else:  # Underpromotions
            toRow = fromRow - self.board.turn

            if plane <= 66:
                toCol = fromCol
                promotion = plane - 62
            elif plane <= 69:
                diagonal = 0
                promotion = plane - 65
                toCol = fromCol - self.board.turn
            else:  # plane <= 72
                diagonal = 1
                promotion = plane - 68
                toCol = fromCol + self.board.turn

            fromSquare = (7 - fromRow) * 8 + fromCol
            toSquare = (7 - toRow) * 8 + toCol
            move = chess.Move(fromSquare, toSquare, promotion=promotion)

        self.board.push(move)

        # self.board = self.board.mirror()

        result = self.board.result(claim_draw=True)
        self.reward = 0 if result == '*' or result == '1/2-1/2' else 1 if result == '1-0' else -1  # if result == '0-1'
        self.terminal = self.board.is_game_over(claim_draw=True)
        self.info = {'last_move': move, 'turn': self.board.turn}

        return self.observe(), self.reward, self.terminal, self.info

    # def get_image(self):
    #   out = BytesIO()
    #   bytestring = chess.svg.board(self.board, size=1000).encode('utf-8')
    #   cairosvg.svg2png(bytestring=bytestring, write_to=out)
    #   image = Image.open(out).convert("RGB")
    #   return np.asarray(image).astype(np.uint8)
    #
    # def render(self, mode='human'):
    #   img = self.get_image()
    #
    #   if mode == 'rgb_array':
    #     return img
    #   elif mode == 'human':
    #     if self.viewer is None:
    #       from gym.envs.classic_control import rendering
    #       self.viewer = rendering.SimpleImageViewer()
    #
    #     self.viewer.imshow(img)
    #     return self.viewer.isopen
    #   else:
    #     raise NotImplementedError

    # def close(self):
    #   if not self.viewer is None:
    #     self.viewer.close()

def uci_move_to_index(move):
    from_square = move.from_square
    to_square = move.to_square
    move_index = 64 * from_square + to_square
    return move_index

def collect(df, env, len_query):
    df = df.groupby('game_id', sort=False)

    observations, new_observations, actions, labels, timesteps = [], [], [], [], []

    for _, group in df:
        group = group.reset_index(drop=True)
        labels.append(group['label'].values[0])

        env.reset()

        moves = group['move'].tolist()
        observations_grp, actions_grp, timesteps_grp = [], [], []

        for timestep, move_uci in enumerate(moves):
            move = chess.Move.from_uci(move_uci)
            env.board.push(move)
            obse = env.observe()

            observations_grp.append(obse)
            actions_grp.append(uci_move_to_index(move))
            timesteps_grp.append(timestep)

        if len(observations_grp) < 2:
            continue

        observation_ = np.array(observations_grp[:-1])
        observation_new_ = np.array(observations_grp[1:])
        actions_ = np.array(actions_grp)
        timesteps_ = np.array(timesteps_grp)


        pad_width = len_query - len(observation_)
        if pad_width > 0:
            observation_ = np.pad(observation_, ((0, pad_width), (0, 0), (0, 0), (0, 0)), mode='constant')
            observation_new_ = np.pad(observation_new_, ((0, pad_width), (0, 0), (0, 0), (0, 0)), mode='constant')
            actions_ = np.pad(actions_, (0, pad_width), mode='constant')
            timesteps_ = np.pad(timesteps_, (0, pad_width), mode='constant')


        max_idx = max(0, len(observation_) - len_query)
        idxx = np.random.randint(0, max_idx + 1) if max_idx > 0 else 0

        observations.append(observation_[idxx:idxx + len_query])
        new_observations.append(observation_new_[idxx:idxx + len_query])
        actions.append(actions_[idxx:idxx + len_query])
        timesteps.append(timesteps_[idxx:idxx + len_query])

    return {
        "observations": np.array(observations, dtype=float),
        "next_observations": np.array(new_observations, dtype=float),
        "actions": np.array(actions),
        "labels": np.array(labels),
        "timestep_1": np.array(timesteps)
    }

def combine(prep_batch1, prep_batch2):
    batch = {}

    labels_1 = np.array(prep_batch1['labels'])
    labels_2 = np.array(prep_batch2['labels'])

    combined_labels = np.where(labels_1 > labels_2, [1, 0],
                               np.where(labels_1 < labels_2, [0, 1], [0.5, 0.5]))

    batch['observations'] = prep_batch1['observations']
    batch['next_observations'] = prep_batch1['next_observations']
    batch['actions'] = prep_batch1['actions']
    batch['observations_2'] = prep_batch2['observations_2']
    batch['next_observations_2'] = prep_batch2['next_observations_2']
    batch['actions_2'] = prep_batch2['actions_2']

    batch['timestep_1'] = prep_batch1['timestep_1']
    batch['timestep_2'] = prep_batch2['timestep_2']

    batch['labels'] = combined_labels
    batch['script_labels'] = combined_labels

    return batch

def generate_sigma(num_samples=251671):
    sigma = np.random.permutation(num_samples)
    return sigma[:30000], sigma[30000:]

def data_sample_by_id(df, sampled_indices):
    index_range = np.concatenate([np.arange(start, end + 1) for start, end in sampled_indices])
    return df.loc[index_range]

def create_dataset(idx1, idx2, classic_df, start_pt, end_pt, env, len_query):
    sampled_idx1 = idx1[start_pt:end_pt]
    sampled_idx2 = idx2[start_pt:end_pt]

    df1 = data_sample_by_id(classic_df, sampled_idx1)
    df2 = data_sample_by_id(classic_df, sampled_idx2)

    batch1 = collect(df1, env, len_query)
    batch2 = collect(df2, env, len_query)

    return combine(batch1, batch2)

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

    observation_dim = 7616
    action_dim = 4672
    criteria_key = None

    data_size = 10000
    interval = int(data_size / FLAGS.batch_size) + 1
    early_stop = EarlyStopping(min_delta=FLAGS.min_delta, patience=FLAGS.patience)


    total_epochs = FLAGS.n_epochs
    config = transformers.GPT2Config(
        **FLAGS.transformer
    )
    config.warmup_steps = int(total_epochs * 0.1 * interval)
    config.total_steps = total_epochs * interval

    trans = TransRewardModel(config=config, observation_dim=observation_dim, action_dim=action_dim,
                             activation=FLAGS.activations, activation_final=FLAGS.activation_final)

    reward_model = PrefTransformer(config, trans)
    train_loss = "reward/trans_loss"

    file_path = '/media/hail/HDD/Chess_data/results/train_data.csv'
    classic_df = pd.read_csv(file_path, usecols=['game_id', 'type', 'move'])

    with open('/media/hail/HDD/Chess_data/results/game_indices.pkl', 'rb') as f:
        game_indices= pickle.load(f)

    eval_data_size = 2000
    eval_interval = int(eval_data_size / FLAGS.batch_size) + 1

    env = Chess()

    for epoch in range(FLAGS.n_epochs + 1):
        sigma1, sigma2 = generate_sigma()
        metrics = defaultdict(list)
        metrics['epoch'] = epoch
        if epoch:
            idx1 = [game_indices[i] for i in sigma1]
            idx2 = [game_indices[i] for i in sigma2]

            for i in range(interval):
                start_pt = i * FLAGS.batch_size
                end_pt = min((i + 1) * FLAGS.batch_size,10000)
                with Timer() as train_timer:
                    # train
                    pref_dataset = create_dataset(idx1, idx2, classic_df,start_pt,end_pt,env,50)
                    batch = batch_to_jax(pref_dataset)
                    for key, val in prefix_metrics(reward_model.train(batch), 'reward').items():
                        metrics[key].append(val)
            metrics['train_time'] = train_timer()
        else:
            # for using early stopping with train loss.
            metrics[train_loss] = [float(FLAGS.query_len)]

            # eval phase
        if epoch % FLAGS.eval_period == 0:
            for j in range(eval_interval):
                eval_start_pt, eval_end_pt = j * FLAGS.batch_size, min((j + 1) * FLAGS.batch_size, 2000)
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


