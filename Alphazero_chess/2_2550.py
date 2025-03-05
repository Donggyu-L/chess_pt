import csv

import torch
import argparse
import numpy as np
import random
import os
import wandb

from env import Chess
from mcts_Alphazero import MCTSPlayer
from file_utils import *
from network import PolicyValueNet as MegaNet
from network_copy import PolicyValueNet as AlphaZeroNet
from collections import defaultdict, deque

parser = argparse.ArgumentParser()
parser.add_argument('--init_elo', type=int, default=1500)  # initial Elo rating
parser.add_argument('--k_factor', type=int, default=20)  # sensitivity of the rating adjustment
parser.add_argument('--c_puct', type=int, default=5)
args = parser.parse_args()

init_elo = args.init_elo
k_factor = args.k_factor
c_puct = args.c_puct


# def create_player(file):
#     if not os.path.exists(file):
#         raise RuntimeError("Unexpected model type.")
#
#     playout = 2
#     policy_value_net = PolicyValueNet(8, 8, model_file=file)
#     player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, playout, is_selfplay=0)
#     player.name = file
#
#     return player


def policy_evaluate(env, current_mcts_player, old_mcts_player, n_games=100):
    """start a game between two players"""
    training_mcts_player = current_mcts_player
    opponent_mcts_player = old_mcts_player
    win_cnt = defaultdict(int)

    for j in range(n_games):
        winner, move = start_play(env, training_mcts_player, opponent_mcts_player)
        if winner==1 :
            print("win")
        elif winner==-1 :
            print("lose")
        else:
            print('draw')
        win_cnt[winner] += 1
        print("{} / 100 - move_len: {} ".format(j + 1, len(move)))
        print(move)

    win_ratio = 1.0 * win_cnt[1] / n_games
    print("---------- win: {}, lose: {}, tie:{} ----------".format(win_cnt[1], win_cnt[-1], win_cnt[0]))
    return win_ratio



def start_play(env, player1, player2):
    """start a game between two players"""
    state = env.reset()
    players = [1, 0]
    p1, p2 = players
    player1.set_player_ind(p1)
    player2.set_player_ind(p2)
    players = {p1: player1, p2: player2}
    current_player = 1
    player_in_turn = players[current_player]
    move_list = []

    while True:
        # synchronize the MCTS tree with the current state of the game
        move = player_in_turn.get_action(env, game_iter=-1, temp=1e-3, return_prob=0)  # self-play temp=1.0, eval temp=1e-3
        move_list.append(move)
        env.step(move)

        if not env.terminal:
            current_player = 1 - current_player
            player_in_turn = players[current_player]

        else:
            wandb.log({"eval/game_len": len(move_list),
                       "eval/reward": env.reward
                       })
            return env.reward, move_list



class Player:
    def __init__(self, name, elo=None):
        self.name = name
        self.elo = elo if elo is not None else init_elo  # Set ELO to 1500 if not specified


def update_elo(winner_elo, loser_elo, draw=False, k=k_factor):
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loser = 1 - expected_winner
    score_winner = 0.5 if draw else 1
    score_loser = 0.5 if draw else 0

    new_winner_elo = round(winner_elo + k * (score_winner - expected_winner), 1)
    new_loser_elo = round(loser_elo + k * (score_loser - expected_loser), 1)
    return new_winner_elo, new_loser_elo


# Simulation of a game between two players
def simulate_game(player1, player2):
    # This is a stub function that randomly determines the outcome
    winner = start_play(env, player1, player2)

    if winner == 1:
        player1.elo, player2.elo = update_elo(player1.elo, player2.elo)
    elif winner == -1:
        player2.elo, player1.elo = update_elo(player2.elo, player1.elo)
    elif winner == 0:
        player1.elo, player2.elo = update_elo(player1.elo, player2.elo, draw=True)
    else:
        assert False


if __name__ == '__main__':

    wandb.init(entity="hails",
               project="gym_chess_elo",
               name="elo_test",
               config=args.__dict__
               )
    ### /home/hail/PreferenceTransformer/AlphaPT_chess/Eval/nmcts2/

    env = Chess()
    state = env.reset()


    model_file1 = "/home/hail/PreferenceTransformer/AlphaPT_chess/Eval/nmcts25/train_150.pth"
    model_file2 = "/home/hail/PreferenceTransformer/Alphazero_chess/Eval/nmcts25/train_200.pth"



    if not os.path.exists(model_file1):
        raise RuntimeError("Unexpected model type.")

    if not os.path.exists(model_file2):
        raise RuntimeError("Unexpected model type.")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")

    policy_value_net2 = AlphaZeroNet(8, 8, device,model_file=model_file2)
    mcts_player2 = MCTSPlayer(policy_value_net2.policy_value_fn, c_puct, 25, is_selfplay=0)

    policy_value_net1 = AlphaZeroNet(8, 8, device,model_file=model_file1)
    mcts_player1 = MCTSPlayer(policy_value_net1.policy_value_fn, c_puct, 25, is_selfplay=0)


    # List to hold game results
    game_results = []

    # Simulate games between all pairs of players
    win_ratio = policy_evaluate(env, mcts_player1, mcts_player2)
    print(win_ratio)

    # win_ratio = policy_evaluate(env, mcts_player2, mcts_player1)
    # print(win_ratio)

