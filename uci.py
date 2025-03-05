import chess

def index_to_uci_move(move_index):
    from_square = move_index // 64
    to_square = move_index % 64
    move = chess.Move(from_square, to_square)
    return move.uci()


if __name__ == '__main__' :
    uci = []

    move_index = [723, 3437, 1243, 3494, 267, 2462, 723, 1942, 983, 3690, 1503, 3307, 1236, 2787, 1301, 3641, 2023, 3704, 2543, 3641, 1373, 3112, 1878, 3891, 1438, 3324, 1959, 3763, 2527, 2917, 2023, 3704, 2527, 3641, 2023, 3322, 2527]





    for i in range(len(move_index)) :
        m = index_to_uci_move(move_index[i])
        uci.append(m)

    print(uci)