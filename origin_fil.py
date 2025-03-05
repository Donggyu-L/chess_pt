import csv
import os
import pandas as pd
import matplotlib.pyplot as plt



def process_large_csv(input_file, output_file):
    selected_columns = [
        'game_id', 'type', 'result',
        'white_elo', 'black_elo', 'time_control', 'num_ply', 'termination',
        'white_won', 'black_won', 'no_winner', 'move_ply', 'move', 'cp', 'cp_rel',
        'cp_loss', 'is_blunder_cp',
        'is_blunder_wr',
        'active_won', 'is_capture', 'clock', 'opp_clock', 'clock_percent',
        'opp_clock_percent', 'low_time', 'board',
        'is_check', 'num_legal_moves'

    ]

    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames

        if not set(selected_columns).issubset(set(headers)):
            missing_cols = set(selected_columns) - set(headers)
            print(f"Missing columns: {missing_cols}")
            return

    with open(output_file, 'w', newline='', encoding='utf-8') as out_csv:
        writer = None

        with open(input_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                try:
                    white_elo = int(row['white_elo'])
                    black_elo = int(row['black_elo'])
                    time_control = row['time_control']
                    num_ply = int(row['num_ply'])
                    result = row['result']
                    type = row['type']


                    total_time = 0
                    if '+' in time_control:
                        base_time, increment_time = time_control.split('+')
                        total_time = int(base_time) + 40 * int(increment_time)
                    elif time_control == '-':
                        total_time = -1

                    #and max(white_elo, black_elo) <= 1740
                    if (type != 'Bullet') :
                        continue
                    # if (total_time < 300  ):
                    #     continue
                    if ( min(white_elo, black_elo) < 1750) :
                        continue
                    # if not (10 <= num_ply <= 200):
                    #     continue
                    if result == '*' or result not in ['1-0', '0-1', '1/2-1/2']:
                        continue


                    if writer is None:
                        writer = csv.DictWriter(out_csv, fieldnames=selected_columns)
                        writer.writeheader()

                    filtered_row = {col: row[col] for col in selected_columns}
                    writer.writerow(filtered_row)

                except Exception as e:
                    print(f"Error processing row: {row}")
                    print(e)
                    continue




if __name__ == '__main__':
    input_filename = '/home/hail/lichess_db_standard_rated_2019-11.csv'
    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)
    output_filename = 'eval_Bullet_expert_data.csv'
    output_path = os.path.join(output_folder, output_filename)

    process_large_csv(input_filename, output_path)
    print(f'{input_filename} processed and saved to {output_path}')
