import pandas as pd
import matplotlib.pyplot as plt


def elo_hist(csv_file):

    try:

        df = pd.read_csv(csv_file, usecols=['game_id', 'type', 'white_elo', 'black_elo'])
        df = df[df['type'] == 'Classical']

        grouped = df.groupby('game_id').first()

        white_elo = grouped['white_elo'].dropna().astype(int)
        black_elo = grouped['black_elo'].dropna().astype(int)

        def cal_stat(elo, name):
            stats = {
                'Mean': elo.mean(),
                'Median': elo.median(),
                'Standard Deviation': elo.std(),
                '25th Percentile': elo.quantile(0.25),
                '50th Percentile (Median)': elo.median(),
                '75th Percentile': elo.quantile(0.75),
                'Minimum': elo.min(),
                'Maximum': elo.max()
            }
            print(f"\n{name} Elo Statistics:")
            for stat, value in stats.items():
                print(f"{stat}: {value}")

        cal_stat(white_elo, "White")
        cal_stat(black_elo, "Black")

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(white_elo, bins=50, color='blue', alpha=0.7)
        plt.xlabel('White Elo Rating')
        plt.ylabel('Frequency')
        plt.title('White Elo Rating Histogram')

        plt.subplot(1, 2, 2)
        plt.hist(black_elo, bins=50, color='red', alpha=0.7)
        plt.xlabel('Black Elo Rating')
        plt.ylabel('Frequency')
        plt.title('Black Elo Rating Histogram')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error while plotting Elo histograms: {e}")


# Example usage
csv_file = '/home/hail/expert_data_07.csv'  # Replace with your actual file path
elo_hist(csv_file)
