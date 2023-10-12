
# https://pypi.org/project/tbparse/0.0.2/
# https://github.com/tensorflow/tensorboard/issues/5395

from tbparse import SummaryReader
from matplotlib import pyplot as plt
from pathlib import Path
import sys
import os

sys.path.append(os.getcwd())

if __name__ == '__main__':

    logs_dir = [Path('logs/cloning/20230930-115945/scalars'),
                Path('logs/cql/20230930-154205/scalars'),
                Path('logs/ac_kl/20231001-082017/scalars')]
    
    diff_actions_dataframes = dict()
    reward_dataframes = dict()

    for log_dir in logs_dir:

        df = SummaryReader(log_dir).tensors
        print(df)

        id = str(log_dir).split('\\')[1] # executed in windows machine

        # Batch loss

        if id != 'ac_kl':
            df_batch_loss = df[df['tag'] == 'batch_loss']
            plt.plot(df_batch_loss['step'], df_batch_loss['value'])
            plt.xlabel('Training step')
            if id == 'cloning':
                plt.ylabel('Batch loss (BC agent)')
            else:
                plt.ylabel('Batch loss (CQL agent)')
            plt.savefig(Path('training/plots/' + id + '_batch_loss_report.png'), dpi=1000, bbox_inches='tight')
            plt.close()
        else:
            df_batch_loss_actor = df[df['tag'] == 'batch_loss_actor']
            plt.plot(df_batch_loss_actor['step'], df_batch_loss_actor['value'])
            plt.xlabel('Training step')
            plt.ylabel('Batch loss actor (BRAC agent)')
            plt.savefig(Path('training/plots/' + id + '_batch_loss_actor_report.png'), dpi=1000, bbox_inches='tight')
            plt.close()

            df_batch_loss_critic = df[df['tag'] == 'batch_loss_critic']
            plt.plot(df_batch_loss_critic['step'], df_batch_loss_critic['value'])
            plt.xlabel('Training step')
            plt.ylabel('Batch loss critic (BRAC agent)')
            plt.savefig(Path('training/plots/' + id + '_batch_loss_critic_report.png'), dpi=1000, bbox_inches='tight')
            plt.close()

        # Normalized distances P0, P1, P2
        df_norm_dist_p0 = df[df['tag'] == 'mean_normalized_distance_p0']
        df_norm_dist_p1 = df[df['tag'] == 'mean_normalized_distance_p1']
        df_norm_dist_pother = df[df['tag'] == 'mean_normalized_distance_pother']
        plt.plot(df_norm_dist_p0['step'], df_norm_dist_p0['value'], label='P0', color='red')
        plt.plot(df_norm_dist_p1['step'], df_norm_dist_p1['value'], label='P1', color='orange')
        plt.plot(df_norm_dist_pother['step'], df_norm_dist_pother['value'], label='P2', color='green')
        plt.legend()
        plt.xlabel('Training step')
        if id == 'cloning':
            plt.ylabel('Mean normalized distance (BC agent)')
        elif id == 'cql':
            plt.ylabel('Mean normalized distance (CQL agent)')
        else:
            plt.ylabel('Mean normalized distance (BRAC agent)')
        plt.savefig(Path('training/plots/' + id + '_norm_distances_report.png'), dpi=1000, bbox_inches='tight')
        plt.close()

        # Number of different ambulances
        diff_actions_dataframes[id] = df[df['tag'] == 'num_diff_actions']

        # Reward
        reward_dataframes[id] = df[df['tag'] == 'reward']

    # Number of different ambulances
    plt.plot(diff_actions_dataframes['cloning']['step'], diff_actions_dataframes['cloning']['value'], label='BC agent')
    plt.plot(diff_actions_dataframes['cql']['step'], diff_actions_dataframes['cql']['value'], label='CQL agent')
    plt.plot(diff_actions_dataframes['ac_kl']['step'], diff_actions_dataframes['ac_kl']['value'], label='BRAC agent')
    plt.legend()
    plt.xlabel('Training step')
    plt.ylabel('Mean number of utilized ambulances during an episode')
    plt.savefig(Path('training/plots/num_ambulances_report.png'), dpi=1000, bbox_inches='tight')
    plt.close()

    # Reward
    plt.plot(reward_dataframes['cloning']['step'], reward_dataframes['cloning']['value'], label='BC agent')
    plt.plot(reward_dataframes['cql']['step'], reward_dataframes['cql']['value'], label='CQL agent')
    plt.plot(reward_dataframes['ac_kl']['step'], reward_dataframes['ac_kl']['value'], label='BRAC agent')
    plt.legend()
    plt.xlabel('Training step')
    plt.ylabel('Mean total reward obtained in an episode')
    plt.savefig(Path('training/plots/reward_report.png'), dpi=1000, bbox_inches='tight')
    plt.close()
        