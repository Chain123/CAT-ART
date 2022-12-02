import _pickle as pickle
import argparse
import os


def print_result(data_dir):
    models_runs = [os.path.join(data_dir, val) for val in os.listdir(data_dir)]
    for file_dir in models_runs:
        if os.path.isfile(os.path.join(file_dir, 'result.pickle')):
            data = pickle.load(open(os.path.join(file_dir, 'result.pickle'), 'rb'))
            print(f" ====: {os.path.join(file_dir, 'result.pickle')}")
            for key in data.keys():
                print(f"{key}")
                print('\t'.join([str(val) for val in data[key]]))


parser = argparse.ArgumentParser('Unsupervised Recommendation Training')
# Path Arguments  gender_method
parser.add_argument('--data_dir', type=str,
                    default='/data/ceph/seqrec/UMMD/www/recommend',
                    help='print out the result in the result dir')

params = parser.parse_args()
print_result(params.data_dir)
