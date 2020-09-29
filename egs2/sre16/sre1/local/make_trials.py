import numpy as np
import argparse
import os

def read_scp(filename):
    utt = {}
    with open(filename) as f:
        for line in f.readlines():
            line = line.replace('\n', '').split(' ')
            utt[line[0]] = line[1:]
    return utt


if __name__ == '__main__':
    np.random.seed(10)
    parser = argparse.ArgumentParser('make_trials')
    parser.add_argument('target_dir', type=str, help='target dir')
    parser.add_argument('--num_trials', type=int, default=None, help='start values of each bin')
    parser.add_argument('--num_diff', type=int, default=1, help='start values of each bin')
    parser.add_argument('--num_same', type=int, default=None, help='start values of each bin')
    args = parser.parse_args()
    spk2utt_file = args.target_dir + '/spk2utt'
    wav_file = args.target_dir +  '/wav.scp'
    assert os.path.exists(spk2utt_file), f'{spk2utt_file} is not found'
    assert os.path.exists(wav_file), f'{wav_file} is not found'

    spk2utt = read_scp(spk2utt_file)
    wavs = read_scp(wav_file)
    with open(args.target_dir + '/label', 'w') as f_label:
        with open(args.target_dir + '/ref.scp', 'w') as f_ref:
            with open(args.target_dir + '/anc.scp', 'w') as f_anc:

                count = 0
                spk_list = [key for key in spk2utt]
                for spk in spk2utt:
                    utts = spk2utt[spk]
                    for i,utt in enumerate(utts):
                        if args.num_same is None or args.num_same > len(utts):
                           num_same = len(utts)
                        else:
                          num_same = args.num_same
                        for i in range(i + 1, num_same):
                            f_label.write(f'pair_{count} 1\n')
                            f_anc.write(f'pair_{count} {" ".join(wavs[utt])}\n')
                            f_ref.write(f'pair_{count} {" ".join(wavs[utts[i]])}\n')
                            #print("target {} {}".format(utt, utts[i]))
                            count += 1
                        diff_spks = np.random.choice(sorted(list(set(spk_list) - set(spk))), args.num_diff, replace=False)
                        for diff_spk in diff_spks:
                            f_label.write(f'pair_{count} 0\n')
                            f_anc.write(f'pair_{count} {" ".join(wavs[spk2utt[spk][np.random.randint(len(spk2utt[spk]))]])}\n')
                            f_ref.write(f'pair_{count} {" ".join(wavs[spk2utt[diff_spk][np.random.randint(len(spk2utt[diff_spk]))]])}\n')
                            count += 1




