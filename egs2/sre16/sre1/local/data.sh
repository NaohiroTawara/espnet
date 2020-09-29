#!/bin/bash
set -euo pipefail
SECONDS=0
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=0
stop_stage=10000000
data_dir=data

train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.

log "$0 $*"
. ./utils/parse_options.sh || exit 1;

if [ $# -ne 0 ]; then
    log "Usage $0"
    exit 1
fi

. ./db.sh    # Specify the corpora directory
. ./path.sh  # Setup the environment


if [ -z "${SWBD_CELLULAR1}" ]; then
    log "Error: \$SWBD_CELLULAR1 is not set. See db.sh"
    exit 1
fi
if [ -z "${SWBD_CELLULAR2}" ]; then
    log "Error: \$SWBD_CELLULAR2 is not set. See db.sh"
    exit 1
fi
if [ -z "${SWBD_PHASE1}" ]; then
    log "Error: \$SWBD_PHASE1 is not set. See db.sh"
    exit 1
fi
if [ -z "${SWBD_PHASE2}" ]; then
    log "Error: \$SWBD_PHASE2 is not set. See db.sh"
    exit 1
fi
if [ -z "${SWBD_PHASE3}" ]; then
    log "Error: \$SWBD_PHASE3 is not set. See db.sh"
    exit 1
fi
if [ -z "${SRE2008_TRAIN}" ]; then
    log "Error: \$SRE2008_TRAIN is not set. See db.sh"
    exit 1
fi
if [ -z "${SRE2008_TEST}" ]; then
    log "Error: \$SRE2-08_TEST is not set. See db.sh"
    exit 1
fi
if [ -z "${SRE2004}" ]; then
    log "Error: \$SRE2004 is not set. See db.sh"
    exit 1
fi
if [ -z "${SRE2005_TRAIN}" ]; then
    log "Error: \$SRE2005_TRAIN is not set. See db.sh"
    exit 1
fi
if [ -z "${SRE2005_TEST}" ]; then
    log "Error: \$SRE2005_TEST is not set. See db.sh"
    exit 1
fi
if [ -z "${SRE2006_TRAIN}" ]; then
    log "Error: \$SRE2006_TRAIN is not set. See db.sh"
    exit 1
fi
if [ -z "${SRE2006_TEST1}" ]; then
    log "Error: \$SRE2006_TEST1 is not set. See db.sh"
    exit 1
fi
if [ -z "${SRE2006_TEST2}" ]; then
    log "Error: \$SRE2006_TEST2 is not set. See db.sh"
    exit 1
fi

if [ -z "${FISHER1}" ]; then
    log "Error: \$FISHER1 is not set. See db.sh"
    exit 1
fi
if [ -z "${FISHER1_TRANS}" ]; then
    log "Error: \$FISHER1_TRANS is not set. See db.sh"
    exit 1
fi
if [ -z "${FISHER2}" ]; then
    log "Error: \$FISHER2 is not set. See db.sh"
    exit 1
fi
if [ -z "${FISHER2_TRANS}" ]; then
    log "Error: \$FISHER2_TRANS is not set. See db.sh"
    exit 1
fi

kaldi_sre=$KALDI_ROOT/egs/sre16/v2/
if [ $stage -le 1 ]; then
    log "stage 1: Generate scp files"

    local/make_fisher.sh $FISHER1/ $FISHER1_TRANS/ $data_dir/fisher
    local/make_fisher.sh $FISHER2/ $FISHER2_TRANS/ $data_dir/fisher
    utils/combine_data.sh $data_dir/fisher \
        $data_dir/fisher1 $data_dir/fisher1
     utils/validate_data_dir.sh --no-text --no-feats $data_dir/fisher

    if [ ! -e $data_dir/local/speaker_list ] ; then
        wget -P $data_dir/local/ http://www.openslr.org/resources/15/speaker_list.tgz
        tar -C $data_dir/local/ -xvf data/local/speaker_list.tgz
        rm data/local/speaker_list.*
    fi
    sre_ref=$data_dir/local/speaker_list
    $kaldi_sre/local/make_sre.pl $SRE2004 04 $sre_ref $data_dir/sre2004
    $kaldi_sre/local/make_sre.pl $SRE2005_TRAIN 05 $sre_ref $data_dir/sre2005_train
    $kaldi_sre/local/make_sre.pl $SRE2005_TEST 05 $sre_ref $data_dir/sre2005_test
    $kaldi_sre/local/make_sre.pl $SRE2006_TRAIN 06 $sre_ref $data_dir/sre2006_train
    $kaldi_sre/local/make_sre.pl $SRE2006_TEST1 06 $sre_ref $data_dir/sre2006_test_1
    $kaldi_sre/local/make_sre.pl $SRE2006_TEST2 06 $sre_ref $data_dir/sre2006_test_2

    #$kaldi_sre/local/make_sre10.pl $SRE2010/eval/ $data_dir
    $kaldi_sre/local/make_sre08.pl $SRE2008_TEST $SRE2008_TRAIN $data_dir

    utils/combine_data.sh $data_dir/sre \
        $data_dir/sre2004 $data_dir/sre2005_train \
        $data_dir/sre2005_test $data_dir/sre2006_train \
        $data_dir/sre2006_test_1 $data_dir/sre2006_test_2 # $data_dir/sre10
     utils/validate_data_dir.sh --no-text --no-feats $data_dir/sre
     utils/fix_data_dir.sh $data_dir/sre

    # Prepare SWBD corpora.
    $kaldi_sre/local/make_swbd_cellular1.pl $SWBD_CELLULAR1/ \
        $data_dir/swbd_cellular1_train
    $kaldi_sre/local/make_swbd_cellular2.pl $SWBD_CELLULAR2/ \
        $data_dir/swbd_cellular2_train
    $kaldi_sre/local/make_swbd2_phase1.pl $SWBD_PHASE1/ \
        $data_dir/swbd2_phase1_train
    $kaldi_sre/local/make_swbd2_phase2.pl $SWBD_PHASE2/ \
        $data_dir/swbd2_phase2_train
    $kaldi_sre/local/make_swbd2_phase3.pl $SWBD_PHASE3/ \
        $data_dir/swbd2_phase3_train

    utils/combine_data.sh $data_dir/swbd \
        $data_dir/swbd_cellular1_train $data_dir/swbd_cellular2_train \
        $data_dir/swbd2_phase1_train $data_dir/swbd2_phase2_train $data_dir/swbd2_phase3_train

    utils/combine_data.sh $data_dir/sre_swbd_fisher \
        $data_dir/swbd $data_dir/sre $data_dir/fisher
    utils/validate_data_dir.sh --no-text --no-feats $data_dir/sre_swbd_fisher
    utils/fix_data_dir.sh $data_dir/sre_swbd_fisher
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Generate train / validation sets "
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 20 $data_dir/sre_swbd_fisher $data_dir/$train_set $data_dir/tmp
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 50 $data_dir/tmp $data_dir/$valid_set $data_dir/$test_sets
    rm -r $data_dir/tmp
    for setname in $valid_set $test_sets; do
        python local/make_trials.py $data_dir/$setname/ --num_diff 3 --num_same 10
    done

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
