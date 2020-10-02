#!/usr/bin/env bash
set -euo pipefail
SECONDS=0
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
help_message=$(cat << EOF
Usage: $0 <in-wav.scp> <out-datadir> [<logdir> [<outdir>]]
e.g.
$0 data/test/wav.scp data/test_format/

Apply utterance-level VAD to 'wav.scp':

Options
  --nj <nj>
  --cmd <cmd>
EOF
)

out_filename=segments
cmd=utils/run.pl
nj=30
min_sec=


log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 2 ] && [ $# -ne 3 ] && [ $# -ne 4 ]; then
    log "${help_message}"
    log "Error: invalid command line arguments"
    exit 1
fi

. ./path.sh  # Setup the environment

scp=$1
if [ ! -f "${scp}" ]; then
    log "${help_message}"
    echo "$0: Error: No such file: ${scp}"
    exit 1
fi
dir=$2


if [ $# -eq 2 ]; then
    logdir=${dir}/logs
    outdir=${dir}/data

elif [ $# -eq 3 ]; then
    logdir=$3
    outdir=${dir}/data

elif [ $# -eq 4 ]; then
    logdir=$3
    outdir=$4
fi


mkdir -p ${logdir}

rm -f "${dir}/${out_filename}"


opts=
if [ -n "${min_sec}" ]; then
    opts="--min_sec ${min_sec} "
fi

nutt=$(<${scp} wc -l)
nj=$((nj<nutt?nj:nutt))

split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/wav.${n}.scp"
done

utils/split_scp.pl "${scp}" ${split_scps}
${cmd} "JOB=1:${nj}" "${logdir}/vad_wav_scp.JOB.log" \
    pyscripts/audio/vad_wav_scp.py \
    ${opts} \
    "${logdir}/wav.JOB.scp" \
    ${logdir}/${out_filename}.JOB

# Workaround for the NFS problem
ls ${logdir}/${out_filename}.* > /dev/null

# concatenate the .scp files together.
for n in $(seq ${nj}); do
    cat "${logdir}/${out_filename}.$n" || exit 1;
done |sort > "${dir}/${out_filename}" || exit 1


log "Successfully finished. [elapsed=${SECONDS}s]"
