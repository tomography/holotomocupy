#!/bin/bash
# Walk the psf_sigma arms one at a time from handyn, over ssh.
#
#     ./sweep_driver_psf.sh                  # all six arms
#     ARMS="12 18" ./sweep_driver_psf.sh     # just these
#
# One job at a time: the debug queue allows a single queued job per user, so
# this blocks on each qsub before submitting the next.  Progress goes to
# ${LOG} -- /local/ssd, never /tmp.
set -u

ARMS=${ARMS:-"00 06 12 18 24 30"}
# The CODE lives in $HOME on Polaris; only path_out points at /eagle.  Keep
# this RELATIVE: ssh starts in the remote $HOME, and a ~ would expand here.
WD=holotomocupy_gpu_reduced/experimental/WT_cell1
LOG=${LOG:-/local/ssd/vnikitin/wt_cell1_sweep_psf.log}

say() { echo "[$(date +%H:%M:%S)] $*" >> "${LOG}"; }

say "psf sweep start: arms = ${ARMS}"
for p in ${ARMS}; do
    jid=$(ssh -o BatchMode=yes polaris "cd ${WD} && qsub -N WTpsf${p} -v CFG=config_psf${p}.conf polaris_run_psf.sh")
    if [ -z "${jid}" ]; then say "arm psf${p}  SUBMIT FAILED"; continue; fi
    say "arm psf${p}  submitted  ${jid}"
    num=${jid%%.*}

    # Poll until PBS no longer lists it.  qstat exits non-zero on an unknown
    # job id, which is the completion signal here.
    while ssh -o BatchMode=yes polaris "qstat ${num}" >/dev/null 2>&1; do sleep 30; done

    # Last error line, whatever niter is -- 'iter=1024:' only existed at
    # niter=1025.  Its absence is the real failure signal; PBS does not write
    # an Exit_status line into the .o file here.
    ex=$(ssh -o BatchMode=yes polaris "cd ${WD} && grep -c 'ALL STAGES DONE' WTpsf${p}.o${num} 2>/dev/null")
    done_line=$(ssh -o BatchMode=yes polaris "cd ${WD} && grep -h -oE 'iter=[0-9]+: [0-9.]+sec err=[0-9.e+-]+' WTpsf${p}.o${num} 2>/dev/null | tail -1")
    drift=$(ssh -o BatchMode=yes polaris "cd ${WD} && grep -h -oE 'p-p=[0-9.]+%' WTpsf${p}.o${num} 2>/dev/null | tail -1")
    say "arm psf${p}  finished  ok=${ex:-0}  ${done_line:-<NO error line -- job failed>}  drift removed ${drift:-?}"
done
say "psf sweep done"
