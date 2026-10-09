#!/bin/bash
# Pull the light step-6 output of AtomiumS1 from Polaris down to /data3.
#
#     ./sync_tiff.sh                 # one shot
#     ./sync_tiff.sh --watch         # re-sync every 5 min until Ctrl-C
#     ./sync_tiff.sh --watch 60      # ... every 60 s
#     ./sync_tiff.sh --dry-run       # list what would be copied
#     ./sync_tiff.sh --h5            # ALSO pull checkpoints/*.h5 (13 GB each at
#                                    # bin 2, ~106 GB at bin 1, ~850 GB at bin 0)
#
# What it copies by default (a couple of GB for the whole ladder):
#     checkpoints_tiff/   the two preview slices per checkpoint
#     pos_errors/  shrink/  conv*.csv    the per-iteration diagnostics
# What it never copies unless you ask: checkpoints/*.h5, the full obj volume.
#
# WHERE TO RUN IT: /data3 is an NFS mount from tomodata3-ib and is NOT mounted
# on handyn -- only on tomo5.  Started anywhere without /data3, this script
# re-runs itself on tomo5 over ssh, so ./sync_tiff.sh works from either box.
#
# POLARIS LOGIN: every machine needs its own MobilePASS+ passcode once; the
# ssh master is then held for 8 h.  The script opens that login for you if no
# master is up.  It uses a per-machine ControlPath (~/.ssh/cm-<host>-...) so
# tomo5's master does not stomp on the one handyn's sshfs mounts depend on.
set -u

REMOTE=${REMOTE:-polaris}
SRC=${SRC:-/eagle/APS_IRI/vnikitin/20260829/ctxl/ctxl_HT_4K_RD300_007p5nm_0001_rec6_p1}
DST=${DST:-/data3/vnikitin/ESRF/ctxl_HT_p1}
DATA3_HOST=${DATA3_HOST:-tomo5}

WATCH=0; INTERVAL=300; DRY=(); WITH_H5=0
while [ $# -gt 0 ]; do
    case "$1" in
        --watch)   WATCH=1; case "${2:-}" in ''|-*) ;; *) INTERVAL=$2; shift;; esac ;;
        --dry-run) DRY=(--dry-run) ;;
        --h5)      WITH_H5=1 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# --- hop to the machine that mounts /data3 -----------------------------------
DST_ROOT="/$(echo "${DST#/}" | cut -d/ -f1)"          # /data3 for the default DST
if [ ! -d "$DST_ROOT" ]; then
    [ "${SYNC_TIFF_HOPPED:-0}" = 1 ] && {
        echo "$DST_ROOT is not mounted on $(hostname -s) either -- giving up" >&2; exit 1; }
    echo "$DST_ROOT is not mounted on $(hostname -s); re-running on ${DATA3_HOST}"
    ARGS=""
    [ $WATCH   = 1 ] && ARGS="$ARGS --watch $INTERVAL"
    [ ${#DRY[@]} -gt 0 ] && ARGS="$ARGS --dry-run"
    [ $WITH_H5 = 1 ] && ARGS="$ARGS --h5"
    exec ssh -t "$DATA3_HOST" "cd '$(cd "$(dirname "$0")" && pwd)' && \
        SYNC_TIFF_HOPPED=1 REMOTE='$REMOTE' SRC='$SRC' DST='$DST' \
        ./$(basename "$0")$ARGS"
fi

# --- one ssh master per machine, so handyn's sshfs is not disturbed ----------
CP=${SSH_CONTROL_PATH:-"$HOME/.ssh/cm-$(hostname -s)-%r@%h:%p"}
SSH="ssh -o ControlMaster=auto -o ControlPath=$CP -o ControlPersist=8h"
if ! $SSH -O check "$REMOTE" >/dev/null 2>&1; then
    if [ -t 0 ]; then
        echo "opening an ssh master to ${REMOTE} (MobilePASS+ passcode needed once)"
        $SSH -fN "$REMOTE" || { echo "ssh to ${REMOTE} failed" >&2; exit 1; }
    else
        echo "no ssh master to ${REMOTE} on $(hostname -s), and no terminal to type" >&2
        echo "the MobilePASS+ passcode into.  Run this once, from a terminal:" >&2
        echo "    ssh -o ControlMaster=auto -o ControlPath=$CP -o ControlPersist=8h -fN $REMOTE" >&2
        exit 1
    fi
fi

FILTER=(--include='checkpoints_tiff/***'
        --include='pos_errors/***'
        --include='shrink/***'
        --include='conv*.csv'
        --include='*.log')
[ $WITH_H5 = 1 ] && FILTER+=(--include='checkpoints/***')
FILTER+=(--exclude='*')

mkdir -p "$DST" || exit 1

sync_once() {
    rsync -rlptvh --partial --info=stats1,progress2 "${DRY[@]}" \
        -e "$SSH" "${FILTER[@]}" "${REMOTE}:${SRC}/" "${DST}/"
}

if [ $WATCH = 1 ]; then
    echo "watching ${REMOTE}:${SRC} -> ${DST} every ${INTERVAL}s (Ctrl-C to stop)"
    while true; do
        echo "=== $(date '+%F %T') ==="
        sync_once
        sleep "$INTERVAL"
    done
else
    sync_once
fi
