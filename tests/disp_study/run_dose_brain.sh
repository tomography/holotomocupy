#!/bin/bash
# Dose-matched comparison of a four-distance and a single-distance acquisition
# -- run_dose.sh, but of the brain volume instead of the synthetic phantom.
#
# The beam diverges from a focus, so at sample-to-focus distance z1 the fluence
# in the sample goes as 1/z1^2 and a far projection costs a fraction of a near
# one.  On the brain geometry (z1 = 8.25/8.60/10.01/12.95 mm) the four
# distances cost 1 + 0.920 + 0.679 + 0.406 = 3.005 near-distance projections,
# so the single-distance scan gets 3.005 x NTHETA angles and both scans put the
# same dose into the sample.  (Rounded to a multiple of NP -- the theta split
# handles a remainder, but an even one keeps the ranks balanced; at NP=4,
# 180 -> 540 and 450 -> 1352.  The phantom geometry gives 2.697 instead, so
# GEOMETRY=default changes these counts.)
#
# Everything about the sample and the reconstruction is what run_brain.sh
# currently uses -- same volume, same OBJ_SCALE, same n, amp, probe, position
# error and initial blur -- so only ndist and ntheta differ between the two
# runs, and only ntheta differs from the plain brain test.
#
#     ./run_dose_brain.sh
#     NTHETA=900 NITER=1025 ./run_dose_brain.sh
#     OBJ_SCALE=1.5 ./run_dose_brain.sh   # weaker phase; see the note below
#
# Note on OBJ_SCALE: 15 comes from the mosaic config, whose voxel is 100 nm at
# 33 keV.  Here the voxel is 24.8 nm at 17.1 keV, so the same scale gives a
# projected phase near -34 rad and a ray shear of ~100 px -- well into the
# nonlinear regime, where the single-distance leg converges badly while the
# four-distance one still manages.  That is a real result to report, but if the
# point is to compare the two acquisitions rather than to find the breakdown
# scale, run this with OBJ_SCALE around 1.5 as well.
#
set -e
cd "$(dirname "$0")"

NP=${NP:-4}                                   # MPI ranks (one GPU each)
N=${N:-1024}                                   # detector size [px]
AMP=${AMP:-32}                                # displacement half-width [px]
PRB_SMOOTH=${PRB_SMOOTH:-1}                   # probe blur sigma [px], 0 = as measured
NTHETA=${NTHETA:-900}                         # angles per distance, multi-distance scan
NDIST=${NDIST:-4}
NITER=${NITER:-513}
POS_ERR=${POS_ERR:-0}                         # initial position error [px]
OBJ_BLUR=${OBJ_BLUR:-20}                      # sigma of the initial object blur [px]
OBJ_SCALE=${OBJ_SCALE:-1}                    # multiplies the stored volume
MARGIN=${MARGIN:-192}                         # blank border on each side of the object grid
# The whole 3072 px source array is rescaled to the detector width N (1/3 at
# N=1024), and the object grid adds MARGIN blank px on each side:
#   NOBJ = N + 2*MARGIN,  1408 px at N=1024 and MARGIN=192.
# (run_dose.sh, the phantom leg, defaults to MARGIN=96 at N=512.)
NOBJ=${NOBJ:-$(( N + 2 * MARGIN ))}
PHOTONS=${PHOTONS:-0}                         # 0 = noiseless
CHECKPOINT_STEP=${CHECKPOINT_STEP:-32}
# rho is coordinate-searched before the main loop by default: the two legs of
# this comparison differ in ndist and ntheta, so the step sizes that balance
# object against probe and positions are not the same for both, and a rho tuned
# for one would handicap the other.  ESTIMATE_RHO=0 keeps the rec.py defaults.
ESTIMATE_RHO=${ESTIMATE_RHO:-0}
RHO_NITER=${RHO_NITER:-16}                    # iterations per trial of the search
OUT_ROOT=${OUT_ROOT:-/data3/vnikitin/brain_dose_study}
DPI=${DPI:-300}                               # figure resolution
ERROR_STEP=${ERROR_STEP:-32}                  # conv.csv row spacing = convergence-plot density
FREEZE_POS=${FREEZE_POS:-0}                   # 1 = do not refine positions at all
                                              # (run_dose.sh defaults this to 1; kept at 0 here
                                              # so the finished brain runs stay reproducible)
OBJ_VOL=${OBJ_VOL:-/data3/vnikitin/mosaic_brain/init.h5::exchange/data}
OBJ_BIN=${OBJ_BIN:-auto}                      # auto | 0 = off | integer block-average factor
REC_FLAGS=${REC_FLAGS:-}
if [ "$ESTIMATE_RHO" = "1" ]; then
    REC_FLAGS="$REC_FLAGS --estimate-rho --rho-estimate-niter $RHO_NITER"
fi
if [ "$FREEZE_POS" = "1" ]; then
    REC_FLAGS="$REC_FLAGS --freeze-pos"
fi
# ndist distances at once cost ndist times the memory per chunk, so the
# multi-distance run takes smaller passes than the single-distance one
NCHUNK_GEN=${NCHUNK_GEN:-8}
NCHUNK_REC=${NCHUNK_REC:-8}
# the brain slices sit around Re(obj) = -70..-20 at OBJ_SCALE=15; a range
# symmetric about 0 would spend half the greyscale on empty air
VMIN=${VMIN:--5}
VMAX=${VMAX:--2.5}
# empty VMIN/VMAX = let the figure pick the range from the 99.8th percentile of
# the ground truth.  Built with if, not [ ] && -- under set -e a failing test as
# the last command of a && list exits the script.
VRANGE=""
if [ -n "$VMIN" ]; then VRANGE="$VRANGE --vmin $VMIN"; fi
if [ -n "$VMAX" ]; then VRANGE="$VRANGE --vmax $VMAX"; fi
export OUT_ROOT
# the ID16A configuration the mosaic volume was measured in: 2.963 um
# unbinned pixel, z1 = 8.25/8.60/10.01/12.95 mm, so the nearest distance
# puts a 20.09 nm voxel on the sample at n = 2048 and 4x that at n = 512.
# Must be exported before any "import common" below.  GEOMETRY=default
# goes back to the phantom study's optics.
export DISP_STUDY_GEOMETRY=${GEOMETRY:-brain}
export DISP_STUDY_OUT="$OUT_ROOT"             # common.OUT_ROOT
export BRAIN_STUDY_OUT="$OUT_ROOT"
export BRAIN_OBJ_VOL="$OBJ_VOL"

PYPATH="import sys;sys.path.insert(0,'.');import common as C;"
NTHETA1=$(python -c "${PYPATH}print(int(round(C.dose_equivalent_ntheta($NTHETA, C.Z1_ALL[:$NDIST])/$NP))*$NP)")
FACTOR=$(python -c "${PYPATH}print('%.4f' % (C.dose_weights(C.Z1_ALL[:$NDIST]).sum()))")
name_of() { python -c "${PYPATH}print(C.dose_case_name($1,$2,$AMP,$PRB_SMOOTH))"; }

DIR_M=$OUT_ROOT/$(name_of "$NDIST" "$NTHETA")     # the multi-distance scan
DIR_1=$OUT_ROOT/$(name_of 1 "$NTHETA1")           # the dose-matched single-distance one

# common.fill_volume block-averages the source, so it reads every voxel: at
# 3072^3 that is the whole 116 GB file per gen_data run (~150 s, ~90 % of it
# disk), and this script generates twice.  Binning the source once to a grid
# still 2x wider than the object removes almost all of it -- the bin-8 copy is
# 226 MB and fill_volume's own antialiasing does the rest.  Built on the first
# run beside the source, reused by every later one.  OBJ_BIN=0 reads the source
# as it is, which is what the existing brain_study results were made from;
# OBJ_BIN=<int> forces a factor.
if [ "$OBJ_BIN" != "0" ]; then
    if [ "$OBJ_BIN" = "auto" ]; then
        PLAN=$(python downsample_volume.py --in "$OBJ_VOL" --span "$N" --plan)
    else
        PLAN=$(python downsample_volume.py --in "$OBJ_VOL" --factor "$OBJ_BIN" --plan)
    fi
    BIN_F=${PLAN%% *}
    BIN_SPEC=${PLAN#* }
    if [ "$BIN_F" != "1" ]; then
        if [ ! -f "${BIN_SPEC%%::*}" ]; then
            echo "binning the source volume ${BIN_F}x -- once, later runs reuse it"
            python downsample_volume.py --in "$OBJ_VOL" --factor "$BIN_F" \
                --out "${BIN_SPEC%%::*}"
        fi
        OBJ_VOL=$BIN_SPEC
        export BRAIN_OBJ_VOL="$OBJ_VOL"
    fi
fi

mkdir -p "$OUT_ROOT"
RUN="mpirun -np $NP ./set_affinity_gpu.sh python"

echo "=============== dose-matched brain comparison ==============="
echo "  volume : $OBJ_VOL  x $OBJ_SCALE, rescaled to $N px + ${MARGIN} px margin"
echo "  optics : n = $N,  amp = +-$AMP px,  probe sigma = $PRB_SMOOTH px"
echo "  recon  : $NITER BH iterations, pos err +-$POS_ERR px, obj blur $OBJ_BLUR px"
echo "  rho    : $([ "$ESTIMATE_RHO" = 1 ] && echo "searched, $RHO_NITER iters per trial" || echo "rec.py default")"
echo "  dose of $NDIST distances = $FACTOR near-distance projections"
echo "  $NDIST x $NTHETA angles   ->  $DIR_M"
echo "  1 x $NTHETA1 angles  ->  $DIR_1"
echo

# --- the multi-distance scan -------------------------------------------------
tag=$(basename "$DIR_M")
$RUN gen_data_brain.py --n "$N" --ntheta "$NTHETA" --ndist "$NDIST" --amp "$AMP" \
    --prb-smooth "$PRB_SMOOTH" --obj-vol "$OBJ_VOL" --obj-scale "$OBJ_SCALE" \
    --nobj "$NOBJ" --photons "$PHOTONS" --out "$DIR_M" \
    --nchunk $((NCHUNK_GEN / NDIST > 0 ? NCHUNK_GEN / NDIST : 1)) \
    2>&1 | tee -a "$OUT_ROOT/log_gen_${tag}.txt"
$RUN rec_brain.py --in "$DIR_M" --niter "$NITER" --pos-err "$POS_ERR" \
    --obj-blur "$OBJ_BLUR" --checkpoint-step "$CHECKPOINT_STEP" --error-step "$ERROR_STEP" \
    --nchunk $((NCHUNK_REC / NDIST > 0 ? NCHUNK_REC / NDIST : 1)) $REC_FLAGS \
    2>&1 | tee -a "$OUT_ROOT/log_rec_${tag}.txt"

# --- the dose-matched single-distance scan -----------------------------------
tag=$(basename "$DIR_1")
$RUN gen_data_brain.py --n "$N" --ntheta "$NTHETA1" --ndist 1 --amp "$AMP" \
    --prb-smooth "$PRB_SMOOTH" --obj-vol "$OBJ_VOL" --obj-scale "$OBJ_SCALE" \
    --nobj "$NOBJ" --photons "$PHOTONS" --out "$DIR_1" \
    --nchunk "$NCHUNK_GEN" \
    2>&1 | tee -a "$OUT_ROOT/log_gen_${tag}.txt"
$RUN rec_brain.py --in "$DIR_1" --niter "$NITER" --pos-err "$POS_ERR" \
    --obj-blur "$OBJ_BLUR" --checkpoint-step "$CHECKPOINT_STEP" --error-step "$ERROR_STEP" \
    --nchunk "$NCHUNK_REC" $REC_FLAGS \
    2>&1 | tee -a "$OUT_ROOT/log_rec_${tag}.txt"

# same colour range as make_figure_brain.py, so the panels are comparable with
# slices_brain_*.png by eye
python compare_dose.py --root "$OUT_ROOT" --ndist "$NDIST" --ntheta "$NTHETA" \
    --ntheta1 "$NTHETA1" --np "$NP" --amp "$AMP" --prb-smooth "$PRB_SMOOTH" \
    --tag brain $VRANGE --crop 0.10 --dpi "$DPI"

echo
echo "=============== summary ==============="
for d in "$DIR_M" "$DIR_1"; do
    f=$(ls -t "$d"/rec*/summary.txt 2>/dev/null | head -1)
    if [ -n "$f" ]; then echo "$(basename "$d")  $(grep nrmse_obj "$f")"; else echo "$(basename "$d")  no summary.txt"; fi
done
