"""Config-file parsing for the pipeline scripts.

Every step reads one flat key = value file. The parsers below differ only
in *which* keys they read; the mechanics (implicit [DEFAULT] section, relative
path resolution, missing-field reporting) live in _Cfg.
"""

import os
import configparser
from types import SimpleNamespace

_MISSING = object()


class _Cfg:
    """Thin wrapper over a configparser section with clear error reporting.

    SectionProxy.get()/getint()/... return None for an absent key instead of
    raising NoOptionError, so a missing required field used to surface far from
    its cause (an AttributeError on None inside os.path.join or .rstrip). Here,
    an accessor called without an explicit ``fallback`` is required and raises
    ValueError naming both the config file and the key.
    """

    def __init__(self, cfg, source, here):
        self._c = cfg
        self._source = source
        self._here = here

    @property
    def here(self):
        """Directory holding the config file."""
        return self._here

    def _conv(self, fn, key, fallback):
        if key not in self._c:
            if fallback is _MISSING:
                raise ValueError(f"Missing required field in {self._source}: {key}")
            return fallback
        try:
            return fn(key)
        except ValueError as e:
            raise ValueError(f"Bad value for '{key}' in {self._source}: {e}") from e

    def str(self, key, fallback=_MISSING):
        return self._conv(self._c.get, key, fallback)

    def int(self, key, fallback=_MISSING):
        return self._conv(self._c.getint, key, fallback)

    def float(self, key, fallback=_MISSING):
        return self._conv(self._c.getfloat, key, fallback)

    def bool(self, key, fallback=_MISSING):
        return self._conv(self._c.getboolean, key, fallback)

    def path(self, key, fallback=_MISSING):
        """A string field with any trailing '/' stripped."""
        v = self.str(key, fallback)
        return v.rstrip('/') if isinstance(v, str) else v

    def opt_str(self, key):
        """Optional string: absent, empty, or whitespace-only all give None."""
        v = self._c.get(key, fallback=None)
        return v.strip() if v and v.strip() else None

    def list(self, key, cast=str, fallback=_MISSING, sep=","):
        s = self.str(key, "" if fallback is _MISSING else fallback)
        return [cast(x.strip()) for x in s.split(sep) if x.strip()]

    def rel(self, p):
        """Resolve a path given relative to the config file's directory."""
        return p if os.path.isabs(p) else os.path.join(self._here, p)


def _load(config_file, interpolation=configparser.BasicInterpolation()):
    """Read a flat key = value file as if every key were in [DEFAULT]."""
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",),
                                       interpolation=interpolation)
    with open(config_file, "r", encoding="utf-8") as f:
        parser.read_string("[DEFAULT]\n" + f.read())
    here = os.path.dirname(os.path.abspath(config_file))
    return _Cfg(parser["DEFAULT"], config_file, here)


def get_list(c, key, cast=str, sep=","):
    """Back-compatible helper for callers holding a raw configparser section."""
    s = c.get(key, fallback="")
    return [cast(x.strip()) for x in s.split(sep) if x.strip()]


_MODELS = ('intensity', 'amplitude')


def _parse_model(cfg):
    """The data-misfit model: 'amplitude' (default) or 'intensity'.

    Both compare the SAME forward intensity K|psi|^2 -- where K is the one
    Gaussian of psf_sigma -- against the same measured intensity d.  They differ
    only in the space the comparison is made in.  With a = sqrt(K|psi|^2):

        intensity   F0 = 1/N sum W (a^2 - d  )^2
        amplitude   F0 = 1/N sum W (a   - sqrt(d))^2

    so psf_sigma is orthogonal to this knob and works in both; amplitude at
    psf_sigma=0 is exactly the misfit the solver used before it moved to
    intensity.

    Which is right is a noise question.  sqrt is the variance-stabilizing
    transform of the Poisson distribution, so under photon-counting noise the
    residuals of the AMPLITUDE model have constant variance and it is the
    correctly weighted least squares; under additive/read-dominated noise that
    is true of the INTENSITY model instead.  The two are related by
    (a^2 - d) = (a - sqrt(d))(a + sqrt(d)), i.e. the intensity misfit is the
    amplitude misfit with an extra per-pixel weight (a + sqrt(d))^2 ~ 4d, which
    is exactly the Poisson variance -- it over-weights bright pixels by their
    own intensity.  On noiseless, exactly-modelled data both have the same
    global minimum.

    Practical consequences, all from that ~4d factor (data are flat-field
    normalized, so d ~ 1 and the factor is ~4):

      * err is ~4x SMALLER under amplitude and the two are not comparable.
      * lam_laplacian must be divided by ~4 when switching to amplitude.  It
        weighs a pure object penalty, which does not care what space the data
        misfit lives in, against an F0 that just shrank ~4x -- an unscaled lam
        would quietly be a 4x heavier regularizer.
      * lam_prbfit, by contrast, CARRIES OVER UNCHANGED.  PrbfitTerm follows
        this same knob, so its residual is amplitude-type exactly when F0 is and
        shrinks by the same ~4; the ratio it multiplies is already invariant.
        That is why it was tied to the knob rather than pinned to intensity.
      * rho is a ratio between variable blocks and the F0 part of every block
        rescales together, so rho carries over unchanged once the lam's do.
      * amplitude divides by a and by a^3, so unlike intensity it needs a floor
        (Rec._AMP_FLOOR) and can stiffen where the model predicts near-zero.
    """
    # DEFAULT CHANGED 2026-10-03: amplitude, not intensity.  Measured on the
    # synthetic pair (see experimental/README), amplitude needs ~2.4x fewer
    # iterations purely from conditioning, and it is the correctly weighted
    # least squares under photon-counting noise.  Every config that pins
    # model= explicitly is unaffected; a config that does not now gets
    # amplitude, and its lam_laplacian wants dividing by ~4 (see above) and
    # its err is ~4x smaller, so it is NOT comparable with that config's own
    # older runs.
    model = cfg.str("model", fallback="amplitude").strip().lower()
    if model not in _MODELS:
        raise ValueError(f"{cfg._source}: model must be one of {_MODELS}, "
                         f"got {model!r}")
    return model


def parse_args(config_file):
    """Config for the main reconstruction (step6.py)."""
    cfg = _load(config_file)
    args = SimpleNamespace()

    args.pfile    = cfg.opt_str("pfile")
    args.path_out = cfg.path("path_out")
    _path         = cfg.path("path", fallback=args.path_out)
    if args.pfile:
        args.in_file = f"{args.path_out}/{args.pfile}.h5"
    else:
        args.in_file = os.path.join(_path, cfg.str("in_file"))

    args.ntheta      = cfg.int("ntheta")
    args.start_theta = cfg.int("start_theta")
    args.nz          = cfg.int("nz")
    args.n           = cfg.int("n")
    args.nzobj       = cfg.int("nzobj")
    args.nobj        = cfg.int("nobj")
    # Detector oversampling of the Radon transform: the projection plane is
    # nobj*tomo_upsample wide while the object stays nobj wide, so the object
    # can sit on a 2x coarser x/y grid than the data.  1 (default) is the
    # historical behaviour, R: [nzobj, nobj, nobj] -> [ntheta, nzobj, nobj].
    args.tomo_upsample = cfg.int("tomo_upsample", fallback=1)
    args.ndist       = cfg.int("ndist")
    args.paganin     = cfg.int("paganin")
    args.mask        = cfg.float("mask")

    # Detector PSF: one Gaussian on the detector intensity, sigma in BINNED
    # detector px.  Per-bin, so halve it at each step of the ladder.
    args.psf_sigma = cfg.float("psf_sigma", fallback=0.0)  # binned detector px

    # Which space the data misfit is measured in, 'intensity' (default, the
    # historical behaviour) or 'amplitude'.  Orthogonal to psf_sigma: both
    # models blur the model intensity with the same K.  See _parse_model --
    # in particular that lam_laplacian must be divided by ~4 when switching to
    # amplitude while lam_prbfit carries over unchanged, and that err is not
    # comparable across the two.
    args.model = _parse_model(cfg)

    args.lam_prbfit    = cfg.float("lam_prbfit")
    args.lam_laplacian = cfg.float("lam_laplacian", fallback=0.0)

    args.rho = cfg.list("rho", float)
    # Optional rho tuning: False (default) -> args.rho used as-is; True ->
    # BH first coordinate-searches rho[prb, pos] around args.rho with short
    # trials of rho_estimate_niter iterations. See Rec.estimate_rho_coord.
    args.estimate_rho       = cfg.bool("estimate_rho",       fallback=False)
    args.rho_estimate_niter = cfg.int ("rho_estimate_niter", fallback=16)
    # -1 (default) keeps estimate_rho_coord's trials silent; N > 0 logs the
    # error every N iterations inside each trial.
    args.rho_trial_error_step = cfg.int("rho_trial_error_step", fallback=-1)

    # Out-of-grid detector pixels: with eff_demag > 1 a detector pixel maps
    # back outside the object grid whenever nobj < n*max(eff_demag). Such
    # pixels carry only the shift kernel's boundary condition, so by default
    # Rec._build_data_mask drops them from the data fit. mask_oob_margin is
    # slack in detector pixels on top of the worst-case position shift.
    args.mask_oob        = cfg.bool ("mask_oob",        fallback=True)
    args.mask_oob_margin = cfg.float("mask_oob_margin", fallback=2.0)

    # Derive the CG beta/alpha from one Hessian sweep instead of three
    # (see the fused_hessian note on Rec). check_fused_hessian re-measures
    # every form the classic path would have measured and logs the relative
    # disagreement — 4 extra sweeps per iteration, for verification only.
    args.fused_hessian       = cfg.bool("fused_hessian",       fallback=True)
    args.check_fused_hessian = cfg.bool("check_fused_hessian", fallback=False)
    # Plot the real functional against the quadratic model that picked
    # alpha, on every checkpoint_step iteration (Rec.check_approximation).
    # npp extra evaluations of the full functional per triggered iteration,
    # so it is off by default. Not named check_approximation: args are
    # copied onto Rec verbatim and would shadow the method.
    args.check_approx = cfg.bool("check_approx", fallback=False)

    args.niter  = cfg.int("niter")
    args.nchunk = cfg.int("nchunk")
    # How many distances share one upload of a theta chunk of proj inside
    # the cascade kernels. 0 = all of them (the default), 1 = the old
    # outer-distance loop. See Rec._resolve_ndistchunk.
    args.ndistchunk      = cfg.int("ndistchunk", fallback=0)
    args.checkpoint_step = cfg.int("checkpoint_step")
    args.error_step      = cfg.int("error_step")
    # Internal instrumentation (cache hit/miss counters and the like).
    # -1 = never, the default: these numbers only matter when tuning the
    # solver, not when running it.
    args.debug_step = cfg.int("debug_step", fallback=-1)
    args.start_iter = cfg.int("start_iter")

    args.rotation_center_shift = cfg.float("rotation_center_shift")
    args.bin           = cfg.int("bin")

    # Step 7's answer: a per-angle shift ADDED to the positions read from the
    # h5, so a refit only costs a step-6 rerun.  Absent file = no-op.
    #
    # A relative path resolves against the CONFIG's directory, not the cwd:
    # step7.py writes the file next to the config, but polaris_run.sh runs
    # step6 from the PARENT directory, so a cwd-relative default silently
    # missed it and the rerun changed nothing.
    #
    # NOTE correct3d_bin is a FACTOR (1, 2, 4 = raw px per file px), NOT a
    # level like `bin`, and deliberately so: it must match the binning ESRF
    # stamped on the file, which steps15 cross-checks against
    # esrf_meta.bin_from_pixelsize (a physical factor), and 0 is already taken
    # as "derive it from <pfile>_rec_.info".  read_pos scales by
    # correct3d_bin / 2**bin.
    args.correct3d_extra      = cfg.int("correct3d_extra", fallback=1)
    _extra_file = cfg.str("correct3d_extra_file",
                          fallback="correct_correct3D_extra.txt")
    args.correct3d_extra_file = (_extra_file if os.path.isabs(_extra_file)
                                 else os.path.join(cfg.here, _extra_file))
    args.correct3d_bin        = cfg.int("correct3d_bin", fallback=1)
    # Shift interpolation: 'cubic' (B-spline, 4x4 taps) or 'fft' (Fourier
    # shift theorem).  'cubic' is the default HERE, unlike step 0: a
    # multi-distance scan has magnification != 1 at every plane but the
    # first, so 'fft' takes the chirp-z on nearly every call -- 7-30x slower
    # and 8-16x the scratch (tests/performance/bench_shift.py).  Step 0 is
    # single-distance, m == 1, a plain phase ramp, and defaults to 'fft'.
    args.shift_type    = cfg.str("shift_type", fallback="cubic")
    args.log_level     = cfg.str("log_level", fallback="WARNING")
    args.energy        = cfg.float("energy", fallback=None)
    args.method        = cfg.int("method",        fallback=0)
    args.start_method  = cfg.int("start_method",  fallback=1)

    _pos_chk            = cfg.opt_str("pos_checkpoint")
    args.pos_checkpoint = os.path.join(_path, _pos_chk) if _pos_chk else None
    _prb                = cfg.opt_str("prb_file")
    args.prb_file       = os.path.join(args.path_out, _prb) if _prb else None
    args.init_vol       = cfg.opt_str("init_vol")
    args.init_vol_scale = cfg.float("init_vol_scale", fallback=1.0)

    # Mosaic: one .h5 per tile, {path_out}/{pfile}_{tile}.h5.  Empty tiles=
    # is single-tile mode and leaves everything below None.  mosaic_file is
    # a NAME, not a file that has to exist: MosaicReader only uses it to
    # find the shared initial object {pfile}_obj.h5, exactly as step6 of
    # the YY037A pipeline uses {pfile}_mosaic.h5.
    args.tiles = cfg.list("tiles", str)
    if args.tiles:
        if not args.pfile:
            raise ValueError("tiles= requires pfile=")
        args.tile_files  = [f"{args.path_out}/{args.pfile}_{t}.h5" for t in args.tiles]
        args.mosaic_file = args.in_file
    else:
        args.tile_files  = None
        args.mosaic_file = None

    return args


def parse_args_step0(config_file):
    """Config for step0.py reading ESRF scan + metadata files."""
    cfg = _load(config_file, interpolation=None)
    args = SimpleNamespace()

    path             = cfg.path("path")
    args.path_out    = cfg.path("path_out")
    args.scan_file   = os.path.join(path, cfg.str("scan_file"))
    args.meta_file   = os.path.join(path, cfg.str("meta_file"))
    args.h5_out      = os.path.join(args.path_out, cfg.str("h5_out"))
    args.dataset_ids = cfg.list("dataset_ids", int)
    args.n               = cfg.int("n",               fallback=2048)
    args.niter           = cfg.int("niter",           fallback=129)
    args.nchunk          = cfg.int("nchunk",          fallback=4)
    args.checkpoint_step = cfg.int("checkpoint_step", fallback=32)
    args.error_step      = cfg.int("error_step",      fallback=32)
    args.rho             = cfg.list("rho", float)
    args.log_level       = cfg.str("log_level", fallback="INFO")

    # Detector PSF: one Gaussian on the detector intensity, sigma in detector
    # px of the grid the data is on -- unbinned for NFP, no bin factor.
    args.psf_sigma = cfg.float("psf_sigma", fallback=0.0)   # detector px
    # Data-misfit model, 'amplitude' (default) or 'intensity'; see _parse_model.
    # Orthogonal to psf_sigma -- both models blur the model intensity with K.
    args.model = _parse_model(cfg)
    # Photon energy in keV.  Optional: unset (default) means take it from the
    # scan metadata, which is what every step0.py did before this knob.  Set it
    # to override -- the recorded value is the monochromator setpoint and is
    # worth sweeping when the probe comes out with the wrong fringe spacing.
    args.energy = cfg.float("energy", fallback=None)
    # Shift interpolation: 'fft' (Fourier shift theorem, the default -- the
    # B-spline stencil error biases the position gradient) or 'cubic'.  'fft'
    # is PERIODIC, so the object grid must clear n + 2*max|pos|, which
    # step0.py's nobj already rounds up to a multiple of 32.
    args.shift_type = cfg.str("shift_type", fallback="fft")

    return args


def parse_args_step0_nx(config_file):
    """Config for step0.py reading ESRF NXtomo (.nx) files."""
    cfg = _load(config_file, interpolation=None)
    args = SimpleNamespace()

    path          = cfg.path("path")
    args.path_out = cfg.path("path_out")
    args.nx_file  = os.path.join(path, cfg.str("nx_file"))
    args.h5_out   = os.path.join(args.path_out, cfg.str("h5_out"))
    args.n               = cfg.int("n",               fallback=2048)
    args.niter           = cfg.int("niter",           fallback=129)
    args.nchunk          = cfg.int("nchunk",          fallback=4)
    args.checkpoint_step = cfg.int("checkpoint_step", fallback=-1)
    args.error_step      = cfg.int("error_step",      fallback=32)
    args.rho             = cfg.list("rho", float)
    # Optional rho tuning, as in parse_args: False (default) -> args.rho is
    # used as-is; True -> RecNFP first coordinate-searches rho[prb], then
    # rho[pos], around args.rho with silent rho_estimate_niter-iteration
    # trials.  rho[proj] is the reference scale and is left alone, exactly as
    # rho[obj] is in the 3-D search.  See RecNFP.estimate_rho_coord.
    args.estimate_rho       = cfg.bool("estimate_rho",       fallback=False)
    args.rho_estimate_niter = cfg.int ("rho_estimate_niter", fallback=16)
    # -1 (default) keeps the trials silent; N > 0 logs the error every N
    # iterations inside each trial, the only way to tell a slow descent from a
    # first-step blow-up.
    args.rho_trial_error_step = cfg.int("rho_trial_error_step", fallback=-1)
    args.log_level       = cfg.str("log_level", fallback="INFO")

    # Detector PSF: one Gaussian on the detector intensity, sigma in detector
    # px of the grid the data is on -- unbinned for NFP, no bin factor.
    args.psf_sigma = cfg.float("psf_sigma", fallback=0.0)   # detector px
    # Data-misfit model, 'amplitude' (default) or 'intensity'; see _parse_model.
    # Orthogonal to psf_sigma -- both models blur the model intensity with K.
    args.model = _parse_model(cfg)
    # Photon energy in keV.  Optional: unset (default) means take it from the
    # scan metadata, which is what every step0.py did before this knob.  Set it
    # to override -- the recorded value is the monochromator setpoint and is
    # worth sweeping when the probe comes out with the wrong fringe spacing.
    args.energy = cfg.float("energy", fallback=None)
    # Shift interpolation: 'fft' (Fourier shift theorem, the default -- the
    # B-spline stencil error biases the position gradient) or 'cubic'.  'fft'
    # is PERIODIC, so the object grid must clear n + 2*max|pos|, which
    # step0.py's nobj already rounds up to a multiple of 32.
    args.shift_type = cfg.str("shift_type", fallback="fft")

    return args


def parse_args_steps15(config_file):
    """Config for steps15.py (EDF->HDF5, preprocessing, shift combination)."""
    cfg = _load(config_file)
    args = SimpleNamespace()

    args.path     = cfg.path("path")
    args.pfile    = cfg.str("pfile")
    args.path_out = cfg.opt_str("path_out")

    args.start_step            = cfg.int  ("start_step",            fallback=1)
    args.start_level_rec       = cfg.int  ("start_level_rec",       fallback=0)
    args.rotation_center_shift = cfg.float("rotation_center_shift", fallback=0.0)
    args.nlevels   = cfg.int  ("nlevels",   fallback=4)
    args.paganin   = cfg.float("paganin",   fallback=120.0)
    args.nchunk    = cfg.int  ("nchunk",    fallback=16)
    args.ref_dist  = cfg.int  ("ref_dist",  fallback=0)

    def _src(key, allowed, fallback="esrf"):
        v = cfg.str(key, fallback=fallback).strip().lower()
        if v not in allowed:
            raise SystemExit(f"{key}={v!r} in {config_file}: expected one of "
                             + " / ".join(allowed))
        return v

    # Where the rotation axis comes from.  'config' (the default, and what
    # every pre-existing experiment folder gets) keeps the historical
    # behaviour: the number typed into rotation_center_shift below is added
    # to the horizontal shift column at step 4, step 5 and step 6
    # independently.  'measured' has step 3 estimate the axis from opposed
    # (theta, theta+180) projection pairs and fold it into cshifts_final
    # once, so the three consumers cannot disagree and the config stays at 0.
    #
    # The two are mutually exclusive by construction: with 'measured' the
    # offset is already inside cshifts_final, so a non-zero
    # rotation_center_shift would be counted a second time.  steps15 refuses
    # that combination rather than quietly doubling the axis offset.
    args.center_src = _src("center_src", ("config", "measured"),
                           fallback="config")

    # Where the sample drift comes from.  'none' (the default, and what the
    # pipeline did before the retake estimator existed) carries no drift term
    # in step 3 and leaves the whole of it to step 7, which fits it out of a
    # finished volume.  'quali' has step 3 measure it from the post-scan
    # retakes -- see estimate_quali_motion.py -- and step 7 then only refines
    # what is left.  'quali' is only worth selecting on a scan where the
    # estimator has been validated against quali.mat.
    args.motion_src = _src("motion_src", ("none", "quali"), fallback="none")

    # Take rotation_center_shift from ESRF's own nabu configs instead of the
    # number typed below.  <pfile>_/naburec/*.conf records
    # rotation_axis_position on the <pfile>_rec_ grid, which is exactly the
    # axis Peter reconstructed with, so reading it back beats re-deriving it --
    # and it cannot go stale when he re-drops the directory.  The conversion
    _n             = cfg.int  ("n",    fallback=0)
    _nobj          = cfg.int  ("nobj", fallback=0)
    args.n         = _n    if _n    > 0 else None
    args.nobj      = _nobj if _nobj > 0 else None
    args.log_level = cfg.str  ("log_level", fallback="INFO")

    # --- synthetic mosaic (tests/mosaic_brain/steps15.py) ------------------
    # All optional: the experimental single-tile steps15 scripts do not set
    # them and keep working off the fallbacks above.
    args.tiles       = cfg.list ("tiles", str)
    args.nzobj       = cfg.int  ("nzobj",       fallback=0)
    args.bin         = cfg.int  ("bin",         fallback=0)
    args.ntheta_rec  = cfg.int  ("ntheta_rec",  fallback=0)
    args.nobj_tile   = cfg.int  ("nobj_tile",   fallback=0)
    args.mask        = cfg.float("mask",        fallback=0.9)
    args.ntile_h     = cfg.int  ("ntile_h",     fallback=1)
    args.ntile_v     = cfg.int  ("ntile_v",     fallback=1)
    args.tile_step_h = cfg.float("tile_step_h", fallback=0.0)
    args.tile_step_v = cfg.float("tile_step_v", fallback=0.0)
    _sd              = cfg.opt_str("shift_dir")
    args.shift_dir   = cfg.rel(_sd) if _sd else None
    args.tile_file   = (os.path.join(args.shift_dir, "tile_offsets.txt")
                        if args.shift_dir else None)

    return args


def parse_args_gen(config_file):
    """Config for the synthetic mosaic generator (gen_data.py / make_geometry.py).

    Paths given relative to the config file are resolved against its directory,
    so the scripts can be launched from anywhere.
    """
    cfg = _load(config_file, interpolation=None)
    args = SimpleNamespace()

    args.path_out = cfg.path("path_out")
    args.pfile    = cfg.str("pfile")
    args.out_file = os.path.join(args.path_out, f"{args.pfile}.h5")

    args.energy                  = cfg.float("energy")
    args.focustodetectordistance = cfg.float("focustodetectordistance")
    args.z1                      = cfg.list ("z1", float)
    args.detector_pixelsize      = cfg.float("detector_pixelsize")
    args.ndet                    = cfg.int  ("ndet")

    args.ntheta      = cfg.int  ("ntheta")
    args.theta_range = cfg.float("theta_range", fallback=180.0)
    args.bin         = cfg.int  ("bin")

    args.ntile_h     = cfg.int  ("ntile_h")
    args.ntile_v     = cfg.int  ("ntile_v")
    args.tile_step_h = cfg.float("tile_step_h")
    args.tile_step_v = cfg.float("tile_step_v")
    args.shift_dir   = cfg.rel(cfg.str("shift_dir"))
    args.tile_file   = os.path.join(args.shift_dir, "tile_offsets.txt")
    args.nobj        = cfg.int  ("nobj")
    args.nzobj       = cfg.int  ("nzobj")

    args.shift_rand_px = cfg.float("shift_rand_px", fallback=0.0)

    args.prb_abs   = cfg.rel(cfg.str("prb_abs"))
    args.prb_phase = cfg.rel(cfg.str("prb_phase"))

    _vol             = cfg.opt_str("obj_vol")
    args.obj_vol     = cfg.rel(_vol) if _vol else None
    args.delta_beta  = cfg.float("delta_beta",  fallback=100.0)
    args.obj_span_px = cfg.float("obj_span_px", fallback=0.0)
    # Multiplies the loaded volume: the sample file has arbitrary grey
    # levels, so this is what sets the projected phase excursion.
    args.obj_scale   = cfg.float("obj_scale",   fallback=1.0)

    args.nchunk    = cfg.int("nchunk", fallback=4)
    args.log_level = cfg.str("log_level", fallback="INFO")

    return args
