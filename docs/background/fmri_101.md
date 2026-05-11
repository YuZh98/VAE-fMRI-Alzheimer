# fMRI 101 for ML Practitioners

This is the bridging note for someone who has trained models on images,
text, or time series but has never opened a NIfTI file. The goal is to
explain the data, the assumptions baked into it, and what this repository
does *not* do for you.

## What is fMRI

Functional magnetic resonance imaging (fMRI) measures the **blood-oxygen
level dependent (BOLD)** signal. Active neurons demand more oxygen; local
blood flow over-compensates a few seconds later; the resulting drop in
deoxyhemoglobin (paramagnetic) versus oxyhemoglobin (diamagnetic) changes
the local magnetic susceptibility, which an MRI scanner can pick up. BOLD
is therefore an **indirect, slow** proxy for neural activity. The
hemodynamic response peaks roughly 4-6 seconds after a stimulus and
unfolds over ~20 seconds. fMRI does not measure spikes; it measures the
vascular shadow of bulk neural activity.

## What does the data look like

A single subject's resting-state fMRI scan is a 4-D array
`(X, Y, Z, T)`:

- `X, Y, Z` index a 3-D grid of **voxels** (3-D pixels). Typical
  resolution is **3-4 mm isotropic** — so each voxel covers ~30 mm^3 of
  tissue and contains on the order of 10^5–10^6 neurons.
- `T` indexes time. Each volume is acquired at one **TR** (repetition
  time), commonly 2-3 seconds. A 6-minute resting-state run at TR = 3 s
  gives ~120 volumes; longer protocols extend to 300-1200.

After registration to a standard template, the spatial extent has fixed
dimensions. This repository targets the **MNI152 2 mm template**, whose
volumes are exactly `(91, 109, 91)`. That is the shape hard-coded in
`recvae/model.py:1-21` and validated all through the encoder/decoder
arithmetic. Multiple subjects therefore stack to a 5-D tensor
`(N, 1, 91, 109, 91, T)`, with the singleton channel dimension PyTorch
expects from `Conv3d`.

So in practice this repo expects:

- one NIfTI file per subject
- already resampled to `91 x 109 x 91` (MNI152 2 mm space)
- truncated or padded to `tol_time` (default 120) timepoints
- stored as `.nii` or `.nii.gz`

The loaders are in `recvae/data.py:39-72`.

## Units, sign, dynamic range

BOLD is **unitless**. The scanner reports a raw intensity per voxel, but
because the steady-state magnetization depends on many session-specific
factors (coil load, scanner drift, field inhomogeneity, T1 effects in
the first few volumes), absolute intensity carries little meaning.
**Only relative changes are interpretable.** A voxel's signal sits at
some arbitrary mean of a few hundred to a few thousand, fluctuates by
~1-3% around that mean during a task, and even less during rest.

This is why every fMRI pipeline normalizes. Common choices:

- **Percent signal change**: `(x_t - mean_t) / mean_t * 100`.
- **Z-score per voxel**: `(x_t - mean_t) / std_t`.
- **Global min-max** per subject to a fixed range.

This repo uses the last option (`recvae/data.py:75-106`), rescaling each
subject's full 4-D tensor into `[-1, 1]`. That's a deliberately simple
choice and the file flags two trade-offs:

1. Outliers (motion spikes, bright artifacts) compress the rest of the
   signal toward zero.
2. Per-subject statistics destroy absolute intensity differences across
   subjects — useful for invariance to scanner gain, harmful for
   classifiers that might have keyed on them.

Both are accepted research decisions in this codebase, not bugs.

## Preprocessing steps this repo does NOT do

A real fMRI pipeline runs a substantial amount of preprocessing before a
deep model ever sees the data. None of that lives here. The repository
**assumes you bring already-preprocessed scans.** What the standard
pipelines do, and you will need elsewhere:

- **Motion correction (rigid-body realignment).** Six-parameter (3
  translations + 3 rotations) rigid alignment of every volume to a
  reference, since subjects move during scanning.
- **Slice-timing correction.** Slices within one TR are acquired at
  different physical times; this interpolates to a common reference time.
- **Spatial registration to a template.** Nonlinear warp of the
  subject's anatomy into a common space (MNI152, fsaverage, ...) so
  voxels mean the same anatomical thing across subjects.
- **Spatial smoothing.** Gaussian kernel of 4-8 mm FWHM, improves SNR
  at the cost of effective resolution.
- **Scrubbing / motion censoring.** Dropping or interpolating frames
  with framewise displacement above a threshold (often 0.5 mm).
- **Nuisance regression.** Removing variance explained by motion
  parameters, mean CSF signal, mean white-matter signal, and sometimes
  global signal — these are non-neural confounds.
- **Bandpass filtering.** Resting-state analysis typically keeps
  0.01-0.1 Hz; this removes high-frequency physiological noise and slow
  scanner drift.

Pipelines that do all of the above include **fMRIPrep**, **FSL FEAT**,
**SPM**, and **AFNI**'s `afni_proc.py`. fMRIPrep is the current
community standard for getting from DICOM/raw NIfTI to a clean,
template-aligned 4-D tensor. Plan on it taking hours per subject.

The implication: if you feed a model in this repo with raw NIfTI files
straight off the scanner, the loss will dominated by motion artefacts,
not biology.

## ADNI specifics

The cohorts referenced throughout this repository come from the
**Alzheimer's Disease Neuroimaging Initiative (ADNI)** — a long-running
multi-site study of aging and dementia. Three cohort labels you'll see:

- **CN** — cognitively normal.
- **MCI** — mild cognitive impairment, an intermediate state.
- **AD** — clinically diagnosed Alzheimer's disease.

This repo's example workflow uses CN vs AD only.

ADNI is **not redistributable**. To use the data you sign a Data Use
Agreement that forbids re-sharing scans, derivatives, or subject IDs
outside the approved investigator team. Two practical consequences:

1. **No subject IDs or filesystem paths in version control.** The README
   spells this out and the example code uses placeholder paths like
   `/path/to/CN`.
2. **No scans, no derived volumes, no plots that contain identifying
   image content.** Even a single sagittal slice can in principle be
   identifying after facial reconstruction.

This repo follows those constraints — the tests use synthetic tensors
(`recvae/data.py:151-251`), so the test suite runs without any real
fMRI data on disk.

## What "120 timepoints" means

`tol_time = 120` in `recvae/config.py:29` is not arbitrary. At a typical
resting-state TR of ~3 seconds, 120 volumes = **6 minutes** of scanning.
That's long enough to capture the slow (0.01-0.1 Hz) spontaneous
fluctuations that resting-state fMRI is sensitive to: a 0.01 Hz
oscillation has a 100-second period, so you need at least a couple of
periods to see it at all. Six minutes is at the short end of the modern
recommended range (8-15 minutes for reliable functional connectivity);
ADNI's older protocols sit there.

Volumes longer than `tol_time` are truncated; shorter ones raise rather
than silently stack at the wrong shape (`recvae/data.py:65-68`).

## Why the train/test split is hard

The single biggest pitfall in fMRI ML papers, and the one most likely to
inflate a reported metric, is **mixing scans from the same subject
across folds**. Scans from one subject are not independent samples — a
model that has seen any data from subject S will know S's brain anatomy
and motion signature, and will do unfairly well on any other scan of S.

The correct procedure:

- **Subject-level split**, not scan-level. Every scan from one subject
  belongs to exactly one fold (train, val, or test).
- **Stratify on cohort label** so each fold has a representative CN/AD
  ratio.
- **Match demographics** (age, sex, scanner site) across folds where
  possible, to avoid confounded splits.
- **Reserve a held-out test set** that you query at most once.

The repo's `recvae.evaluation.split_subjects` handles the basic case of
returning subject-level k-fold indices. The brief in the README also
flags that the legacy notebook's `test_loader` aliased the training set
— a real evaluation must use a real split.

## Pointers

Software:

- **nilearn** — `nilearn.org`, a scikit-learn-style API for masking,
  connectivity, GLM, and visualization of neuroimaging data. The first
  Python library to reach for.
- **TorchIO** — `torchio.readthedocs.io`, MONAI's lighter cousin;
  PyTorch transforms for 3-D / 4-D medical images, including the
  intensity and spatial augmentations that fMRI tends to need.
- **MONAI** — `monai.io`, the PyTorch-native medical-imaging framework
  with networks, transforms, losses, metrics, and pipelines.
- **fMRIPrep** — `fmriprep.org`, the canonical preprocessing pipeline.
- **FSL**, **SPM**, **AFNI** — older but still production-grade
  preprocessing suites.

Textbooks and reviews:

- Huettel, Song, McCarthy. *Functional Magnetic Resonance Imaging*, 3rd
  ed., Sinauer/Oxford 2014 — the standard physics-and-acquisition
  textbook.
- Poldrack, Mumford, Nichols. *Handbook of Functional MRI Data
  Analysis*, Cambridge 2011 — the practical analysis companion.
- Power et al. 2014, *NeuroImage* — the "motion artefact" papers, on
  why scrubbing exists.

No specific clinical or biological claims are made in this document; for
any such claim, consult the source literature.
