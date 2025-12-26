# Scripts for OrbitCT-1K Preparation and Preprocessing

This repository contains the scripts used for dataset preparation and preprocessing in our OrbitCT-1K dataset.

The overall workflow consists of two main steps:
1. **DICOM de-identification** to remove private information from raw clinical imaging data.
2. **Automated preprocessing and format conversion** of selected axial non-contrast CT (NCCT) series from DICOM to NIfTI.

## Recommended directory structure

```text
root_dicom_dir/
├── exam_001/
│   ├── *.dcm
├── exam_002/
│   ├── *.dcm
└── ...
```

Each first-level subdirectory is treated as one examination.

## Dependencies

Install dependencies via:

```bash
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License.

# Script1: DICOM De-identification Script (dicom_anonymizer.py)

This script performs batch de-identification of clinical DICOM files prior to data preprocessing and public release.  
It is designed to remove private information while preserving imaging content and series structure.

## Anonymization pipeline

For each examination folder:
1. Recursively iterate over all DICOM files within the folder.
2. Group slices by `SeriesInstanceUID`.
3. Remove entire series whose slice count is less than or equal to a predefined threshold (default: ≤ 10 slices),
   which commonly correspond to scout images or exam summary series containing private information.
4. For the remaining DICOM files:
   - Mask predefined standard DICOM tags containing identifying information.
   - Remove all private DICOM tags.
5. Overwrite the original DICOM files with anonymized versions.

## Masked DICOM fields

The following standard DICOM attributes are replaced with anonymized values:
- PatientName
- PatientID
- PatientBirthDate
- PatientSex
- PatientAge
- PatientAddress
- ReferringPhysicianName
- PerformingPhysicianName
- OperatorsName
- InstitutionName
- InstitutionAddress
- AccessionNumber

All private DICOM tags are removed.

## Usage

```bash
python dicom_anonymizer.py /path/to/root_dicom_dir
```

The script modifies files in place.

It is strongly recommended to operate on a copy of the original DICOM data.

## Notes

- This script should be executed before any downstream preprocessing or format conversion.

- The anonymization strategy follows a conservative approach suitable for public data release.

- Users should verify compliance with local ethics and data governance requirements before distribution.


# Script2: Axial Non-Contrast CT (NCCT) DICOM → NIfTI Preprocessing Script (nifti_extractor.py)

This repository provides an automated preprocessing pipeline to convert axial non-contrast CT series from DICOM into compressed NIfTI (`.nii.gz`) for downstream analysis.

## Preprocess pipeline

NCCT series were pre-selected prior to running the preprocessing script. 
For each DICOM series:
1. Identify axial NCCT series based on DICOM metadata:
   - `Modality == "CT"`
   - Axial orientation: `"AXIAL"` in `ImageType`, or computed slice-normal from `ImageOrientationPatient` aligned dominantly with the superior-inferior axis.
2. Sort slices geometrically using `ImagePositionPatient`.
3. Convert pixels to HU using per-slice `RescaleSlope` and `RescaleIntercept`.
4. (Optional) Series with an effective slice spacing outside a user-defined range (e.g., 0.5–1.5 mm), computed from geometric slice positions, were excluded.
5. Create a coarse body mask: threshold HU volume at `-200 HU`, keep the largest 3D connected component;
   set voxels outside the mask to `-1000 HU` (air).
6. Resample the masked HU volume to isotropic target spacing (default `0.5×0.5×0.5 mm`) with linear interpolation.
7. Standardize in-plane size: center by the body-mask centroid, then crop/pad to `512×512`.
8. Build an affine matrix by converting DICOM LPS orientation vectors to RAS coordinates and embedding target spacing.
9. Save the processed volume as `.nii.gz`.

## Usage
Basic usage:

The `--thickness` argument is optional.  
If not specified, all axial NCCT series are processed regardless of slice spacing.

```bash
python nifti_extractor.py /path/to/parent_folder --out ./nii --log ./nii/extract.log
```

Use the `--thickness` argument to filter thin-section series by slice spacing (dz) range (in mm), e.g. 0.5–1.5 mm:

```bash
python nifti_extractor.py /path/to/parent_folder \
  --out ./nii \
  --thickness 0.5 1.5 \
  --log ./nii/extract.log
```

## Output CSV
series_index.csv records:
- case identifier (dicom_id derived from folder name)
- SeriesInstanceUID, SeriesDescription
- original and target spacing
- output path

