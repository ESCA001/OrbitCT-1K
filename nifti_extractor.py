#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nifti_extractor.py

Extract axial non-contrast CT series to NIfTI (.nii.gz) with preprocessing + optional resampling.

Pipeline:
1) Identify axial non-contrast CT series based on DICOM metadata:
   - Modality == "CT"
   - Axial orientation: either "AXIAL" in ImageType, or slice-normal from ImageOrientationPatient
     aligned dominantly with the superior-inferior axis.
2) Sort slices geometrically using ImagePositionPatient.
3) Convert pixels to HU using per-slice RescaleSlope and RescaleIntercept.
4) (Optional) Series with an effective slice spacing outside a user-defined range (e.g., 0.5–1.5 mm), 
    computed from geometric slice positions, were excluded.
5) Build a coarse body mask: threshold HU at -200 and keep the largest 3D connected component;
   set voxels outside to -1000 HU (air).
6) Resample masked HU volume to isotropic target spacing (default 0.5 mm) using linear interpolation;
   source spacing is derived from PixelSpacing and slice-to-slice geometric distances.
7) Standardize in-plane FOV: center using the centroid of the body mask, then crop/pad to 512×512.
8) Build affine by converting DICOM LPS orientation vectors to RAS and embedding target voxel spacing;
   set origin to (0,0,0).
9) Save as compressed NIfTI (.nii.gz).

Usage:
$ python nifti_extractor.py /path/to/parent_folder --out ./nii
"""

from tqdm import tqdm
import os, argparse
import numpy as np
import pandas as pd
import pydicom
from collections import OrderedDict
import nibabel as nib
from scipy.ndimage import label, zoom

AXIAL_NZ_THRESHOLD = 0.75
DEFAULT_TARGET_SPACING = (0.5, 0.5, 0.5)  # (sx, sy, sz) in mm
FOV_XY = 512  # target in-plane pixel size per slice after padding/cropping


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def is_axial_series(dicom_list) -> bool:
    ds0 = dicom_list[0]
    modality = str(getattr(ds0, "Modality", "")).upper()
    if modality != "CT":
        return False
    image_type = getattr(ds0, "ImageType", [])
    tokens = [str(t).upper() for t in image_type]
    if any("AXIAL" in t for t in tokens):
        return True
    iop = getattr(ds0, "ImageOrientationPatient", None)
    if iop is None or len(iop) != 6:
        return False
    row = np.array(iop[:3], dtype=float)
    col = np.array(iop[3:], dtype=float)
    n = np.cross(row, col)
    n /= (np.linalg.norm(n) + 1e-8)
    nx, ny, nz = np.abs(n)
    return (nz >= AXIAL_NZ_THRESHOLD) and (nz >= nx) and (nz >= ny)


def largest_cc_3d(mask: np.ndarray) -> np.ndarray:
    labeled, n = label(mask)
    if n == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return labeled == largest


def sort_slices(dicom_list):
    if all(hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient") for ds in dicom_list):
        iop = np.array(dicom_list[0].ImageOrientationPatient, dtype=float)
        row, col = iop[:3], iop[3:]
        slice_cos = np.cross(row, col)
        slice_cos /= (np.linalg.norm(slice_cos) + 1e-8)
        dicom_list.sort(key=lambda d: np.dot(d.ImagePositionPatient, slice_cos))
        return dicom_list, slice_cos
    else:
        dicom_list.sort(key=lambda d: getattr(d, "InstanceNumber", 0))
        return dicom_list, np.array([0, 0, 1], dtype=float)


def stack_pixels_sorted(dicom_list_sorted):
    arrs = []
    h, w = dicom_list_sorted[0].pixel_array.shape[:2]
    for ds in dicom_list_sorted:
        arr = ds.pixel_array
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.shape != (h, w):
            raise ValueError("Inconsistent slice shapes")
        arrs.append(arr)
    return np.stack(arrs, axis=0)  # (Z,Y,X)


def rescale_to_hu(stack, dicoms):
    hu = np.empty_like(stack, dtype=np.float32)
    for i, ds in enumerate(dicoms):
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        hu[i] = stack[i].astype(np.float32) * slope + inter
    return hu


def compute_spacing(dicoms, slice_cos):
    px_sp = getattr(dicoms[0], "PixelSpacing", [1.0, 1.0])
    row_spacing, col_spacing = float(px_sp[0]), float(px_sp[1])  # DICOM: [Row, Column]
    dz = None
    if all(hasattr(ds, "ImagePositionPatient") for ds in dicoms) and len(dicoms) > 1:
        pos = [np.dot(ds.ImagePositionPatient, slice_cos) for ds in dicoms]
        diffs = np.diff(pos)
        finite = np.isfinite(diffs)
        if np.any(finite):
            dz = float(np.median(np.abs(diffs[finite])))
    if dz is None or dz <= 0:
        dz = float(getattr(dicoms[0], "SliceThickness", 1.0))
    if dz <= 0 or not np.isfinite(dz):
        dz = 1.0
    return row_spacing, col_spacing, dz


def build_affine_ras(direction_row_lps, direction_col_lps, direction_slice_lps,
                     spacing_rowcol_target, spacing_slice_target, origin_xyz=(0, 0, 0)):
    row = np.array(direction_row_lps, dtype=float)
    col = np.array(direction_col_lps, dtype=float)
    slc = np.array(direction_slice_lps, dtype=float)
    row /= (np.linalg.norm(row) + 1e-8)
    col /= (np.linalg.norm(col) + 1e-8)
    slc /= (np.linalg.norm(slc) + 1e-8)
    row = -row  # LPS->RAS
    col = -col
    row_spacing_target, col_spacing_target = spacing_rowcol_target  # [RowSpacing(target sy), ColumnSpacing(target sx)]
    aff = np.eye(4, dtype=np.float32)
    aff[:3, 0] = row * col_spacing_target   # X axis (columns)
    aff[:3, 1] = col * row_spacing_target   # Y axis (rows)
    aff[:3, 2] = slc * spacing_slice_target # Z axis
    aff[:3, 3] = np.array(origin_xyz, dtype=np.float32)
    return aff


def center_crop_pad_to_fov(vol, fov_xy=512, center_xy=None, pad_value=-1000.0):
    """
    Crop/pad based on a center point and return (Z, fov_xy, fov_xy).

    vol: (Z, Y, X)
    center_xy: (cy, cx) in pixel coordinates; if None, use geometric center
    pad_value: padding value (HU air = -1000)
    """
    Z, Y, X = vol.shape
    if center_xy is None:
        cy = (Y - 1) / 2.0
        cx = (X - 1) / 2.0
    else:
        cy, cx = float(center_xy[0]), float(center_xy[1])

    y0 = int(round(cy - fov_xy / 2.0))
    y1 = y0 + fov_xy
    x0 = int(round(cx - fov_xy / 2.0))
    x1 = x0 + fov_xy

    y0_src = max(0, y0); y1_src = min(Y, y1)
    x0_src = max(0, x0); x1_src = min(X, x1)

    y0_dst = max(0, -y0)
    x0_dst = max(0, -x0)

    out = np.full((Z, fov_xy, fov_xy), pad_value, dtype=vol.dtype)

    if (y1_src > y0_src) and (x1_src > x0_src):
        out[:, y0_dst:y0_dst + (y1_src - y0_src), x0_dst:x0_dst + (x1_src - x0_src)] = \
            vol[:, y0_src:y1_src, x0_src:x1_src]

    return out


def estimate_body_center_xy(mask_zyx: np.ndarray):
    """
    Estimate the 2D centroid (cy, cx) from a 3D mask (Z,Y,X).
    If the mask is empty, fall back to the geometric center.
    """
    Z, Y, X = mask_zyx.shape
    idx = np.argwhere(mask_zyx)  # (N, 3) -> z,y,x
    if idx.size == 0:
        return ((Y - 1) / 2.0, (X - 1) / 2.0)
    cy = float(idx[:, 1].mean())
    cx = float(idx[:, 2].mean())
    return (cy, cx)


def trim_constant_border(vol, value=-1000.0, max_trim=2):
    Z, Y, X = vol.shape
    trim_top = 0
    while trim_top < max_trim and Y - trim_top > 1 and np.all(vol[:, trim_top, :] == value):
        trim_top += 1
    trim_bottom = 0
    while trim_bottom < max_trim and Y - trim_bottom - 1 >= 0 and np.all(vol[:, Y - 1 - trim_bottom, :] == value):
        trim_bottom += 1
    trim_left = 0
    while trim_left < max_trim and X - trim_left > 1 and np.all(vol[:, :, trim_left] == value):
        trim_left += 1
    trim_right = 0
    while trim_right < max_trim and X - trim_right - 1 >= 0 and np.all(vol[:, :, X - 1 - trim_right] == value):
        trim_right += 1
    if any([trim_top, trim_bottom, trim_left, trim_right]):
        vol = vol[:, trim_top:Y - trim_bottom if trim_bottom > 0 else Y, trim_left:X - trim_right if trim_right > 0 else X]
    return vol


def resample_volume(vol, src_spacing, tgt_spacing, order=1, mode="constant", cval=-1000.0):
    sx_src, sy_src, sz_src = src_spacing
    sx_tgt, sy_tgt, sz_tgt = tgt_spacing
    zoom_z = sz_src / sz_tgt
    zoom_y = sy_src / sy_tgt
    zoom_x = sx_src / sx_tgt
    zoom_z = float(zoom_z) if np.isfinite(zoom_z) and zoom_z > 0 else 1.0
    zoom_y = float(zoom_y) if np.isfinite(zoom_y) and zoom_y > 0 else 1.0
    zoom_x = float(zoom_x) if np.isfinite(zoom_x) and zoom_x > 0 else 1.0
    return zoom(vol, (zoom_z, zoom_y, zoom_x), order=order, mode=mode, cval=cval)


def load_dicom_series(folder):
    series_dict = OrderedDict()
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".dcm"):
                ds = pydicom.dcmread(os.path.join(root, f), force=True)
                uid = getattr(ds, "SeriesInstanceUID", None)
                if uid:
                    series_dict.setdefault(uid, []).append(ds)
    return series_dict


def extract_axial_ncct_to_nii(parent, out_root, log_file,
                              thickness_range=None,
                              target_spacing=DEFAULT_TARGET_SPACING,
                              csv_name="series_index.csv",
                              resample_order=1):
    ensure_dir(out_root)
    out_dir = os.path.join(out_root, "nii_files")
    ensure_dir(out_dir)
    csv_path = os.path.join(out_root, csv_name)
    ensure_dir(os.path.dirname(log_file) if os.path.dirname(log_file) else ".")
    log = open(log_file, "a", encoding="utf-8")

    rows = []
    for sub in tqdm(sorted(os.listdir(parent)), desc="Scanning folders"):
        sub_path = os.path.join(parent, sub)
        if not os.path.isdir(sub_path):
            continue

        # Keep original naming logic as a best-effort; fall back to folder name.
        parts = sub.split("_")
        dicom_id = parts[-2] if len(parts) >= 2 else sub

        series_dict = load_dicom_series(sub_path)
        series_saved = 0

        for uid in sorted(series_dict.keys()):
            dicoms = series_dict[uid]
            if not dicoms:
                continue
            if not is_axial_series(dicoms):
                continue

            try:
                dicoms_sorted, slice_cos_lps = sort_slices(dicoms)
                vol_px = stack_pixels_sorted(dicoms_sorted)
                if vol_px.shape[0] < 2:
                    continue

                vol_hu = rescale_to_hu(vol_px, dicoms_sorted)
                row_sp, col_sp, dz = compute_spacing(dicoms_sorted, slice_cos_lps)

                if thickness_range is not None:
                    tmin, tmax = thickness_range
                    if not (tmin <= dz <= tmax):
                        continue

                # Build a coarse body mask on HU, keep the largest connected component, set outside to air (-1000 HU)
                mask_body = largest_cc_3d(vol_hu > -200)
                vol_hu_masked = mask_body * vol_hu + (-1000.0) * (~mask_body)

                # Resample in HU space; src_spacing is (sx, sy, sz) == (col_spacing, row_spacing, dz)
                vol_hu_rs = resample_volume(
                    vol_hu_masked,
                    (col_sp, row_sp, dz),
                    target_spacing,
                    order=resample_order,
                    mode="constant",
                    cval=-1000.0
                )

                # Raw HU output (no windowing/normalization, no zero_air shifting)
                vol_proc = vol_hu_rs.astype(np.float32)
                pad_value = -1000.0

                # Trim constant border then enforce 512×512 FOV using body centroid
                vol_proc = trim_constant_border(vol_proc, value=pad_value, max_trim=2)
                mask_body_rs = largest_cc_3d(vol_hu_rs > -200)
                cy, cx = estimate_body_center_xy(mask_body_rs)
                vol_proc = center_crop_pad_to_fov(vol_proc, fov_xy=FOV_XY, center_xy=(cy, cx), pad_value=pad_value)

                # Suppress tiny numerical residues (keep air/background clean)
                vol_proc[np.abs(vol_proc) < 1e-6] = 0.0

                # Build affine with target spacings (sx, sy, sz), origin set to (0,0,0)
                iop = getattr(dicoms_sorted[0], "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
                row_lps, col_lps = iop[:3], iop[3:]
                row_spacing_target = target_spacing[1]  # Y
                col_spacing_target = target_spacing[0]  # X
                affine = build_affine_ras(
                    row_lps, col_lps, slice_cos_lps,
                    spacing_rowcol_target=(row_spacing_target, col_spacing_target),
                    spacing_slice_target=target_spacing[2],
                    origin_xyz=(0.0, 0.0, 0.0)
                )

            except Exception as e:
                log.write(f"[SKIP] dicom_id={dicom_id}, series_uid={uid}, error={repr(e)}\n")
                continue

            series_saved += 1
            sid = series_saved
            out_name = f"{dicom_id}_{sid}.nii.gz"
            out_path = os.path.join(out_dir, out_name)

            vol_out_nifti = np.transpose(vol_proc, (2, 1, 0))  # (X,Y,Z)
            img = nib.Nifti1Image(vol_out_nifti, affine.astype(np.float32))
            nib.save(img, out_path)

            z, y, x = vol_proc.shape
            rows.append({
                "dicom_id": dicom_id,
                "series_id": sid,
                "series_uid": uid,
                "series_name": getattr(dicoms_sorted[0], "SeriesDescription", "UNKNOWN"),
                "size_x": int(x),
                "size_y": int(y),
                "size_z": int(z),
                "orig_row_spacing": float(row_sp),
                "orig_col_spacing": float(col_sp),
                "orig_spacing_z": float(dz),
                "tgt_spacing_x": float(target_spacing[0]),
                "tgt_spacing_y": float(target_spacing[1]),
                "tgt_spacing_z": float(target_spacing[2]),
                "path": out_path
            })

    if rows:
        df = pd.DataFrame(rows).sort_values(["dicom_id", "series_id"])
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    log.close()
    print(f"Done. CSV saved to {csv_path}")


def parse_tuple(s: str):
    s = s.strip().replace("(", "").replace(")", "")
    parts = [p for p in s.split(",") if p != ""]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("target spacing must be three comma-separated numbers, e.g., 0.5,0.5,0.5")
    try:
        vals = tuple(float(p) for p in parts)
    except Exception:
        raise argparse.ArgumentTypeError("target spacing values must be numeric")
    if any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("target spacing must be > 0")
    return vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parent_folder", help="Root directory containing case subfolders with DICOM files.")
    p.add_argument("--out", default="./nii", help="Output root directory.")
    p.add_argument("--log", default="./nii/extract.log", help="Log file path.")
    p.add_argument("--thickness", nargs=2, type=float, metavar=("MIN", "MAX"),
                   help="Only process series whose effective slice spacing dz lies in [MIN, MAX] (mm).")
    p.add_argument("--target_spacing", type=parse_tuple, default="0.5,0.5,0.5",
                   help="Resample target spacing in mm as (sx,sy,sz). Default=0.5,0.5,0.5")
    p.add_argument("--resample_order", type=int, default=1,
                   help="Interpolation order for scipy.ndimage.zoom: 0=nearest, 1=linear, 3=cubic. Default=1")
    p.add_argument("--csv_name", default="series_index.csv", help="Output CSV filename (stored under --out).")

    a = p.parse_args()
    ensure_dir(a.out)
    thickness_range = tuple(a.thickness) if a.thickness is not None else None

    extract_axial_ncct_to_nii(
        a.parent_folder,
        a.out,
        a.log,
        thickness_range=thickness_range,
        target_spacing=a.target_spacing,
        csv_name=a.csv_name,
        resample_order=a.resample_order
    )


if __name__ == "__main__":
    main()
