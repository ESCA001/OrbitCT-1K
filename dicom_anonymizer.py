#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dicom_anonymizer.py

Batch DICOM de-identification script with small-series removal

Features: 
1. Recursively iterate over first-level sub-folders of ``root_dir`` (each
   folder represents one examination).
2. Count slices that share the same SeriesInstanceUID. If the slice count
   is ≤ 10, delete the entire series (commonly "Exam Summary" images
   containing PHI).
3. For the remaining DICOM files, mask predefined tags and remove all
   private tags, then overwrite the original file.

Usage:
$ python dicom_anonymizer.py /path/to/dicom_1k
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import pydicom
from pydicom.errors import InvalidDicomError
from tqdm import tqdm

# ---------- Configuration ----------
FIELDS_TO_MASK = [
    ("ReferringPhysicianName", "MASKED"),
    ("PatientName",            "MASKED"),
    ("PatientID",              "MASKED"),
    ("PatientBirthDate",       "19000101"),
    ("PatientSex",             "O"),
    ("PatientAge",             "000Y"),
    ("PatientAddress",         "MASKED"),
    ("InstitutionName",        "MASKED"),
    ("InstitutionAddress",     "MASKED"),
    ("PerformingPhysicianName","MASKED"),
    ("OperatorsName",          "MASKED"),
    ("AccessionNumber",        "MASKED"),  
]
SMALL_SERIES_THRESHOLD = 10  # slices or fewer will be removed
# -----------------------------------


def collect_series(folder_path: str) -> dict[str, list[str]]:
    """Return a mapping {SeriesInstanceUID: [file1, file2, ...]}."""
    series_map: dict[str, list[str]] = defaultdict(list)
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".dcm"):
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            series_uid = getattr(ds, "SeriesInstanceUID", None)
            if series_uid:
                series_map[series_uid].append(fpath)
        except InvalidDicomError:
            # Skip non-DICOM files
            continue
    return series_map


def delete_small_series(series_map: dict[str, list[str]],
                        threshold: int = SMALL_SERIES_THRESHOLD) -> int:
    """Delete entire series whose slice count ≤ threshold.

    Returns
    -------
    int
        Total number of files removed.
    """
    removed = 0
    for _, files in series_map.items():
        if len(files) <= threshold:
            for fp in files:
                try:
                    os.remove(fp)
                    removed += 1
                except OSError:
                    continue
    return removed


def anonymize_file(file_path: str) -> None:
    ds = pydicom.dcmread(file_path, force=True)

    # Mask predefined standard DICOM tags containing identifying information
    for tag, value in FIELDS_TO_MASK:
        if tag in ds:
            ds.data_element(tag).value = value

    # Remove all private DICOM tags
    for tag in [t for t in ds.keys() if t.is_private]:
        del ds[tag]

    ds.save_as(file_path)


def process_exam_folder(folder_path: str) -> None:
    """Process a single examination folder."""
    series_map = collect_series(folder_path)
    _ = delete_small_series(series_map)

    # Re-scan remaining DICOM files and anonymize
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".dcm"):
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            anonymize_file(fpath)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {fpath}: {exc}")


def main(root_dir: str) -> None:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"{root_dir} is not a valid directory.")

    exam_folders = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    for folder in tqdm(exam_folders, desc="Exam folders"):
        process_exam_folder(folder)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dicom_anonymizer.py <root_dicom_dir>")
        sys.exit(1)
    main(sys.argv[1])
