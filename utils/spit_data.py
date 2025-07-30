import os
import shutil
import random
from pathlib import Path


def get_all_study_paths(root_dir):
    root = Path(root_dir)
    study_paths = []
    for patient in root.iterdir():
        if patient.is_dir():
            for study in patient.iterdir():
                if study.is_dir():
                    study_paths.append(study)
    return study_paths

def split_and_copy(study_paths, output_dir, train_ratio=0.8, seed=42):
    random.seed(seed)
    random.shuffle(study_paths)

    train_count = int(train_ratio * len(study_paths))
    train_paths = study_paths[:train_count]
    test_paths = study_paths[train_count:]

    for split_name, paths in [('train', train_paths), ('test', test_paths)]:
        split_dir = Path(output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for study_path in paths:
            patient_id = study_path.parent.name
            study_id = study_path.name
            target_dir = split_dir / f"{patient_id}_{study_id}"
            target_dir.mkdir(exist_ok=True)

            for file in study_path.glob("*.nii.gz"):
                shutil.copy(file, target_dir / file.name)

    print(f"✓ Split completed: {len(train_paths)} train / {len(test_paths)} test")

if __name__ == "__main__":
    root_dir = "/path/to/original_dataset"
    output_dir = "/path/to/output_dataset"

    all_studies = get_all_study_paths(root_dir)
    split_and_copy(all_studies, output_dir)
