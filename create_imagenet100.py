#!/usr/bin/env python3
"""
Create ImageNet-100 subset by symlinking 100 classes from full ImageNet.

ImageNet-100 is a standard benchmark subset using 100 carefully selected classes
from the full ImageNet-1K dataset. This script creates a new directory structure
with symlinks to the original data, avoiding data duplication.

Usage:
    python create_imagenet100.py --source data/imagenet --target data/imagenet100
"""

import argparse
import os
from pathlib import Path

# Standard ImageNet-100 class list (100 WordNet IDs)
# This is the commonly used subset for efficient experimentation
IMAGENET100_WNIDS = [
    'n01498041', 'n01514668', 'n01582220', 'n01592084', 'n01614925',
    'n01616318', 'n01631663', 'n01641577', 'n01669191', 'n01677366',
    'n01687978', 'n01694178', 'n01698640', 'n01735189', 'n01770081',
    'n01770393', 'n01774750', 'n01784675', 'n01819313', 'n01820546',
    'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01882714',
    'n01910747', 'n01917289', 'n01944390', 'n01945685', 'n01950731',
    'n01983481', 'n01984695', 'n02002724', 'n02006656', 'n02007558',
    'n02009912', 'n02009229', 'n02011460', 'n02018207', 'n02018795',
    'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110',
    'n02051845', 'n02056570', 'n02058221', 'n02066245', 'n02071294',
    'n02074367', 'n02077923', 'n02085620', 'n02086240', 'n02088094',
    'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078',
    'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721',
    'n02091032', 'n02091134', 'n02091244', 'n02091467', 'n02091635',
    'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428',
    'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114',
    'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889',
    'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585',
    'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474',
    'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267',
]


def create_imagenet100_subset(source_dir: Path, target_dir: Path, dry_run: bool = False):
    """
    Create ImageNet-100 subset by creating symlinks to source classes.
    
    Args:
        source_dir: Path to full ImageNet directory (containing train/ and val/)
        target_dir: Path where ImageNet-100 will be created
        dry_run: If True, only print what would be done without creating symlinks
    """
    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()
    
    # Verify source structure
    train_src = source_dir / 'train'
    val_src = source_dir / 'val'
    
    if not train_src.exists():
        raise FileNotFoundError(f"Training directory not found: {train_src}")
    if not val_src.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_src}")
    
    # Create target directories
    train_tgt = target_dir / 'train'
    val_tgt = target_dir / 'val'
    
    if not dry_run:
        train_tgt.mkdir(parents=True, exist_ok=True)
        val_tgt.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    found_classes = 0
    missing_classes = []
    train_images = 0
    val_images = 0
    
    print(f"Creating ImageNet-100 subset...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Classes: {len(IMAGENET100_WNIDS)}")
    print()
    
    for wnid in IMAGENET100_WNIDS:
        train_class_src = train_src / wnid
        val_class_src = val_src / wnid
        
        # Check if class exists in source
        if not train_class_src.exists():
            missing_classes.append(wnid)
            print(f"  WARNING: Class {wnid} not found in training set")
            continue
            
        found_classes += 1
        
        # Count images
        train_count = len(list(train_class_src.glob('*')))
        val_count = len(list(val_class_src.glob('*'))) if val_class_src.exists() else 0
        train_images += train_count
        val_images += val_count
        
        # Create symlinks
        train_class_tgt = train_tgt / wnid
        val_class_tgt = val_tgt / wnid
        
        if dry_run:
            print(f"  Would link: {wnid} (train: {train_count}, val: {val_count})")
        else:
            # Remove existing symlink if present
            if train_class_tgt.is_symlink():
                train_class_tgt.unlink()
            if val_class_tgt.is_symlink():
                val_class_tgt.unlink()
                
            # Create symlinks
            train_class_tgt.symlink_to(train_class_src)
            if val_class_src.exists():
                val_class_tgt.symlink_to(val_class_src)
    
    # Print summary
    print()
    print("=" * 50)
    print("Summary:")
    print(f"  Classes found: {found_classes}/{len(IMAGENET100_WNIDS)}")
    print(f"  Training images: {train_images:,}")
    print(f"  Validation images: {val_images:,}")
    
    if missing_classes:
        print(f"\n  Missing classes ({len(missing_classes)}):")
        for wnid in missing_classes:
            print(f"    - {wnid}")
    
    if not dry_run:
        print(f"\nImageNet-100 created at: {target_dir}")
        print("\nTo use with training:")
        print(f"  python train.py fit --config config_imagenet100.yaml")


def main():
    parser = argparse.ArgumentParser(
        description='Create ImageNet-100 subset from full ImageNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create subset
  python create_imagenet100.py --source data/imagenet --target data/imagenet100
  
  # Preview what would be created (dry run)
  python create_imagenet100.py --source data/imagenet --target data/imagenet100 --dry-run
        """
    )
    parser.add_argument(
        '--source', '-s',
        type=Path,
        default=Path('data/imagenet'),
        help='Path to full ImageNet directory (default: data/imagenet)'
    )
    parser.add_argument(
        '--target', '-t', 
        type=Path,
        default=Path('data/imagenet100'),
        help='Path for ImageNet-100 output (default: data/imagenet100)'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Print what would be done without creating symlinks'
    )
    
    args = parser.parse_args()
    create_imagenet100_subset(args.source, args.target, args.dry_run)


if __name__ == '__main__':
    main()
