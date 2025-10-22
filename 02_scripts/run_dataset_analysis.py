"""
Dataset Analysis and Class Weight Generation
Standalone script for analyzing glaucoma dataset
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

class DatasetAnalyzer:
    """Analyze dataset structure and generate class weights"""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.splits = ['train', 'validation', 'test']
        self.classes = ['NRG', 'RG']
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
        self.dataset_info = self._scan_dataset()

    def _scan_dataset(self) -> Dict:
        """Scan dataset directory and count images"""
        info = {
            'splits': {},
            'total_images': 0,
            'class_distribution': {cls: 0 for cls in self.classes}
        }

        for split in self.splits:
            info['splits'][split] = {}
            eyepac_path = self.base_dir / split / 'eyepac'

            if eyepac_path.exists():
                for cls in self.classes:
                    cls_path = eyepac_path / cls
                    if cls_path.exists():
                        images = self._get_image_files(cls_path)
                        count = len(images)
                        info['splits'][split][cls] = count
                        info['class_distribution'][cls] += count
                        info['total_images'] += count

        return info

    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files in directory"""
        images = []
        for ext in self.image_extensions:
            images.extend(list(directory.glob(f'*{ext}')))
            images.extend(list(directory.glob(f'*{ext.upper()}')))
        return images

    def calculate_class_weights(self) -> Dict[str, float]:
        """Calculate class weights for weighted loss"""
        total = self.dataset_info['total_images']
        n_classes = len(self.classes)
        weights = {}
        for cls in self.classes:
            count = self.dataset_info['class_distribution'][cls]
            if count > 0:
                weights[cls] = total / (n_classes * count)
            else:
                weights[cls] = 1.0
        return weights

    def calculate_imbalance_ratio(self) -> float:
        """Calculate imbalance ratio (majority/minority)"""
        counts = list(self.dataset_info['class_distribution'].values())
        if min(counts) == 0:
            return float('inf')
        return max(counts) / min(counts)

    def get_minority_majority_classes(self) -> Tuple[str, str]:
        """Identify minority and majority classes"""
        dist = self.dataset_info['class_distribution']
        minority = min(dist, key=dist.get)
        majority = max(dist, key=dist.get)
        return minority, majority

    def print_summary(self):
        """Print comprehensive dataset summary"""
        print("="*80)
        print("DATASET ANALYSIS - DATA PREPARATION")
        print("="*80)
        print(f"\nBase Directory: {self.base_dir}")
        print(f"Total Images: {self.dataset_info['total_images']:,}")

        print("\n" + "-"*80)
        print("CLASS DISTRIBUTION")
        print("-"*80)

        for cls, count in self.dataset_info['class_distribution'].items():
            percentage = (count / self.dataset_info['total_images'] * 100) if self.dataset_info['total_images'] > 0 else 0
            print(f"  {cls:15s}: {count:6,} images ({percentage:5.2f}%)")

        minority, majority = self.get_minority_majority_classes()
        imbalance = self.calculate_imbalance_ratio()

        print(f"\n  Minority Class: {minority}")
        print(f"  Majority Class: {majority}")
        print(f"  Imbalance Ratio: {imbalance:.2f}:1")

        print("\n" + "-"*80)
        print("SPLIT DISTRIBUTION")
        print("-"*80)

        for split in self.splits:
            if split in self.dataset_info['splits']:
                split_data = self.dataset_info['splits'][split]
                total_split = sum(split_data.values())
                print(f"\n{split.upper()}:")
                for cls, count in split_data.items():
                    percentage = (count / total_split * 100) if total_split > 0 else 0
                    print(f"  {cls:15s}: {count:6,} images ({percentage:5.2f}%)")
                print(f"  {'TOTAL':15s}: {total_split:6,} images")

        print("\n" + "-"*80)
        print("RECOMMENDED CLASS WEIGHTS (for weighted loss)")
        print("-"*80)

        weights = self.calculate_class_weights()
        for cls, weight in weights.items():
            print(f"  {cls:15s}: {weight:.4f}")

        print("\n" + "="*80)

    def export_statistics(self, output_path: str) -> Dict:
        """Export detailed statistics to JSON"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'base_directory': str(self.base_dir),
            'total_images': self.dataset_info['total_images'],
            'class_distribution': self.dataset_info['class_distribution'],
            'splits': self.dataset_info['splits'],
            'imbalance_ratio': self.calculate_imbalance_ratio(),
            'minority_class': self.get_minority_majority_classes()[0],
            'majority_class': self.get_minority_majority_classes()[1],
            'class_weights': self.calculate_class_weights()
        }

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def generate_class_weights_config(self, output_path: str):
        """Generate class weights configuration for training"""
        weights = self.calculate_class_weights()
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.classes))}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        weight_array = [weights[idx_to_class[i]] for i in range(len(self.classes))]

        config = {
            'strategy': 'class_weights',
            'description': 'Use these weights in your loss function (CRITICAL for all 4 models)',
            'class_to_index': class_to_idx,
            'class_weights_dict': weights,
            'class_weights_array': weight_array,
            'usage_pytorch': {
                'code': 'criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights_array).to(device))',
                'note': 'Apply weights to loss function during training for ALL 4 models'
            },
            'usage_tensorflow': {
                'code': 'model.fit(x_train, y_train, class_weight=class_weights_dict, ...)',
                'note': 'Pass class_weight parameter to fit() method'
            },
            'critical_reminders': [
                'USE IN ALL 4 MODELS: U-Net, ResNet-50, U-Net+Canny, ResNet-50+Canny',
                'Apply during training phase only (not validation/test)',
                'Optimizes for sensitivity on minority class (RG - Referable Glaucoma)',
                'Essential for medical imaging with class imbalance',
                'Fair comparison requires same weights across all models'
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n[OK] Class weights configuration saved to: {output_path}")
        print("\nCRITICAL REMINDERS:")
        print("  [OK] Use these weights in ALL 4 model training loops")
        print("  [OK] Apply to loss function: nn.CrossEntropyLoss(weight=...)")
        print("  [OK] Ensures fair comparison across all architectures")
        print("  [OK] Optimizes for detecting glaucoma (minority class)")


def main():
    """Main execution function"""
    # Get current directory
    base_dir = os.getcwd()

    # Initialize analyzer
    print("Initializing dataset analyzer...")
    analyzer = DatasetAnalyzer(base_dir)

    # Print summary
    analyzer.print_summary()

    # Export statistics
    print("\nExporting statistics...")
    stats = analyzer.export_statistics('dataset_statistics.json')
    print(f"[OK] Statistics exported to: dataset_statistics.json")

    # Generate class weights config
    print("\nGenerating class weights configuration...")
    analyzer.generate_class_weights_config('class_weights_config.json')

    # Final recommendations
    print("\n" + "="*80)
    print("RECOMMENDATION: Use CLASS WEIGHTS strategy (MANDATORY)")
    print("="*80)
    print("\nWhy class weights are CRITICAL:")
    print("  • Preserves all valuable medical data")
    print("  • No artificial duplication or data loss")
    print("  • Standard practice in medical AI")
    print("  • Prevents bias toward majority class")
    print("  • Optimizes for high sensitivity (detecting glaucoma)")
    print("\nNext Steps:")
    print("  1. Load class_weights_config.json in your training scripts")
    print("  2. Apply weights to loss function in ALL 4 models")
    print("  3. Proceed to enhanced preprocessing pipeline")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
