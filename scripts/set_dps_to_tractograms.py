import numpy as np
import nibabel as nib
from dipy.io.streamline import load_tractogram, save_tractogram
import os
import argparse
import glob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Set DPS to tractograms')
    parser.add_argument('tractograms', type=str,
                        help='Glob pattern to the tractograms to set the DPS.', nargs='+')
    parser.add_argument('score', type=int,
                        help='Score to assign to the tractograms score DPS.')
    parser.add_argument('--reference', type=str, default=None,
                        help='Reference file to load the tractograms when needed.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path to save the tractograms.')
    return parser.parse_args()


def set_score_to_tractograms(
        tractograms: list,
        score: int,
        reference: str = None,
        output: str = None,
        bbox_valid_check: bool = True):
    for tractogram in tractograms:
        print(f"Setting DPS to {score} in {tractogram}")
        sft = load_tractogram(tractogram, reference,
                              bbox_valid_check=False)

        sft.data_per_streamline['score'] = np.full(
            len(sft.streamlines), score)

        save_path = output if output else tractogram
        save_tractogram(sft, save_path, bbox_valid_check=bbox_valid_check)


def main():
    args = parse_args()

    assert args.score in [-1, 0, 1], 'Score must be -1, 0 or 1'
    # tractograms = glob.glob(args.tractograms)
    set_score_to_tractograms(args.tractograms, args.score,
                             args.reference, args.output)


if __name__ == '__main__':
    main()
