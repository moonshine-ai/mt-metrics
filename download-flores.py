#!/usr/bin/env python3
"""
Script to download FLORES-200 and extract parallel sentence pairs.
"""

from datasets import load_dataset
import argparse


def download_and_extract_flores200(lang1, lang2, output_file, split="devtest"):
    """
    Download FLORES-200 and extract parallel sentences for a language pair.
    
    Args:
        lang1: First language code (e.g., 'eng_Latn' for English)
        lang2: Second language code (e.g., 'fra_Latn' for French)
        output_file: Path to output TSV file
        split: Which split to use ('dev' or 'devtest')
    """
    print(f"Downloading FLORES-200 dataset...")
    
    # Load the dataset
    dataset = load_dataset("facebook/flores", f"{lang1}-{lang2}", split=split)
    
    print(f"Extracting {len(dataset)} sentence pairs for {lang1}-{lang2}...")
    
    # Extract and save sentence pairs
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in dataset:
            sentence1 = example['sentence_' + lang1]
            sentence2 = example['sentence_' + lang2]
            # Write as tab-separated values
            f.write(f"{sentence1}\t{sentence2}\n")
    
    print(f"Saved {len(dataset)} sentence pairs to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download FLORES-200 and extract parallel sentences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python flores200_extract.py eng_Latn fra_Latn output.tsv
  python flores200_extract.py spa_Latn deu_Latn output.tsv --split dev

Common language codes:
  English: eng_Latn
  French: fra_Latn
  Spanish: spa_Latn
  German: deu_Latn
  Chinese: zho_Hans (Simplified), zho_Hant (Traditional)
  Arabic: arb_Arab
  Japanese: jpn_Jpan
  Korean: kor_Hang
  Russian: rus_Cyrl
  
For full list, see: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
        """
    )
    
    parser.add_argument('--lang1', help='First language code (e.g., eng_Latn)')
    parser.add_argument('--lang2', help='Second language code (e.g., fra_Latn)')
    parser.add_argument('--output-file', help='Output TSV file path')
    parser.add_argument(
        '--split', 
        default='devtest', 
        choices=['dev', 'devtest'],
        help='Which split to use (default: devtest)'
    )
    
    args = parser.parse_args()
    
    download_and_extract_flores200(
        args.lang1, 
        args.lang2, 
        args.output_file,
        args.split
    )


if __name__ == "__main__":
    main()