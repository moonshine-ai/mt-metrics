import sacrebleu
import argparse
import comet
import torch
import numpy as np

def score_file(input_file, language1, language2, comet_scorer, gpus):
    lang1_to_2_total_bleu = 0
    lang2_to_1_total_bleu = 0
    sentence_count = 0
    lang1_to_2_comet_data = []
    lang2_to_1_comet_data = []
    with open(input_file, "r") as input_file:
        for input_line in input_file.readlines():
            lang1_reference, lang2_reference, lang1_to_2_trans, lang2_to_1_trans = input_line.strip().split("\t")
            lang1_to_2_comet_data.append({
                "src": lang1_reference,
                "mt": lang1_to_2_trans,
                "ref": lang2_reference,
            })
            lang2_to_1_comet_data.append({
                "src": lang2_reference,
                "mt": lang2_to_1_trans,
                "ref": lang1_reference,
            })
            lang1_to_2_bleu = sacrebleu.sentence_bleu(lang1_to_2_trans, [lang2_reference]).score
            lang2_to_1_bleu = sacrebleu.sentence_bleu(lang2_to_1_trans, [lang1_reference]).score
            lang1_to_2_total_bleu += lang1_to_2_bleu
            lang2_to_1_total_bleu += lang2_to_1_bleu
            sentence_count += 1
    print(f"{language1} to {language2} BLEU score: {lang1_to_2_total_bleu / sentence_count:.2f}")
    print(f"{language2} to {language1} BLEU score: {lang2_to_1_total_bleu / sentence_count:.2f}")
    lang1_to_2_comet_results = comet_scorer.predict(lang1_to_2_comet_data, progress_bar=False)
    lang1_to_2_comet_score = np.mean(lang1_to_2_comet_results.scores)
    print(f"{language1} to {language2} COMET score: {lang1_to_2_comet_score:.2f}")
    lang2_to_1_comet_results = comet_scorer.predict(lang2_to_1_comet_data, progress_bar=False)
    lang2_to_1_comet_score = np.mean(lang2_to_1_comet_results.scores)
    print(f"{language2} to {language1} COMET score: {lang2_to_1_comet_score:.2f}")

def main(file_suffix):
    print("Loading COMET scorer...")
    comet_scorer = comet.load_from_checkpoint(comet.download_model("wmt20-comet-da"))
    print("COMET scorer loaded")

    gpus = 1 if torch.cuda.is_available() else 0

    print("Scoring files...")
    languages = ["de", "es", "fr", "it", "nl"]
    language1 = "en"
    for language2 in languages:
        input_file = f"data/{language1}_{language2}_{file_suffix}.tsv"
        score_file(input_file, language1, language2, comet_scorer, gpus)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score files")
    parser.add_argument("--file-suffix", help="Suffix of the file to score", default="translated_moonshine")
    args = parser.parse_args()
    main(args.file_suffix)