# Machine Translation Metrics

An example of downloading the FLORES test datasets for multiple languages, and then evaluating machine translations using the BLEU and COMET metrics that ground truth data.

## Usage

```bash
pip install -r requirements.txt
python score-files.py
```

The FLORES files from the devtest split for English to and from German, Spanish, French, Italian, and Dutch are included in the data folder in this repository, as `en_de.tsv`, etc. If you need to redownload them, or gather other languages, you can use the `download-flores.py` script.

The results of translating using the Moonhine models are held in the `en_de_translated_moonshine.tsv`, etc, files. These are what are evaluated by the `score-files.py` script.

If you need to evaluate other translation models, the format of the FLORES files is that each line contains two sentences, each a human-created translation of the other, separated by a tab character. The `_translated_moonshine.tsv` files should contain these two columns, plus the result of translating from the first to second language, and the second to first, using an ML model.
