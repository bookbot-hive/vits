import argparse
import text
from utils import load_filepaths_and_text

from functools import partial
from tqdm.contrib.concurrent import process_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_extension", default="cleaned")
    parser.add_argument("--text_index", default=1, type=int)
    parser.add_argument(
        "--filelists",
        nargs="+",
        default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"],
    )
    parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

    args = parser.parse_args()

    for filelist in args.filelists:
        print("START:", filelist)
        filepaths_and_text = load_filepaths_and_text(filelist)
        original_texts = [row[args.text_index] for row in filepaths_and_text]
        cleaner = partial(text._clean_text, cleaner_names=args.text_cleaners)
        cleaned_texts = process_map(cleaner, original_texts)

        for i in range(len(filepaths_and_text)):
            filepaths_and_text[i][args.text_index] = cleaned_texts[i]

        new_filelist = filelist + "." + args.out_extension
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
