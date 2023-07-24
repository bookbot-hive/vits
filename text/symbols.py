""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.
"""
_pad = "_"
_punctuation = ";:,.!? "
_letters_ipa = [
    "a",
    "b",
    "tʃ",
    "d",
    "e",
    "f",
    "ɡ",
    "h",
    "i",
    "dʒ",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "j",
    "z",
    "ŋ",
    "ə",
    "ɲ",
    "ʃ",
    "x",
    "ʔ",
]


# Export all symbols:
symbols = [_pad] + list(_punctuation) + _letters_ipa

# Special symbol ids
SPACE_ID = symbols.index(" ")
