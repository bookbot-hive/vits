""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.
"""
_pad = "_"
_punctuation = ";:,.!? "
_letters_ipa = [
    "f",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "s",
    "t",
    "t͡ʃ",
    "u",
    "v",
    "w",
    "x",
    "z",
    "ð",
    "ɑ",
    "ɓ",
    "ɔ",
    "ɗ",
    "ɛ",
    "ɠ",
    "ɣ",
    "ɾ",
    "ʃ",
    "ʄ",
    "θ",
    "ᵐɓ",
    "ᵑg",
    "ᶬv",
    "ⁿz",
    "ⁿɗ",
    "ⁿɗ͡ʒ",
]


# Export all symbols:
symbols = [_pad] + list(_punctuation) + _letters_ipa

# Special symbol ids
SPACE_ID = symbols.index(" ")
