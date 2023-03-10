import re
from typing import Union

from data.text.symbols import all_phonemes
from data.text.tokenizer import Phonemizer, Tokenizer


class TextToTokens:
    def __init__(self, phonemizer: Phonemizer, tokenizer: Tokenizer):
        self.phonemizer = phonemizer
        self.tokenizer = tokenizer
    
    def __call__(self, input_text: Union[str, list]) -> list:
        phons = self.phonemizer(input_text)

        # fix phones
        # phons = re.sub("MAH1NDIY0", "MAH1NDEY2", phons)          # Monday
        # phons = re.sub("TUW1ZDIY0", "TUW1ZDEY2", phons)          # Tuesday
        # phons = re.sub("WEH1NZDIY0", "WEH1NZDEY2", phons)        # Wendesday
        # phons = re.sub("FRAY1DIY0", "FRAY1DEY2", phons)          # Friday
        # phons = re.sub("SAE1TER0DIY0", "SAE1TER0DEY2", phons)    # Saturday
        # phons = re.sub("SAE1TER0DIY0", "SAE1TER0DEY2", phons)    # Sunday
        # phons = re.sub("BAA1RAH0K", "BAE0RAA1K", phons)          # Barack
        # phons = re.sub("FIH1DZ", "PIY1EY2CHDIY1", phons)          # PhD

        print("\033[92m" + "**** Synthesizing phone sequence ****\n" + "\033[0m", phons, "\n")
        tokens = self.tokenizer(phons)
        return tokens
    
    @classmethod
    def default(cls, language: str, add_start_end: bool, with_stress: bool, model_breathing: bool, njobs=1):
        phonemizer = Phonemizer(language=language, njobs=njobs, with_stress=with_stress)
        tokenizer = Tokenizer(add_start_end=add_start_end, model_breathing=model_breathing)
        return cls(phonemizer=phonemizer, tokenizer=tokenizer)
