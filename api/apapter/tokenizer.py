from typing import List

from loguru import logger
from transformers import LlamaTokenizer


class CodeLlamaTokenizer(LlamaTokenizer):
    """ https://github.com/facebookresearch/codellama/blob/main/llama/tokenizer.py """
    def __init__(self, *args, **kwargs):
        super(CodeLlamaTokenizer, self).__init__(*args, **kwargs)

        self.prefix_token_id = self.sp_model.piece_to_id("▁<PRE>") or None
        self.middle_token_id = self.sp_model.piece_to_id("▁<MID>") or None
        self.suffix_token_id = self.sp_model.piece_to_id("▁<SUF>") or None
        self.eot_token_id = self.sp_model.piece_to_id("▁<EOT>") or None

        logger.info(
            f"- PRE ID: {self.prefix_token_id} - MID ID: {self.middle_token_id} - SUF ID: {self.suffix_token_id} - EOT ID: {self.eot_token_id}"
        )

    def encode_infilling(self, s: str) -> List[int]:
        """Encode a string without an implicit leading space."""
        return self.sp_model.encode("☺" + s)[2:]

    def decode_infilling(self, t: List[int]) -> str:
        """Decode a string without an implicit leading space."""
        return self.sp_model.decode([self.sp_model.piece_to_id("☺")] + t)[1:]

    def infilling_prompt_tokens(self, pre: str, suf: str, suffix_first: bool = False) -> List[int]:
        """
        Format and encode an infilling problem.
        If `suffix_first` is set, format in suffix-prefix-middle format.
        """
        if suffix_first:
            # format as "<PRE> <SUF>{suf} <MID> {pre}"
            return (
                    [self.bos_token_id, self.prefix_token_id, self.suffix_token_id]
                    + self.encode_infilling(suf)
                    + [self.middle_token_id]
                    + self.encode(pre)[1:]
            )
        else:
            # format as "<PRE> {pre} <SUF>{suf} <MID>"
            return (
                    [self.bos_token_id, self.prefix_token_id]
                    + self.encode(pre)[1:]
                    + [self.suffix_token_id]
                    + self.encode_infilling(suf)
                    + [self.middle_token_id]
            )
