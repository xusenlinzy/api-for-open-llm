""" from https://github.com/netease-youdao/BCEmbedding """
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import (
    List,
    Dict,
    Tuple,
    Any,
    Union,
    Optional,
    Sequence,
)

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from api.utils.protocol import (
    DocumentObj,
    Document,
    RerankResponse,
)


class BaseReranker(ABC):
    @abstractmethod
    @torch.inference_mode()
    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = 256,
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = False,
    ) -> Dict[str, Any]:
        ...


class RAGReranker(ABC):
    def __init__(
        self,
        model_name_or_path: str,
        use_fp16: bool = False,
        device: str = None,
        **kwargs,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
        logger.info(f"Loading from `{model_name_or_path}`.")

        num_gpus = torch.cuda.device_count()
        if device is None:
            self.device = "cuda" if num_gpus > 0 else "cpu"
        else:
            self.device = "cuda:{}".format(int(device)) if device.isdigit() else device

        if self.device == "cpu":
            self.num_gpus = 0
        elif self.device.startswith("cuda:") and num_gpus > 0:
            self.num_gpus = 1
        elif self.device == "cuda":
            self.num_gpus = num_gpus
        else:
            raise ValueError("Please input valid device: 'cpu', 'cuda', 'cuda:0', '0' !")

        if use_fp16:
            self.model.half()

        self.model.eval()
        self.model = self.model.to(self.device)

        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

        logger.info(f"Execute device: {self.device};\t gpu num: {self.num_gpus};\t use fp16: {use_fp16}")

        # for advanced preproc of tokenization
        self.max_length = kwargs.get("max_length", 512)
        self.overlap_tokens = kwargs.get("overlap_tokens", 80)

    @torch.inference_mode()
    def compute_score(
        self,
        text_pairs: Union[Sequence[Tuple[str, str]], Tuple[str, str]],
        batch_size: Optional[int] = 256,
        max_length: Optional[int] = 512,
        enable_tqdm: Optional[bool] = True,
    ) -> Union[List[List[float]], List[float]]:
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus

        assert isinstance(text_pairs, list)
        if isinstance(text_pairs[0], str):
            text_pairs = [text_pairs]

        scores_collection: List[List[float]] = []
        for sentence_id in tqdm(
            range(0, len(text_pairs), batch_size), desc="Calculate scores", disable=not enable_tqdm
        ):
            sentence_pairs_batch = text_pairs[sentence_id:sentence_id + batch_size]
            inputs = self.tokenizer(
                sentence_pairs_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
            scores = self.model(**inputs_on_device, return_dict=True).logits.view(-1, ).float()
            scores = torch.sigmoid(scores)
            scores_collection.extend(scores.cpu().numpy().tolist())

        if len(scores_collection) == 1:
            return scores_collection[0]

        return scores_collection

    @torch.inference_mode()
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        batch_size: Optional[int] = 256,
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = False,
    ) -> Optional[RerankResponse]:
        # remove invalid passages
        passages = [p[:128000] for p in documents if isinstance(p, str) and 0 < len(p)]
        if query is None or len(query) == 0 or len(passages) == 0:
            return None

        # preproc of tokenization
        sentence_pairs, sentence_pairs_pids = self._tokenize_preproc(
            query,
            passages,
            max_length=self.max_length,
            overlap_tokens=self.overlap_tokens,
        )

        # batch inference
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus

        tot_scores = []
        for k in range(0, len(sentence_pairs), batch_size):
            batch = self.tokenizer.pad(
                sentence_pairs[k: k + batch_size],
                padding=True,
                max_length=None,
                pad_to_multiple_of=None,
                return_tensors="pt"
            )
            batch_on_device = {k: v.to(self.device) for k, v in batch.items()}
            scores = self.model(**batch_on_device, return_dict=True).logits.view(-1, ).float()
            scores = torch.sigmoid(scores)
            tot_scores.extend(scores.cpu().numpy().tolist())

        # ranking
        merge_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(sentence_pairs_pids, tot_scores):
            merge_scores[pid] = max(merge_scores[pid], score)

        merge_scores_argsort = np.argsort(merge_scores)[::-1].tolist()
        sorted_scores = []
        for mid in merge_scores_argsort:
            sorted_scores.append(merge_scores[mid])

        if top_n is not None:
            sorted_scores = sorted_scores[:top_n]
            merge_scores_argsort = merge_scores_argsort[:top_n]

        if return_documents:
            docs = [
                DocumentObj(
                    index=int(_id),
                    relevance_score=float(score),
                    document=Document(text=documents[int(_id)]),
                )
                for _id, score in zip(merge_scores_argsort, sorted_scores)
            ]
        else:
            docs = [
                DocumentObj(
                    index=int(_id),
                    relevance_score=float(score),
                    document=None,
                )
                for _id, score in zip(merge_scores_argsort, sorted_scores)
            ]
        return RerankResponse(results=docs)

    def _tokenize_preproc(
        self,
        query: str,
        passages: Sequence[str],
        max_length: Optional[int] = 512,
        overlap_tokens: Optional[int] = 80,
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        assert self.tokenizer is not None, "Please provide a valid tokenizer for tokenization!"
        sep_id = self.tokenizer.sep_token_id

        def _merge_inputs(chunk1_raw, chunk2):
            chunk1 = deepcopy(chunk1_raw)

            chunk1["input_ids"].append(sep_id)
            chunk1["input_ids"].extend(chunk2["input_ids"])
            chunk1["input_ids"].append(sep_id)

            chunk1["attention_mask"].append(chunk2["attention_mask"][0])
            chunk1["attention_mask"].extend(chunk2["attention_mask"])
            chunk1["attention_mask"].append(chunk2["attention_mask"][0])

            if "token_type_ids" in chunk1:
                token_type_ids = [1 for _ in range(len(chunk2["token_type_ids"]) + 2)]
                chunk1["token_type_ids"].extend(token_type_ids)
            return chunk1

        query_inputs = self.tokenizer.encode_plus(query, truncation=False, padding=False)
        max_passage_inputs_length = max_length - len(query_inputs["input_ids"]) - 2
        assert max_passage_inputs_length > 100, "Your query is too long! Please make sure your query less than 400 tokens!"
        overlap_tokens_implt = min(overlap_tokens, max_passage_inputs_length // 4)

        res_merge_inputs = []
        res_merge_inputs_pids = []
        for pid, passage in enumerate(passages):
            passage_inputs = self.tokenizer.encode_plus(passage, truncation=False, padding=False, add_special_tokens=False)
            passage_inputs_length = len(passage_inputs["input_ids"])

            if passage_inputs_length <= max_passage_inputs_length:
                qp_merge_inputs = _merge_inputs(query_inputs, passage_inputs)
                res_merge_inputs.append(qp_merge_inputs)
                res_merge_inputs_pids.append(pid)
            else:
                start_id = 0
                while start_id < passage_inputs_length:
                    end_id = start_id + max_passage_inputs_length
                    sub_passage_inputs = {k: v[start_id: end_id] for k, v in passage_inputs.items()}
                    start_id = end_id - overlap_tokens_implt if end_id < passage_inputs_length else end_id

                    qp_merge_inputs = _merge_inputs(query_inputs, sub_passage_inputs)
                    res_merge_inputs.append(qp_merge_inputs)
                    res_merge_inputs_pids.append(pid)

        return res_merge_inputs, res_merge_inputs_pids
