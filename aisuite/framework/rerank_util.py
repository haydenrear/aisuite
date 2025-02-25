import logging
from typing import List, Optional, Union

import rerankers.results
from google.cloud.discoveryengine_v1 import RankingRecord

try:
    from python_util.logger.logger import LoggerFacade
    log_info = lambda x : LoggerFacade.info(x)
    log_warn = lambda x: LoggerFacade.warn(x)
except:
    log_info = lambda x: logging.info(f"INFO: {x}")
    log_warn = lambda x: logging.warning(f"WARN: {x}")


def create_ranking_records(
        docs: Union[str, List[str], rerankers.results.Document, List[rerankers.results.Document]],
        doc_ids: Optional[Union[List[str], List[int]]],
        metadata
) -> list[RankingRecord]:
    if isinstance(docs, str) or isinstance(docs, rerankers.results.Document):
        record = parse_single_ranking_record(docs, metadata, doc_ids)
        if record:
            return [record]
        else:
            _log_no_ranking_records()
            return []
    elif isinstance(docs, List):
        ranking_records = [parse_to_ranking_record(d, metadata, doc_ids, i, len(docs))
                           for i, d in  enumerate(docs)]
        ranking_records = [record for record in filter(lambda x: x is not None, ranking_records)]
        if len(ranking_records) == 0:
            _log_no_ranking_records()
        return ranking_records
    else:
        _log_no_ranking_records()
        return []


def _log_no_ranking_records():
    log_warn("Did not find any valid ranking records")


def create_ranking_record(doc_id: str, text: str, metadata: dict[str, ...]):
    title = key_from_metadata_or_none(metadata, "title")
    if not text and not title:
        log_warn("Rerank must be provided with text or title.")
        return None

    return RankingRecord(
        id=doc_id,
        content=text,
        title=title,
        score=key_from_metadata_or_none(metadata, "score"))


def key_from_metadata_or_none(metadata, key: str):
    if metadata:
        return metadata[key] if key in metadata.keys() else None

    return None


def parse_single_ranking_record(
        d: Union[str, rerankers.results.Document],
        metadata: dict[str, ...],
        doc_ids: Optional[Union[List[str], List[int]]]
) -> RankingRecord:
    if len(doc_ids) >= 1:
        return create_ranking_record(str(doc_ids[0]), get_doc_text(d), metadata)
    else:
        return create_ranking_record("1", get_doc_text(d), metadata)


def get_doc_text(d: Union[str, rerankers.results.Document]) -> str:
    if isinstance(d, str):
        return d
    elif isinstance(d, rerankers.results.Document):
        return d.text
    else:
        log_warn(f"Get doc called with unkonwn type: {d.__class__.__name__}.")
        return ""


def parse_to_ranking_record(
        d: Union[str, rerankers.results.Document],
        metadata: dict[str, ...],
        doc_ids: Optional[Union[List[str], List[int]]],
        i: int,
        docs_len: int
) -> RankingRecord:
    if isinstance(d, rerankers.results.Document):
        return create_ranking_record(get_doc_id(doc_ids, docs_len, i), d.text, metadata)
    else:
        return create_ranking_record(get_doc_id(doc_ids, docs_len, i), d, metadata)


def get_doc_id(doc_ids, docs_len, i):
    if doc_ids and len(doc_ids) >= docs_len and doc_ids[i]:
        return str(doc_ids[i])
    else:
        return str(i)
