import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests
from openai import OpenAI


FILE_PARSE_API = os.getenv("MINERU_FILE_PARSE_API", "http://58.19.38.185:8111/file_parse")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "XX")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

client = None
if DEEPSEEK_API_KEY:
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


def clean_mineru_md(
    md_text: str,
    remove_tables: bool = True,
    remove_images: bool = True,
) -> str:
    if not md_text:
        return ""

    if remove_tables:
        md_text = re.sub(
            r"<table.*?>.*?</table>",
            "",
            md_text,
            flags=re.DOTALL | re.IGNORECASE,
        )

    if remove_images:
        md_text = re.sub(
            r"!\[.*?\]\(.*?\.(jpg|jpeg|png)\)",
            "",
            md_text,
            flags=re.IGNORECASE,
        )

    def normalize_header(match):
        header = match.group(1)
        header = re.sub(r"^\d+(\.\d+)*\s*", "", header)
        header = header.strip()
        return f"\n[{header}]\n"

    md_text = re.sub(
        r"^\s*#{1,6}\s*(.+)$",
        normalize_header,
        md_text,
        flags=re.MULTILINE,
    )

    md_text = re.sub(r"\n{3,}", "\n\n", md_text)
    return md_text.strip()


def parse_pdf_to_md(pdf_path: Path) -> str:
    with open(pdf_path, "rb") as f:
        files = {"files": (pdf_path.name, f, "application/pdf")}
        data = {
            "return_md": "true",
            "return_middle_json": "false",
            "return_model_output": "false",
            "return_images": "false",
            "parse_method": "auto",
            "backend": "pipeline",
            "formula_enable": "false",
            "table_enable": "false",
            "return_content_list": "false",
        }

        resp = requests.post(FILE_PARSE_API, files=files, data=data, timeout=300)
        resp.raise_for_status()
        result = resp.json()

    doc_id = pdf_path.stem
    results = result.get("results") or {}
    if doc_id not in results:
        raise ValueError(f"MinerU 返回结果中缺少 doc_id={doc_id}")

    raw = results[doc_id].get("md_content")
    if raw is None:
        raise ValueError(f"MinerU 返回结果中缺少 md_content: {doc_id}")

    if isinstance(raw, str):
        return raw

    if isinstance(raw, list):
        return "\n\n".join(
            block.get("text", "")
            for block in raw
            if isinstance(block, dict) and block.get("text")
        )

    raise ValueError("MinerU 返回的 md_content 格式不支持")


def _normalize_title(s: str, max_len: int = 300) -> str:
    s = (s or "").strip()
    s = s.strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[。\.]+$", "", s).strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s


def extract_title_by_rules(md_text: str, scan_lines: int = 120) -> Optional[str]:
    if not md_text:
        return None

    raw_lines = md_text.splitlines()[:scan_lines]

    start_idx = 0
    for i, ln in enumerate(raw_lines):
        if re.match(r"(?i)^\s*accepted manuscript\b", (ln or "").strip()):
            start_idx = i + 1
            break

    lines = [ln.strip() for ln in raw_lines[start_idx:] if ln.strip()]
    if len(lines) < 2:
        return None

    for ln in lines[:50]:
        m = re.match(r"(?i)^title\s*[:：]\s*(.+)$", ln)
        if m:
            t = _normalize_title(m.group(1))
            return t if t else None

    BAD_HEADS = {
        "review", "review article",
        "article", "accepted article", "accepted manuscript",
        "full research paper", "research article", "communication", "letter",
        "highlights", "summary", "abstract", "graphical abstract", "keywords",
    }

    def norm_head(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"^\W+|\W+$", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def is_bad_head(s: str) -> bool:
        return norm_head(s) in BAD_HEADS

    def looks_like_author_line(s: str) -> bool:
        if not s:
            return False
        if re.search(r"(?i)\babstract\b", s):
            return False
        if re.search(r"(?i)\bdoi\s*:\b", s):
            return False
        comma = s.count(",")
        if comma >= 1 and re.search(r"(?i)\b(and|und)\b", s):
            return True
        if comma >= 2:
            return True
        if ("*" in s or re.search(r"\d", s)) and comma >= 1:
            return True
        return False

    def looks_like_meta_line(s: str) -> bool:
        if re.search(r"(?i)\bdoi\s*:\b", s):
            return True
        if re.search(r"(?i)\bp(ii|ii:)\b", s):
            return True
        if re.search(r"(?i)\breceived\b|\baccepted\b", s):
            return True
        return False

    for i in range(min(len(lines) - 1, 80)):
        ln = lines[i]
        if not ln.startswith("#"):
            continue

        cand = re.sub(r"^\s*#{1,6}\s*", "", ln).strip()
        cand = re.sub(r"^\d+(\.\d+)*\s*", "", cand).strip()

        if not cand:
            continue
        if is_bad_head(cand):
            continue
        if len(cand) < 15 or len(cand) > 220:
            continue

        nxt = lines[i + 1]
        if looks_like_meta_line(nxt):
            continue
        if looks_like_author_line(nxt):
            return _normalize_title(cand)

    for i in range(min(len(lines) - 2, 80)):
        cur = lines[i]
        nxt = lines[i + 1]
        nxt2 = lines[i + 2]

        cur_text = cur.lstrip("#").strip() if cur.startswith("#") else cur.strip()
        if not is_bad_head(cur_text):
            continue

        if looks_like_meta_line(nxt) or nxt.startswith("#"):
            continue
        if len(nxt) < 15 or len(nxt) > 220:
            continue
        if looks_like_author_line(nxt2):
            return _normalize_title(nxt)

    return None


def make_title_snippet(md_text: str, max_chars: int = 4000) -> str:
    abstract_re = re.compile(
        r"(?im)^\s*(?:#{1,6}\s*)?abstract\b\s*[:：]?\s*$|^\s*abstract\b\s*[:：]"
    )
    s = (md_text or "").strip()

    m = abstract_re.search(s)
    if m:
        s = s[:m.start()].strip()

    if len(s) > max_chars:
        s = s[:max_chars].strip()

    return s


def get_title_from_deepseek(
    md_text: str,
    *,
    model: str = DEEPSEEK_MODEL,
    max_chars: int = 4000,
    retries: int = 2,
) -> str:
    if client is None:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")

    snippet = make_title_snippet(md_text, max_chars=max_chars)
    last_err = None

    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract the paper/article title from provided text.\n"
                            "Return ONLY the title as plain text, one line.\n"
                            "No quotes, no markdown, no extra words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Extract the article title from the following content. "
                            "If multiple candidates exist, pick the most likely main title.\n\n"
                            f"{snippet}"
                        ),
                    },
                ],
                temperature=0.0,
            )

            title = _normalize_title(resp.choices[0].message.content)
            if not title:
                raise ValueError("DeepSeek 返回空标题")
            return title

        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            break

    raise RuntimeError(f"DeepSeek 标题提取失败: {last_err}")


def get_title(md_text: str) -> str:
    title = extract_title_by_rules(md_text)
    if title:
        return title

    try:
        return get_title_from_deepseek(md_text)
    except Exception:
        return "Untitled"


def parse_pdf_to_jsonl(
    pdf_path: Path,
    *,
    remove_tables: bool = True,
    remove_images: bool = True,
) -> str:
    doc_id = pdf_path.stem

    md_text = parse_pdf_to_md(pdf_path)
    md_text = clean_mineru_md(
        md_text,
        remove_tables=remove_tables,
        remove_images=remove_images,
    )

    if not md_text.strip():
        raise ValueError("解析得到的 markdown 为空")

    title = get_title(md_text)

    record = {
        "id": doc_id,
        "title": title,
        "text": f"[TITLE] {title}\n\n{md_text}",
        "metadata": {
            "doc_id": doc_id,
            "title": title,
            "source": pdf_path.name,
        },
    }

    return json.dumps(record, ensure_ascii=False)