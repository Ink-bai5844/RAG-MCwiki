from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen


USER_AGENT = "mcwiki-text-crawler/0.1 (+https://zh.minecraft.wiki/)"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_INDEX_DIR = "index"
SPECIAL_ALL_PAGES_URL = "https://zh.minecraft.wiki/w/Special:AllPages"
API_CANDIDATES = [
    "https://zh.minecraft.wiki/w/api.php",
    "https://zh.minecraft.wiki/api.php",
]


@dataclass
class PageResult:
    title: str
    source_url: str
    text: str


class MediaWikiTextExtractor(HTMLParser):
    """Extract readable text from the HTML fragment returned by MediaWiki."""

    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "caption",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
    DROP_TAGS = {"noscript", "script", "style"}
    DROP_CLASSES = {
        "catlinks",
        "mw-cite-backlink",
        "mw-editsection",
        "navbox",
        "noprint",
        "reference",
        "reflist",
        "toc",
        "vertical-navbox",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._drop_stack: list[bool] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class") or "").split())
        should_drop = (
            tag in self.DROP_TAGS
            or bool(classes & self.DROP_CLASSES)
            or attrs_dict.get("aria-hidden") == "true"
        )
        self._drop_stack.append(should_drop)

        if self._is_dropping():
            return

        if tag in self.BLOCK_TAGS:
            self._append_newline()

    def handle_endtag(self, tag: str) -> None:
        dropping_before_pop = self._is_dropping()

        if not dropping_before_pop and tag in self.BLOCK_TAGS:
            self._append_newline()

        if self._drop_stack:
            self._drop_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._is_dropping():
            return

        cleaned = re.sub(r"\s+", " ", data)
        if cleaned.strip():
            self._parts.append(cleaned)

    def get_text(self) -> str:
        text = "".join(self._parts)
        lines = []
        previous = None
        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            if line != previous:
                lines.append(line)
            previous = line
        return "\n".join(lines)

    def _append_newline(self) -> None:
        if not self._parts or self._parts[-1].endswith("\n"):
            return
        self._parts.append("\n")

    def _is_dropping(self) -> bool:
        return any(self._drop_stack)


class RequestPacer:
    """Global pacing controller shared by all workers."""

    def __init__(self, interval_seconds: float, jitter_seconds: float) -> None:
        self.interval_seconds = max(interval_seconds, 0.0)
        self.jitter_seconds = max(jitter_seconds, 0.0)
        self._next_allowed_time = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.interval_seconds > 0:
            with self._lock:
                now = time.monotonic()
                slot = max(now, self._next_allowed_time)
                self._next_allowed_time = slot + self.interval_seconds
            wait_seconds = max(slot - time.monotonic(), 0.0)
            if wait_seconds > 0:
                time.sleep(wait_seconds)

        if self.jitter_seconds > 0:
            time.sleep(random.uniform(0.0, self.jitter_seconds))


class SpecialAllPagesParser(HTMLParser):
    """Parse title links and next-page link from Special:AllPages HTML."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.titles: list[str] = []
        self.next_href: str | None = None

        self._body_stack: list[bool] = []
        self._seen: set[str] = set()

        self._current_link_href: str | None = None
        self._current_link_text_parts: list[str] = []
        self._best_fallback_next: tuple[str, str] | None = None

        current_query = parse_qs(urlparse(base_url).query)
        self._current_from = current_query.get("from", [None])[0]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class") or "").split())

        if tag == "div":
            self._body_stack.append("mw-allpages-body" in classes or "mw-allpages-chunk" in classes)

        if tag != "a":
            return

        href = attrs_dict.get("href")
        if not href:
            return

        if self._is_in_body():
            title = href_to_title(href, self.base_url)
            if title and title not in self._seen:
                self._seen.add(title)
                self.titles.append(title)

        self._current_link_href = href
        self._current_link_text_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_link_href is not None:
            text = "".join(self._current_link_text_parts).strip()
            self._maybe_set_next_link(self._current_link_href, text)
            self._current_link_href = None
            self._current_link_text_parts = []

        if tag == "div":
            if self._body_stack:
                self._body_stack.pop()

        if tag == "html" and self.next_href is None and self._best_fallback_next is not None:
            self.next_href = self._best_fallback_next[1]

    def handle_data(self, data: str) -> None:
        if self._current_link_href is not None:
            self._current_link_text_parts.append(data)

    def _is_in_body(self) -> bool:
        return any(self._body_stack)

    def _maybe_set_next_link(self, href: str, text: str) -> None:
        absolute = urljoin(self.base_url, href)
        parsed = urlparse(absolute)
        if not parsed.path.endswith("/Special:AllPages"):
            return

        query = parse_qs(parsed.query)
        from_value = query.get("from", [None])[0]
        if not from_value:
            return

        if "下一页" in text or "next page" in text.lower():
            if self.next_href is None:
                self.next_href = href
            return

        # Fallback when localized page text changes: choose the greatest "from" cursor.
        if self._current_from is None or from_value > self._current_from:
            if self._best_fallback_next is None or from_value > self._best_fallback_next[0]:
                self._best_fallback_next = (from_value, href)

    def get_next_href(self) -> str | None:
        if self.next_href is not None:
            return self.next_href
        if self._best_fallback_next is not None:
            return self._best_fallback_next[1]
        return None


def href_to_title(href: str, base_url: str) -> str | None:
    absolute = urljoin(base_url, href)
    parsed = urlparse(absolute)

    if not parsed.path.startswith("/w/"):
        return None

    raw_title = parsed.path[len("/w/") :]
    if not raw_title or raw_title.startswith("Special:"):
        return None

    return unquote(raw_title).replace("_", " ").strip()


def fetch_html(url: str) -> str:
    request = Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": USER_AGENT,
        },
    )
    with urlopen(request, timeout=20) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def api_get(api_url: str, params: dict[str, object]) -> dict:
    query = urlencode(params, doseq=True)
    request = Request(
        f"{api_url}?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urlopen(request, timeout=20) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset))


def detect_api_url(explicit_api_url: str | None = None) -> str:
    candidates = [explicit_api_url] if explicit_api_url else API_CANDIDATES
    failures: list[str] = []

    for api_url in candidates:
        try:
            data = api_get(
                api_url,
                {
                    "action": "query",
                    "meta": "siteinfo",
                    "format": "json",
                },
            )
            if "query" in data:
                return api_url
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            failures.append(f"{api_url}: {exc}")

    joined = "\n".join(failures) or "No candidate API URLs were checked."
    raise RuntimeError(f"无法连接到中文 mcwiki 的 API。\n{joined}")


def fetch_page(api_url: str, title: str) -> PageResult:
    data = api_get(
        api_url,
        {
            "action": "parse",
            "page": title,
            "prop": "text|displaytitle",
            "redirects": 1,
            "format": "json",
        },
    )
    if "error" not in data:
        parsed = data["parse"]
        html_fragment = parsed["text"]["*"]
        extractor = MediaWikiTextExtractor()
        extractor.feed(html_fragment)
        text = extractor.get_text()
        page_title = parsed.get("title") or title
        source_url = build_page_url(page_title)
        return PageResult(title=page_title, source_url=source_url, text=text)

    # Fallback to plain extract when parse is unavailable.
    extract_data = api_get(
        api_url,
        {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "redirects": 1,
            "explaintext": 1,
            "exsectionformat": "plain",
            "format": "json",
        },
    )
    pages = extract_data.get("query", {}).get("pages", {})
    if pages:
        page = next(iter(pages.values()))
        page_title = page.get("title") or title
        extract_text = normalize_plaintext(page.get("extract", ""))
        if extract_text:
            source_url = build_page_url(page_title)
            return PageResult(title=page_title, source_url=source_url, text=extract_text)

    raise RuntimeError(data["error"].get("info", "Unknown MediaWiki API error"))


def normalize_plaintext(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    previous = None

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1", line)
        line = re.sub(r"\s+", " ", line)
        if not line:
            continue
        if line == "导航":
            continue
        if line == previous:
            continue
        lines.append(line)
        previous = line

    return "\n".join(lines)


def iter_category_titles(api_url: str, category: str, limit: int | None) -> Iterable[str]:
    category_title = category if category.startswith("分类:") else f"分类:{category}"
    cursor: dict[str, object] = {}
    emitted = 0

    while True:
        remaining = None if limit is None else max(limit - emitted, 0)
        if remaining == 0:
            return

        batch_size = 500 if remaining is None else min(500, remaining)
        data = api_get(
            api_url,
            {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category_title,
                "cmtype": "page",
                "cmnamespace": 0,
                "cmlimit": batch_size,
                "format": "json",
                **cursor,
            },
        )

        members = data.get("query", {}).get("categorymembers", [])
        for member in members:
            yield member["title"]
            emitted += 1
            if limit is not None and emitted >= limit:
                return

        if "continue" not in data:
            return
        cursor = data["continue"]


def iter_all_titles(all_pages_url: str, limit: int | None) -> Iterable[str]:
    next_url: str | None = all_pages_url
    visited_pages: set[str] = set()
    seen_titles: set[str] = set()
    emitted = 0

    while next_url:
        if next_url in visited_pages:
            return
        visited_pages.add(next_url)

        html = fetch_html(next_url)
        parser = SpecialAllPagesParser(next_url)
        parser.feed(html)

        for title in parser.titles:
            if title in seen_titles:
                continue
            seen_titles.add(title)

            yield title
            emitted += 1
            if limit is not None and emitted >= limit:
                return

        next_href = parser.get_next_href()
        next_url = urljoin(next_url, next_href) if next_href else None


def build_page_url(title: str) -> str:
    normalized = title.replace(" ", "_")
    return f"https://zh.minecraft.wiki/w/{quote(normalized, safe='')}"


def safe_filename(title: str) -> str:
    filename = re.sub(r'[<>:"/\\\\|?*]', "_", title).strip(" .")
    filename = re.sub(r"\s+", "_", filename)
    return filename or "untitled"


def save_result(result: PageResult, output_dir: str, output_format: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    extension = "txt" if output_format == "txt" else "json"
    path = os.path.join(output_dir, f"{safe_filename(result.title)}.{extension}")

    if output_format == "txt":
        content = result.text
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
    else:
        payload = {
            "title": result.title,
            "source_url": result.source_url,
            "text": result.text,
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    return path


def collect_titles_for_index(title_iter: Iterable[str]) -> list[str]:
    titles: list[str] = []
    for index, title in enumerate(title_iter, 1):
        titles.append(title)
        if index % 500 == 0:
            print(f"[INDEX] 已收集 {index} 个页面")
    return titles


def save_global_index(
    titles: list[str],
    output_dir: str,
    scope: str,
    category: str | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    if scope == "category" and category:
        filename = f"global_index_category_{safe_filename(category)}.json"
    else:
        filename = "global_index_all_pages.json"

    payload = {
        "scope": scope,
        "category": category,
        "total": len(titles),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "titles": titles,
    }
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return path


def load_titles_from_index(index_file: str, limit: int | None) -> list[str]:
    with open(index_file, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise RuntimeError("索引文件格式错误：根节点应为 JSON 对象")

    titles = payload.get("titles")
    if not isinstance(titles, list):
        raise RuntimeError("索引文件格式错误：缺少 titles 列表")

    normalized_titles = [title for title in titles if isinstance(title, str) and title.strip()]
    if limit is not None:
        normalized_titles = normalized_titles[:limit]
    return normalized_titles


def default_progress_file(output_dir: str) -> str:
    return os.path.join(output_dir, "crawl_progress.jsonl")


def load_completed_titles(progress_file: str) -> set[str]:
    completed: set[str] = set()
    if not os.path.exists(progress_file):
        return completed

    with open(progress_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(payload, dict) or payload.get("status") != "ok":
                continue

            requested_title = payload.get("requested_title")
            if isinstance(requested_title, str) and requested_title.strip():
                completed.add(requested_title)

    return completed


def append_progress_record(progress_file: str, record: dict[str, object], lock: threading.Lock) -> None:
    os.makedirs(os.path.dirname(progress_file) or ".", exist_ok=True)
    with lock:
        with open(progress_file, "a", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in {408, 425, 429, 500, 502, 503, 504}

    if isinstance(exc, (URLError, TimeoutError, json.JSONDecodeError)):
        return True

    message = str(exc).lower()
    return "maxlag" in message or "timeout" in message or "temporarily unavailable" in message


def fetch_and_save_title(
    api_url: str,
    title: str,
    output_dir: str,
    output_format: str,
    pacer: RequestPacer,
    max_retries: int,
    retry_backoff: float,
    retry_jitter: float,
) -> tuple[PageResult, str]:
    max_attempts = max_retries + 1
    for attempt in range(1, max_attempts + 1):
        try:
            pacer.wait()
            result = fetch_page(api_url, title)
            output_path = save_result(result, output_dir, output_format)
            return result, output_path
        except Exception as exc:
            if attempt >= max_attempts or not is_retryable_error(exc):
                raise

            delay = retry_backoff * (2 ** (attempt - 1)) + random.uniform(0.0, retry_jitter)
            print(
                f"[RETRY] {title} 第 {attempt}/{max_retries} 次重试，{delay:.1f}s 后继续：{exc}",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise RuntimeError(f"抓取失败：{title}")


def crawl_titles(
    api_url: str,
    titles: list[str],
    output_dir: str,
    output_format: str,
    sleep_seconds: float,
    print_stdout: bool,
    workers: int,
    max_retries: int,
    retry_backoff: float,
    retry_jitter: float,
    progress_file: str,
    resume: bool,
) -> int:
    if not titles:
        print("未找到可抓取页面。")
        return 1

    total = len(titles)
    completed_titles = load_completed_titles(progress_file) if resume else set()
    pending_titles = [title for title in titles if title not in completed_titles]

    if resume:
        print(
            f"[RESUME] 状态文件: {progress_file}，历史完成 {len(completed_titles)} 个，待抓取 {len(pending_titles)} 个"
        )

    if not pending_titles:
        print("全部页面已抓取完成，无需继续。")
        return 0

    success = 0
    failures = 0
    progress_lock = threading.Lock()

    if workers > 1 and print_stdout:
        print("提示：并发模式下 --stdout 输出可能交错，建议用于小批量调试。")

    pacer = RequestPacer(interval_seconds=sleep_seconds, jitter_seconds=retry_jitter)
    completed = total - len(pending_titles)

    if workers == 1:
        for title in pending_titles:
            completed += 1
            progress = f"[{completed}/{total} {completed / total:.1%}]"
            try:
                result, output_path = fetch_and_save_title(
                    api_url=api_url,
                    title=title,
                    output_dir=output_dir,
                    output_format=output_format,
                    pacer=pacer,
                    max_retries=max_retries,
                    retry_backoff=retry_backoff,
                    retry_jitter=retry_jitter,
                )
                success += 1
                print(f"{progress} [OK] {result.title} -> {output_path}")
                append_progress_record(
                    progress_file,
                    {
                        "time": datetime.datetime.now(datetime.UTC).isoformat(),
                        "status": "ok",
                        "requested_title": title,
                        "resolved_title": result.title,
                        "output_path": output_path,
                    },
                    progress_lock,
                )
                if print_stdout:
                    print(result.text)
                    print()
            except Exception as exc:
                failures += 1
                print(f"{progress} [ERROR] {title}: {exc}", file=sys.stderr)
                append_progress_record(
                    progress_file,
                    {
                        "time": datetime.datetime.now(datetime.UTC).isoformat(),
                        "status": "error",
                        "requested_title": title,
                        "error": str(exc),
                    },
                    progress_lock,
                )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_title = {
                executor.submit(
                    fetch_and_save_title,
                    api_url,
                    title,
                    output_dir,
                    output_format,
                    pacer,
                    max_retries,
                    retry_backoff,
                    retry_jitter,
                ): title
                for title in pending_titles
            }

            for future in concurrent.futures.as_completed(future_to_title):
                completed += 1
                progress = f"[{completed}/{total} {completed / total:.1%}]"
                title = future_to_title[future]
                try:
                    result, output_path = future.result()
                    success += 1
                    print(f"{progress} [OK] {result.title} -> {output_path}")
                    append_progress_record(
                        progress_file,
                        {
                            "time": datetime.datetime.now(datetime.UTC).isoformat(),
                            "status": "ok",
                            "requested_title": title,
                            "resolved_title": result.title,
                            "output_path": output_path,
                        },
                        progress_lock,
                    )
                    if print_stdout:
                        print(result.text)
                        print()
                except Exception as exc:
                    failures += 1
                    print(f"{progress} [ERROR] {title}: {exc}", file=sys.stderr)
                    append_progress_record(
                        progress_file,
                        {
                            "time": datetime.datetime.now(datetime.UTC).isoformat(),
                            "status": "error",
                            "requested_title": title,
                            "error": str(exc),
                        },
                        progress_lock,
                    )

    print(f"完成。成功 {success} 个，失败 {failures} 个。")
    return 1 if success == 0 else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抓取中文 mcwiki 页面，并提取页面中的文字信息。"
    )
    target_group = parser.add_mutually_exclusive_group(required=False)
    target_group.add_argument("--title", help="单个页面标题，例如：钻石剑")
    target_group.add_argument("--category", help="分类名，例如：物品")
    target_group.add_argument(
        "--all-pages",
        action="store_true",
        help="抓取全部主命名空间页面",
    )
    target_group.add_argument(
        "--index-file",
        default=None,
        help="从已有索引文件抓取内容，例如 output/global_index_all_pages.json",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"正文输出目录，默认是 {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help=f"索引输出目录，默认是 {DEFAULT_INDEX_DIR}",
    )
    parser.add_argument(
        "--format",
        choices=("json", "txt"),
        default="json",
        help="输出格式，默认是 json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="批量抓取时最多处理多少个页面（用于 --category/--all-pages/--index-file）",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="批量抓取时全局请求最小间隔秒数，默认 1.0",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="批量抓取并发线程数，默认 1（建议 3-10）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="单个页面失败后的最大重试次数，默认 2",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.0,
        help="重试指数退避基数秒，默认 1.0",
    )
    parser.add_argument(
        "--retry-jitter",
        type=float,
        default=0.2,
        help="请求与重试随机抖动秒，默认 0.2",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="手动指定 MediaWiki API 地址",
    )
    parser.add_argument(
        "--all-pages-url",
        default=SPECIAL_ALL_PAGES_URL,
        help=f"全站索引来源页面，默认是 {SPECIAL_ALL_PAGES_URL}",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="保存文件的同时，把抓到的文本打印到终端",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="只创建索引文件，不抓取页面内容（仅用于 --category 或 --all-pages）",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="断点续抓状态文件路径（默认在正文输出目录下生成 crawl_progress.jsonl）",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用中断重续，忽略历史状态文件",
    )

    args = parser.parse_args()
    if not any((args.title, args.category, args.all_pages, args.index_file)):
        parser.error("必须指定 --title、--category、--all-pages 或 --index-file")

    if args.index_only and not (args.category or args.all_pages):
        parser.error("--index-only 仅可用于 --category 或 --all-pages")

    if args.workers < 1:
        parser.error("--workers 必须 >= 1")

    if args.max_retries < 0:
        parser.error("--max-retries 必须 >= 0")

    if args.retry_backoff < 0:
        parser.error("--retry-backoff 必须 >= 0")

    if args.retry_jitter < 0:
        parser.error("--retry-jitter 必须 >= 0")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须 >= 1")

    return args


def crawl_single(api_url: str, title: str, output_dir: str, output_format: str, print_stdout: bool) -> int:
    result = fetch_page(api_url, title)
    output_path = save_result(result, output_dir, output_format)
    print(f"[OK] {result.title} -> {output_path}")
    if print_stdout:
        print(result.text)
    return 0


def crawl_category(
    api_url: str,
    category: str,
    output_dir: str,
    index_dir: str,
    output_format: str,
    limit: int | None,
    sleep_seconds: float,
    print_stdout: bool,
    index_only: bool,
    workers: int,
    max_retries: int,
    retry_backoff: float,
    retry_jitter: float,
    progress_file: str,
    resume: bool,
) -> int:
    titles = collect_titles_for_index(iter_category_titles(api_url, category, limit))
    if not titles:
        print("未找到可抓取页面。")
        return 1

    index_path = save_global_index(
        titles=titles,
        output_dir=index_dir,
        scope="category",
        category=category,
    )
    print(f"全局索引已生成: {index_path}（共 {len(titles)} 个页面）")

    if index_only:
        print("索引创建完成，已按要求跳过内容抓取。")
        return 0

    return crawl_titles(
        api_url=api_url,
        titles=titles,
        output_dir=output_dir,
        output_format=output_format,
        sleep_seconds=sleep_seconds,
        print_stdout=print_stdout,
        workers=workers,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_jitter=retry_jitter,
        progress_file=progress_file,
        resume=resume,
    )


def crawl_all_pages(
    api_url: str,
    all_pages_url: str,
    output_dir: str,
    index_dir: str,
    output_format: str,
    limit: int | None,
    sleep_seconds: float,
    print_stdout: bool,
    index_only: bool,
    workers: int,
    max_retries: int,
    retry_backoff: float,
    retry_jitter: float,
    progress_file: str,
    resume: bool,
) -> int:
    titles = collect_titles_for_index(iter_all_titles(all_pages_url, limit))
    if not titles:
        print("未找到可抓取页面。")
        return 1

    index_path = save_global_index(
        titles=titles,
        output_dir=index_dir,
        scope="all-pages",
    )
    print(f"全局索引已生成: {index_path}（共 {len(titles)} 个页面）")

    if index_only:
        print("索引创建完成，已按要求跳过内容抓取。")
        return 0

    return crawl_titles(
        api_url=api_url,
        titles=titles,
        output_dir=output_dir,
        output_format=output_format,
        sleep_seconds=sleep_seconds,
        print_stdout=print_stdout,
        workers=workers,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_jitter=retry_jitter,
        progress_file=progress_file,
        resume=resume,
    )


def crawl_from_index(
    api_url: str,
    index_file: str,
    output_dir: str,
    output_format: str,
    limit: int | None,
    sleep_seconds: float,
    print_stdout: bool,
    workers: int,
    max_retries: int,
    retry_backoff: float,
    retry_jitter: float,
    progress_file: str,
    resume: bool,
) -> int:
    titles = load_titles_from_index(index_file, limit)
    if not titles:
        print("索引文件中没有可抓取页面。")
        return 1

    print(f"已加载索引: {index_file}（将抓取 {len(titles)} 个页面）")
    return crawl_titles(
        api_url=api_url,
        titles=titles,
        output_dir=output_dir,
        output_format=output_format,
        sleep_seconds=sleep_seconds,
        print_stdout=print_stdout,
        workers=workers,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_jitter=retry_jitter,
        progress_file=progress_file,
        resume=resume,
    )


def main() -> int:
    args = parse_args()
    progress_file = args.progress_file or default_progress_file(args.output_dir)
    resume = not args.no_resume

    need_api = bool(
        args.title
        or args.index_file
        or args.category
        or (args.all_pages and not args.index_only)
    )
    api_url = detect_api_url(args.api_url) if need_api else ""
    if need_api:
        print(f"使用 API: {api_url}")

    if args.title:
        return crawl_single(
            api_url=api_url,
            title=args.title,
            output_dir=args.output_dir,
            output_format=args.format,
            print_stdout=args.stdout,
        )

    if args.category:
        return crawl_category(
            api_url=api_url,
            category=args.category,
            output_dir=args.output_dir,
            index_dir=args.index_dir,
            output_format=args.format,
            limit=args.limit,
            sleep_seconds=args.sleep,
            print_stdout=args.stdout,
            index_only=args.index_only,
            workers=args.workers,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            retry_jitter=args.retry_jitter,
            progress_file=progress_file,
            resume=resume,
        )

    if args.index_file:
        return crawl_from_index(
            api_url=api_url,
            index_file=args.index_file,
            output_dir=args.output_dir,
            output_format=args.format,
            limit=args.limit,
            sleep_seconds=args.sleep,
            print_stdout=args.stdout,
            workers=args.workers,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            retry_jitter=args.retry_jitter,
            progress_file=progress_file,
            resume=resume,
        )

    return crawl_all_pages(
        api_url=api_url,
        all_pages_url=args.all_pages_url,
        output_dir=args.output_dir,
        index_dir=args.index_dir,
        output_format=args.format,
        limit=args.limit,
        sleep_seconds=args.sleep,
        print_stdout=args.stdout,
        index_only=args.index_only,
        workers=args.workers,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        retry_jitter=args.retry_jitter,
        progress_file=progress_file,
        resume=resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
