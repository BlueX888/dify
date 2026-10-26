from typing import override

from core.rag.extractor.entity.extract_setting import WebsiteJobRef
from core.rag.extractor.extractor_base import BaseExtractor
from core.rag.models.document import Document
from services.data_source.website_service import WebsiteService


class JinaReaderWebExtractor(BaseExtractor):
    """
    Crawl and scrape websites and return content in clean llm-ready markdown.
    """

    def __init__(
        self,
        url: str,
        job_ref: WebsiteJobRef,
        mode: str = "crawl",
        only_main_content: bool = False,
        *,
        website_service: WebsiteService,
    ):
        """Initialize with url, api_key, base_url and mode."""
        self._website_service = website_service
        self._url = url
        self.job_ref = job_ref
        self.mode = mode
        self.only_main_content = only_main_content

    @override
    def extract(self) -> list[Document]:
        """Extract content from the URL."""
        if self.mode == "crawl":
            use_job = True
        elif self.mode == "scrape":
            # A website sync must fetch the page again instead of reading the
            # completed crawl job that created the document. Jina's single-page
            # reader is represented by the explicit no-job path.
            use_job = False
        else:
            raise ValueError(f"Unsupported Jina Reader mode: {self.mode}")

        data = self._website_service.get_crawl_url_data_by_ref(self.job_ref, self._url, use_job=use_job)
        if data is None:
            raise ValueError(f"Jina Reader returned no content for URL: {self._url}")
        content = data.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Jina Reader returned empty content for URL: {self._url}")
        return [
            Document(
                page_content=content,
                metadata={
                    "source_url": data.get("url"),
                    "description": data.get("description"),
                    "title": data.get("title"),
                },
            )
        ]
