from typing import override

from core.rag.extractor.entity.extract_setting import WebsiteJobRef
from core.rag.extractor.extractor_base import BaseExtractor
from core.rag.models.document import Document
from services.data_source.website_service import WebsiteService


class FirecrawlWebExtractor(BaseExtractor):
    """
    Crawl and scrape websites and return content in clean llm-ready markdown.

    Args:
        url: The URL to scrape.
        job_id: The crawl job id.
        tenant_id: The tenant id.
        mode: The mode of operation. Defaults to 'scrape'. Options are 'crawl', 'scrape' and 'crawl_return_urls'.
        only_main_content: Only return the main content of the page excluding headers, navs, footers, etc.
    """

    def __init__(
        self,
        url: str,
        job_ref: WebsiteJobRef,
        mode: str = "crawl",
        only_main_content: bool = True,
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
        documents = []
        if self.mode == "crawl":
            crawl_data = self._website_service.get_crawl_url_data_by_ref(self.job_ref, self._url)
            if crawl_data is None:
                return []
            document = Document(
                page_content=crawl_data.get("markdown", ""),
                metadata={
                    "source_url": crawl_data.get("source_url"),
                    "description": crawl_data.get("description"),
                    "title": crawl_data.get("title"),
                },
            )
            documents.append(document)
        elif self.mode == "scrape":
            scrape_data = self._website_service.get_scrape_url_data_by_ref(
                self.job_ref, self._url, self.only_main_content
            )

            document = Document(
                page_content=scrape_data.get("markdown", ""),
                metadata={
                    "source_url": scrape_data.get("source_url"),
                    "description": scrape_data.get("description"),
                    "title": scrape_data.get("title"),
                },
            )
            documents.append(document)
        return documents
