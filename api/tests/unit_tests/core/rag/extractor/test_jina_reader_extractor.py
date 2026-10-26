import pytest
from pytest_mock import MockerFixture

from core.rag.extractor.entity.extract_setting import WebsiteJobRef
from core.rag.extractor.jina_reader_extractor import JinaReaderWebExtractor
from tests.unit_tests.core.rag.extractor.website_test_fakes import unused_website_service


def _job_ref(job_id: str = "job-1") -> WebsiteJobRef:
    return WebsiteJobRef(
        tenant_id="tenant-1",
        provider="jinareader",
        provider_instance_id="0" * 64,
        job_id=job_id,
    )


class TestJinaReaderWebExtractor:
    def test_extract_crawl_mode_returns_document(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "core.rag.extractor.jina_reader_extractor.WebsiteService.get_crawl_url_data_by_ref",
            return_value={
                "content": "markdown-content",
                "url": "https://example.com",
                "description": "desc",
                "title": "title",
            },
        )

        extractor = JinaReaderWebExtractor(
            "https://example.com",
            _job_ref(),
            mode="crawl",
            website_service=unused_website_service(),
        )
        docs = extractor.extract()

        assert len(docs) == 1
        assert docs[0].page_content == "markdown-content"
        assert docs[0].metadata == {
            "source_url": "https://example.com",
            "description": "desc",
            "title": "title",
        }

    def test_extract_crawl_mode_with_missing_data_fails(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "core.rag.extractor.jina_reader_extractor.WebsiteService.get_crawl_url_data_by_ref",
            return_value=None,
        )

        extractor = JinaReaderWebExtractor(
            "https://example.com",
            _job_ref(),
            mode="crawl",
            website_service=unused_website_service(),
        )

        with pytest.raises(ValueError, match="returned no content"):
            extractor.extract()

    def test_extract_scrape_mode_refetches_single_page_without_old_job(self, mocker: MockerFixture) -> None:
        mock_get_crawl = mocker.patch(
            "core.rag.extractor.jina_reader_extractor.WebsiteService.get_crawl_url_data_by_ref",
            return_value={"content": "fresh", "url": "https://example.com"},
        )
        extractor = JinaReaderWebExtractor(
            "https://example.com",
            _job_ref(),
            mode="scrape",
            website_service=unused_website_service(),
        )

        documents = extractor.extract()

        assert [document.page_content for document in documents] == ["fresh"]
        mock_get_crawl.assert_called_once_with(_job_ref(), "https://example.com", use_job=False)

    @pytest.mark.parametrize("content", ["", "   ", None])
    def test_extract_rejects_empty_content(self, mocker: MockerFixture, content: object) -> None:
        mocker.patch(
            "core.rag.extractor.jina_reader_extractor.WebsiteService.get_crawl_url_data_by_ref",
            return_value={"content": content},
        )
        extractor = JinaReaderWebExtractor(
            "https://example.com",
            _job_ref(""),
            mode="scrape",
            website_service=unused_website_service(),
        )

        with pytest.raises(ValueError, match="returned empty content"):
            extractor.extract()

    def test_extract_rejects_unknown_mode(self) -> None:
        extractor = JinaReaderWebExtractor(
            "https://example.com",
            _job_ref(),
            mode="unsupported",
            website_service=unused_website_service(),
        )

        with pytest.raises(ValueError, match="Unsupported Jina Reader mode"):
            extractor.extract()
