from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session, sessionmaker

from core.entities.knowledge_entities import PreviewDetail
from core.rag.entities import ParentMode, Rule, Segmentation
from core.rag.extractor.entity.extract_setting import ExtractionResult
from core.rag.index_processor.constant.index_type import IndexTechniqueType
from core.rag.index_processor.index_processor_base import AttachmentRef, IndexingMutationGuard
from core.rag.index_processor.processor.parent_child_index_processor import ParentChildIndexProcessor
from core.rag.models.document import AttachmentDocument, ChildDocument, Document
from models.account import Account
from models.dataset import ChildChunk, Dataset, DatasetProcessRule, DocumentCreatedFrom, DocumentSegment
from models.dataset import Document as DatasetDocument
from models.enums import DataSourceType
from tests.unit_tests.config_override import config_overrides_context


class _PublishAllAttachmentVectorStore:
    def __init__(self) -> None:
        self.routes: list[tuple[str | None, str, str, str]] = []

    def publish_bound(
        self,
        session: Session,
        *,
        tenant_id: str,
        dataset_id: str,
        collection_binding_id: str | None,
        indexing_technique: str,
        attachment_refs: Sequence[AttachmentRef],
        vector_type: str,
        collection_name: str,
        routing_index_struct: str | None,
        publish: Callable[
            [frozenset[AttachmentRef], Callable[[Sequence[str]], None], Callable[[Sequence[str]], None]], None
        ],
        mutation_guard: IndexingMutationGuard | None = None,
    ) -> frozenset[AttachmentRef]:
        del session, tenant_id, dataset_id, routing_index_struct, mutation_guard
        self.routes.append((collection_binding_id, indexing_technique, vector_type, collection_name))
        bound_refs = frozenset(attachment_refs)
        publish(bound_refs, self.before_write, self.after_write)
        return bound_refs

    @staticmethod
    def before_write(_attachment_ids: Sequence[str]) -> None:
        pass

    @staticmethod
    def after_write(_attachment_ids: Sequence[str]) -> None:
        pass

    def delete_unreferenced(
        self,
        session: Session,
        *,
        tenant_id: str,
        dataset_id: str,
        collection_binding_id: str | None,
        indexing_technique: str,
        vector_type: str,
        collection_name: str,
        routing_index_struct: str | None,
        document_ids: Sequence[str] | None = None,
        segment_ids: Sequence[str] | None = None,
        candidate_attachment_ids: Sequence[str] | None = None,
        mutation_guard: IndexingMutationGuard | None = None,
        validate: Callable[[Session], None] | None = None,
    ) -> tuple[str, ...]:
        _ = (
            session,
            tenant_id,
            dataset_id,
            collection_binding_id,
            indexing_technique,
            vector_type,
            collection_name,
            routing_index_struct,
            document_ids,
            segment_ids,
            candidate_attachment_ids,
            mutation_guard,
            validate,
        )
        raise AssertionError("attachment deletion is outside these processor tests")


@dataclass(frozen=True)
class _ChildVectorRouteCall:
    session: Session
    tenant_id: str
    dataset_id: str
    collection_binding_id: str | None
    indexing_technique: str
    doc_form: str
    node_ids: tuple[str, ...]
    vector_type: str
    collection_name: str


class _RecordingChildVectorRouteStore:
    def __init__(self) -> None:
        self.calls: list[_ChildVectorRouteCall] = []
        self.on_persist: Callable[[], None] | None = None

    def persist(
        self,
        session: Session,
        *,
        tenant_id: str,
        dataset_id: str,
        collection_binding_id: str | None,
        indexing_technique: str,
        doc_form: str,
        node_ids: Sequence[str],
        vector_type: str,
        collection_name: str,
        routing_index_struct: str | None,
    ) -> None:
        _ = routing_index_struct
        self.calls.append(
            _ChildVectorRouteCall(
                session=session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                collection_binding_id=collection_binding_id,
                indexing_technique=indexing_technique,
                doc_form=doc_form,
                node_ids=tuple(node_ids),
                vector_type=vector_type,
                collection_name=collection_name,
            )
        )
        if self.on_persist is not None:
            self.on_persist()


class TestParentChildIndexProcessor:
    session: Session
    session_factory: sessionmaker[Session]

    @pytest.fixture(autouse=True)
    def _inject_sqlite_sessions(
        self,
        sqlite_session: Session,
        sqlite_session_factory: sessionmaker[Session],
    ) -> None:
        self.session = sqlite_session
        self.session_factory = sqlite_session_factory

        self.attachment_vectors = _PublishAllAttachmentVectorStore()
        self.child_vector_routes = _RecordingChildVectorRouteStore()

    @pytest.fixture
    def processor(self) -> ParentChildIndexProcessor:
        return ParentChildIndexProcessor(
            attachment_vectors=self.attachment_vectors,
            child_vector_routes=self.child_vector_routes,
        )

    @pytest.fixture
    def dataset(self) -> Dataset:
        dataset = Dataset(
            id="dataset-1",
            tenant_id="tenant-1",
            name="Dataset",
            created_by="user-1",
            indexing_technique=IndexTechniqueType.HIGH_QUALITY,
            is_multimodal=True,
        )
        self.session.add(dataset)
        self.session.flush()
        return dataset

    @pytest.fixture
    def dataset_document(self, dataset: Dataset) -> DatasetDocument:
        document = DatasetDocument(
            id="doc-1",
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            position=1,
            data_source_type=DataSourceType.UPLOAD_FILE,
            batch="batch-1",
            name="Document",
            created_from=DocumentCreatedFrom.API,
            created_by="user-1",
        )
        self.session.add(document)
        self.session.flush()
        return document

    def _segmentation(self) -> SimpleNamespace:
        return SimpleNamespace(max_tokens=200, chunk_overlap=10, separator="\n")

    def _paragraph_rules(self) -> SimpleNamespace:
        return SimpleNamespace(
            parent_mode=ParentMode.PARAGRAPH,
            segmentation=self._segmentation(),
            subchunk_segmentation=self._segmentation(),
        )

    def _full_doc_rules(self) -> SimpleNamespace:
        return SimpleNamespace(
            parent_mode=ParentMode.FULL_DOC, segmentation=None, subchunk_segmentation=self._segmentation()
        )

    def test_extract_forwards_automatic_flag(self, processor: ParentChildIndexProcessor) -> None:
        extract_setting = Mock()
        session = self.session
        expected = ExtractionResult(documents=[Document(page_content="chunk", metadata={})])

        with patch(
            "core.rag.index_processor.processor.parent_child_index_processor.ExtractProcessor.extract"
        ) as mock_extract:
            mock_extract.return_value = expected
            documents = processor.extract(extract_setting, process_rule_mode="hierarchical", session=session)

        assert documents == expected
        mock_extract.assert_called_once_with(
            extract_setting=extract_setting,
            is_automatic=True,
            session=session,
            on_upload_artifact_created=None,
        )

    def test_transform_validates_process_rule(self, processor: ParentChildIndexProcessor) -> None:
        session = self.session
        with pytest.raises(ValueError, match="No process rule found"):
            processor.transform([Document(page_content="text", metadata={})], process_rule=None, session=session)

        with pytest.raises(ValueError, match="No rules found in process rule"):
            processor.transform(
                [Document(page_content="text", metadata={})], process_rule={"mode": "custom"}, session=session
            )

    def test_transform_paragraph_requires_segmentation(self, processor: ParentChildIndexProcessor) -> None:
        rules = SimpleNamespace(parent_mode=ParentMode.PARAGRAPH, segmentation=None)

        with patch(
            "core.rag.index_processor.processor.parent_child_index_processor.Rule.model_validate", return_value=rules
        ):
            with pytest.raises(ValueError, match="No segmentation found in rules"):
                processor.transform(
                    [Document(page_content="text", metadata={})],
                    process_rule={"mode": "custom", "rules": {"enabled": True}},
                    session=self.session,
                )

    def test_transform_paragraph_builds_parent_and_child_docs(self, processor: ParentChildIndexProcessor) -> None:
        splitter = Mock()
        splitter.split_documents.return_value = [
            Document(page_content=".parent", metadata={}),
            Document(page_content=" ", metadata={}),
        ]
        parent_document = Document(page_content="source", metadata={"dataset_id": "dataset-1", "document_id": "doc-1"})
        child_docs = [ChildDocument(page_content="child-1", metadata={"dataset_id": "dataset-1"})]

        with (
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.Rule.model_validate",
                return_value=self._paragraph_rules(),
            ),
            patch.object(processor, "_get_splitter", return_value=splitter),
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.CleanProcessor.clean",
                return_value=".parent",
            ),
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.helper.generate_text_hash",
                return_value="hash",
            ),
            patch.object(
                processor, "_get_content_files", return_value=[AttachmentDocument(page_content="image", metadata={})]
            ),
            patch.object(processor, "split_child_nodes", return_value=child_docs),
        ):
            result = processor.transform(
                [parent_document],
                process_rule={"mode": "custom", "rules": {"enabled": True}},
                preview=False,
                tenant_id="tenant-1",
                session=self.session,
            )

        assert len(result) == 1
        assert result[0].page_content == "parent"
        assert result[0].children == child_docs
        assert result[0].attachments is not None

    def test_transform_preview_returns_after_ten_parent_chunks(self, processor: ParentChildIndexProcessor) -> None:
        splitter = Mock()
        splitter.split_documents.return_value = [Document(page_content=f"chunk-{i}", metadata={}) for i in range(10)]
        documents = [
            Document(page_content="doc-1", metadata={"dataset_id": "dataset-1", "document_id": "doc-1"}),
            Document(page_content="doc-2", metadata={"dataset_id": "dataset-1", "document_id": "doc-2"}),
        ]

        with (
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.Rule.model_validate",
                return_value=self._paragraph_rules(),
            ),
            patch.object(processor, "_get_splitter", return_value=splitter),
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.CleanProcessor.clean",
                side_effect=lambda text, _: text,
            ),
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.helper.generate_text_hash",
                return_value="hash",
            ),
            patch.object(processor, "_get_content_files", return_value=[]),
            patch.object(processor, "split_child_nodes", return_value=[]),
        ):
            result = processor.transform(
                documents,
                process_rule={"mode": "custom", "rules": {"enabled": True}},
                preview=True,
                tenant_id="tenant-1",
                session=self.session,
            )

        assert len(result) == 10

    def test_transform_full_doc_mode_trims_children_for_preview(self, processor: ParentChildIndexProcessor) -> None:
        docs = [
            Document(page_content="first", metadata={"dataset_id": "dataset-1", "document_id": "doc-1"}),
            Document(page_content="second", metadata={"dataset_id": "dataset-1", "document_id": "doc-1"}),
        ]
        child_docs = [ChildDocument(page_content=f"child-{i}", metadata={}) for i in range(5)]

        with (
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.Rule.model_validate",
                return_value=self._full_doc_rules(),
            ),
            patch.object(
                processor, "_get_content_files", return_value=[AttachmentDocument(page_content="image", metadata={})]
            ),
            patch.object(processor, "split_child_nodes", return_value=child_docs),
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.helper.generate_text_hash",
                return_value="hash",
            ),
            config_overrides_context(CHILD_CHUNKS_PREVIEW_NUMBER=2),
        ):
            result = processor.transform(
                docs,
                process_rule={"mode": "hierarchical", "rules": {"enabled": True}},
                preview=True,
                tenant_id="tenant-1",
                session=self.session,
            )

        assert len(result) == 1
        assert len(result[0].children or []) == 2
        assert result[0].attachments is not None

    def test_load_creates_vectors_for_child_docs(self, processor: ParentChildIndexProcessor, dataset: Dataset) -> None:
        parent_doc = Document(
            page_content="parent",
            metadata={},
            children=[
                ChildDocument(page_content="child-1", metadata={"doc_id": "child-node-1"}),
                ChildDocument(page_content="child-2", metadata={"doc_id": "child-node-2"}),
            ],
        )
        multimodal_docs = [
            AttachmentDocument(page_content="image", metadata={"document_id": "document-1", "doc_id": "attachment-1"})
        ]
        session = self.session

        with patch("core.rag.index_processor.processor.parent_child_index_processor.Vector") as mock_vector_cls:
            vector = mock_vector_cls.return_value
            vector.get_routing_target.return_value = ("qdrant", "collection-a")
            processor.load(dataset, [parent_doc], multimodal_documents=multimodal_docs, session=session)

        mock_vector_cls.assert_called_once_with(dataset, session=session)
        assert vector.create.call_count == 1
        formatted_docs = vector.create.call_args[0][0]
        assert len(formatted_docs) == 2
        assert all(isinstance(doc, Document) for doc in formatted_docs)
        vector.create_multimodal.assert_called_once_with(
            multimodal_docs,
            before_write=self.attachment_vectors.before_write,
            after_write=self.attachment_vectors.after_write,
        )
        assert self.attachment_vectors.routes == [
            (dataset.collection_binding_id, str(dataset.indexing_technique), "qdrant", "collection-a")
        ]
        assert self.child_vector_routes.calls == [
            _ChildVectorRouteCall(
                session=session,
                tenant_id=dataset.tenant_id,
                dataset_id=dataset.id,
                collection_binding_id=dataset.collection_binding_id,
                indexing_technique=str(dataset.indexing_technique),
                doc_form="hierarchical_model",
                node_ids=("child-node-1", "child-node-2"),
                vector_type="qdrant",
                collection_name="collection-a",
            )
        ]

    def test_load_batches_child_vectors_for_many_parents(
        self, processor: ParentChildIndexProcessor, dataset: Dataset
    ) -> None:
        parent_docs = [
            Document(
                page_content=f"parent-{position}",
                metadata={},
                children=[
                    ChildDocument(page_content=f"child-{position}", metadata={"doc_id": f"child-node-{position}"})
                ],
            )
            for position in range(1000)
        ]

        with patch("core.rag.index_processor.processor.parent_child_index_processor.Vector") as mock_vector_cls:
            vector = mock_vector_cls.return_value
            vector.get_routing_target.return_value = ("qdrant", "collection-a")
            processor.load(dataset, parent_docs, session=self.session)

        vector.create.assert_called_once()
        assert len(vector.create.call_args.args[0]) == 1000

    def test_load_does_not_switch_route_when_vector_publication_fails(
        self, processor: ParentChildIndexProcessor, dataset: Dataset
    ) -> None:
        parent_doc = Document(
            page_content="parent",
            metadata={},
            children=[ChildDocument(page_content="child", metadata={"doc_id": "child-node-1"})],
        )

        with patch("core.rag.index_processor.processor.parent_child_index_processor.Vector") as mock_vector_cls:
            mock_vector_cls.return_value.create.side_effect = RuntimeError("vector publication failed")
            with pytest.raises(RuntimeError, match="vector publication failed"):
                processor.load(dataset, [parent_doc], session=self.session)

        assert self.child_vector_routes.calls == []

    def test_clean_with_precomputed_child_ids(self, processor: ParentChildIndexProcessor, dataset: Dataset) -> None:
        session = self.session

        with (
            patch("core.rag.index_processor.processor.parent_child_index_processor.Vector") as mock_vector_cls,
        ):
            vector = mock_vector_cls.return_value
            processor.clean(
                dataset,
                ["node-1"],
                delete_child_chunks=True,
                precomputed_child_node_ids=["child-1", "child-2"],
                session=session,
            )

        vector.delete_by_ids.assert_called_once_with(["child-1", "child-2"])
        assert session.query(ChildChunk).count() == 0

    def test_clean_queries_child_ids_when_not_precomputed(
        self, processor: ParentChildIndexProcessor, dataset: Dataset
    ) -> None:
        session = self.session
        parent = DocumentSegment(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id="doc-1",
            position=1,
            content="parent",
            word_count=1,
            tokens=1,
            created_by="user-1",
            index_node_id="node-1",
        )
        session.add(parent)
        session.flush()
        session.add_all(
            [
                ChildChunk(
                    tenant_id=dataset.tenant_id,
                    dataset_id=dataset.id,
                    document_id="doc-1",
                    segment_id=parent.id,
                    position=index,
                    content=f"child-{index}",
                    word_count=1,
                    created_by="user-1",
                    index_node_id=f"child-{index}",
                )
                for index in (1, 2)
            ]
        )
        session.flush()

        with (
            patch("core.rag.index_processor.processor.parent_child_index_processor.Vector") as mock_vector_cls,
        ):
            vector = mock_vector_cls.return_value
            processor.clean(dataset, ["node-1"], delete_child_chunks=False, session=session)

        vector.delete_by_ids.assert_called_once_with(["child-1", "child-2"])

    def test_clean_dataset_wide_cleanup(self, processor: ParentChildIndexProcessor, dataset: Dataset) -> None:
        session = self.session

        with (
            patch("core.rag.index_processor.processor.parent_child_index_processor.Vector") as mock_vector_cls,
        ):
            vector = mock_vector_cls.return_value
            processor.clean(dataset, None, delete_child_chunks=True, session=session)

        vector.delete.assert_called_once()
        assert session.query(ChildChunk).count() == 0

    def test_clean_deletes_summaries_when_requested(
        self, processor: ParentChildIndexProcessor, dataset: Dataset
    ) -> None:
        session = self.session
        segment = DocumentSegment(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id="doc-1",
            position=1,
            content="parent",
            word_count=1,
            tokens=1,
            created_by="user-1",
            index_node_id="node-1",
        )
        session.add(segment)
        session.flush()

        with (
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.SummaryIndexAdapter.delete_summaries_for_segments"
            ) as mock_summary,
            patch("core.rag.index_processor.processor.parent_child_index_processor.Vector"),
        ):
            processor.clean(dataset, ["node-1"], delete_summaries=True, precomputed_child_node_ids=[], session=session)

        mock_summary.assert_called_once_with(dataset, [segment.id], session=session)

    def test_clean_deletes_all_summaries_when_node_ids_missing(
        self, processor: ParentChildIndexProcessor, dataset: Dataset
    ) -> None:
        with (
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.SummaryIndexAdapter.delete_summaries_for_segments"
            ) as mock_summary,
            patch("core.rag.index_processor.processor.parent_child_index_processor.Vector"),
        ):
            session = self.session
            processor.clean(dataset, None, delete_summaries=True, session=session)

        mock_summary.assert_called_once_with(dataset, None, session=session)

    def test_split_child_nodes_requires_subchunk_segmentation(self, processor: ParentChildIndexProcessor) -> None:
        rules = Rule(subchunk_segmentation=None)

        with pytest.raises(ValueError, match="No subchunk segmentation found"):
            processor.split_child_nodes(Document(page_content="parent", metadata={}), rules, "custom", None)

    def test_split_child_nodes_generates_child_documents(self, processor: ParentChildIndexProcessor) -> None:
        rules = Rule(subchunk_segmentation=Segmentation(max_tokens=200, chunk_overlap=10, separator="\n"))
        splitter = Mock()
        splitter.split_documents.return_value = [
            Document(page_content=".child-1", metadata={}),
            Document(page_content=" ", metadata={}),
        ]

        with (
            patch.object(processor, "_get_splitter", return_value=splitter),
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.helper.generate_text_hash",
                return_value="hash",
            ),
        ):
            child_docs = processor.split_child_nodes(
                Document(page_content="parent", metadata={}), rules, "custom", None
            )

        assert len(child_docs) == 1
        assert child_docs[0].page_content == "child-1"
        assert child_docs[0].metadata["doc_hash"] == "hash"

    def test_index_creates_process_rule_segments_and_vectors(
        self, processor: ParentChildIndexProcessor, dataset: Dataset, dataset_document: DatasetDocument
    ) -> None:
        parent_childs = SimpleNamespace(
            parent_mode=ParentMode.PARAGRAPH,
            parent_child_chunks=[
                SimpleNamespace(
                    parent_content="parent text",
                    child_contents=["child-1", "child-2"],
                    files=[SimpleNamespace(id="file-1", filename="image.png")],
                )
            ],
        )
        session = self.session
        phase_events: list[str] = []
        event.listen(session, "after_commit", lambda _session: phase_events.append("commit"))

        with (
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.ParentChildStructureChunk.model_validate",
                return_value=parent_childs,
            ),
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.helper.generate_text_hash",
                side_effect=lambda text: f"hash-{text}",
            ),
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.DatasetDocumentStore"
            ) as mock_store_cls,
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.calculate_segment_token_counts"
            ) as mock_token_counter,
            patch("core.rag.index_processor.processor.parent_child_index_processor.Vector") as mock_vector_cls,
        ):
            mock_token_counter.side_effect = lambda **_kwargs: phase_events.append("count") or [11]
            mock_store_cls.return_value.add_documents.side_effect = lambda **_kwargs: phase_events.append("store")
            self.child_vector_routes.on_persist = lambda: phase_events.append("route")
            mock_vector_cls.return_value.get_routing_target.return_value = ("qdrant", "collection-a")
            mock_vector_cls.return_value.create.side_effect = lambda _documents: phase_events.append("vector")
            processor.index(dataset, dataset_document, {"parent_child_chunks": []}, session)

        assert phase_events == ["count", "store", "commit", "vector", "route"]
        dataset_rule = session.get(DatasetProcessRule, dataset_document.dataset_process_rule_id)
        assert dataset_rule is not None
        documents = mock_token_counter.call_args.kwargs["documents"]
        assert [document.page_content for document in documents] == ["parent text"]
        mock_token_counter.assert_called_once_with(dataset=dataset, documents=documents)
        mock_store_cls.return_value.add_documents.assert_called_once_with(
            session=session,
            docs=documents,
            token_counts=[11],
            save_child=True,
        )
        mock_vector_cls.assert_called_once_with(dataset, session=session)
        assert mock_vector_cls.return_value.create.call_count == 1
        mock_vector_cls.return_value.create_multimodal.assert_called_once()
        assert (
            mock_vector_cls.return_value.create_multimodal.call_args.kwargs["before_write"]
            == self.attachment_vectors.before_write
        )
        assert (
            mock_vector_cls.return_value.create_multimodal.call_args.kwargs["after_write"]
            == self.attachment_vectors.after_write
        )
        assert self.attachment_vectors.routes == [
            (dataset.collection_binding_id, str(dataset.indexing_technique), "qdrant", "collection-a")
        ]

    def test_index_uses_content_files_when_files_missing(
        self, processor: ParentChildIndexProcessor, dataset: Dataset, dataset_document: DatasetDocument
    ) -> None:
        parent_childs = SimpleNamespace(
            parent_mode=ParentMode.PARAGRAPH,
            parent_child_chunks=[SimpleNamespace(parent_content="parent", child_contents=["child"], files=None)],
        )
        session = self.session
        loaded_account = Account(name="User", email="user@example.com")
        load_user = Mock(return_value=loaded_account)

        with (
            patch("core.rag.index_processor.processor.parent_child_index_processor.load_account", new=load_user),
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.ParentChildStructureChunk.model_validate",
                return_value=parent_childs,
            ),
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.helper.generate_text_hash",
                return_value="hash",
            ),
            patch.object(
                processor,
                "_get_content_files",
                return_value=[
                    AttachmentDocument(
                        page_content="image",
                        metadata={"document_id": dataset_document.id, "doc_id": "attachment-1"},
                    )
                ],
            ) as mock_files,
            patch("core.rag.index_processor.processor.parent_child_index_processor.DatasetDocumentStore"),
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.calculate_segment_token_counts",
                return_value=[11],
            ),
            patch("core.rag.index_processor.processor.parent_child_index_processor.Vector") as mock_vector_cls,
        ):
            mock_vector_cls.return_value.get_routing_target.return_value = ("qdrant", "collection-a")
            processor.index(dataset, dataset_document, {"parent_child_chunks": []}, session)

        mock_files.assert_called_once()
        load_user.assert_called_once_with(dataset_document.created_by)
        assert mock_files.call_args.kwargs["current_user"] is loaded_account
        assert mock_files.call_args.kwargs["session"] is session

    def test_index_raises_when_account_missing(
        self, processor: ParentChildIndexProcessor, dataset: Dataset, dataset_document: DatasetDocument
    ) -> None:
        parent_childs = SimpleNamespace(
            parent_mode=ParentMode.PARAGRAPH,
            parent_child_chunks=[SimpleNamespace(parent_content="parent", child_contents=["child"], files=None)],
        )

        with (
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.ParentChildStructureChunk.model_validate",
                return_value=parent_childs,
            ),
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.helper.generate_text_hash",
                return_value="hash",
            ),
            patch("core.rag.index_processor.processor.parent_child_index_processor.load_account", return_value=None),
        ):
            with pytest.raises(ValueError, match="Invalid account"):
                processor.index(dataset, dataset_document, {"parent_child_chunks": []}, self.session)

    def test_format_preview_returns_parent_child_structure(self, processor: ParentChildIndexProcessor) -> None:
        parent_childs = SimpleNamespace(
            parent_mode=ParentMode.PARAGRAPH,
            parent_child_chunks=[SimpleNamespace(parent_content="parent", child_contents=["child-1", "child-2"])],
        )

        with patch(
            "core.rag.index_processor.processor.parent_child_index_processor.ParentChildStructureChunk.model_validate",
            return_value=parent_childs,
        ):
            preview = processor.format_preview({"parent_child_chunks": []})

        assert preview["chunk_structure"] == "hierarchical_model"
        assert preview["parent_mode"] == ParentMode.PARAGRAPH
        assert preview["total_segments"] == 1

    def test_generate_summary_preview_sets_summaries(self, processor: ParentChildIndexProcessor) -> None:
        preview_texts = [PreviewDetail(content="chunk-1"), PreviewDetail(content="chunk-2")]
        session = self.session
        worker_sessions = [self.session_factory(), self.session_factory()]

        with (
            patch(
                "core.rag.index_processor.processor.parent_child_index_processor.session_factory.create_session",
                side_effect=[nullcontext(worker_session) for worker_session in worker_sessions],
            ) as create_session,
            patch(
                "core.rag.index_processor.processor.paragraph_index_processor.ParagraphIndexProcessor.generate_summary",
                return_value=("summary", None),
            ) as mock_generate_summary,
        ):
            result = processor.generate_summary_preview(
                "tenant-1", preview_texts, {"enable": True}, doc_language="English", session=session
            )

        assert all(item.summary == "summary" for item in result)
        call_sessions = [call.kwargs["session"] for call in mock_generate_summary.call_args_list]
        assert create_session.call_count == len(preview_texts)
        assert all(call_session is not session for call_session in call_sessions)
        assert {id(call_session) for call_session in call_sessions} == {
            id(worker_session) for worker_session in worker_sessions
        }

    def test_generate_summary_preview_raises_when_worker_fails(self, processor: ParentChildIndexProcessor) -> None:
        preview_texts = [PreviewDetail(content="chunk-1")]

        with patch(
            "core.rag.index_processor.processor.paragraph_index_processor.ParagraphIndexProcessor.generate_summary",
            side_effect=RuntimeError("summary failed"),
        ):
            with pytest.raises(ValueError, match="Failed to generate summaries"):
                processor.generate_summary_preview("tenant-1", preview_texts, {"enable": True}, session=self.session)

    def test_generate_summary_preview_falls_back_without_flask_context(
        self, processor: ParentChildIndexProcessor
    ) -> None:
        preview_texts = [PreviewDetail(content="chunk-1")]
        fake_current_app = SimpleNamespace(_get_current_object=Mock(side_effect=RuntimeError("no app")))

        with (
            patch("flask.current_app", fake_current_app),
            patch(
                "core.rag.index_processor.processor.paragraph_index_processor.ParagraphIndexProcessor.generate_summary",
                return_value=("summary", None),
            ),
        ):
            result = processor.generate_summary_preview(
                "tenant-1", preview_texts, {"enable": True}, session=self.session
            )

        assert result[0].summary == "summary"

    def test_generate_summary_preview_handles_timeout(
        self, processor: ParentChildIndexProcessor, fake_executor_cls: type
    ) -> None:
        preview_texts = [PreviewDetail(content="chunk-1")]
        future = Mock()
        executor = fake_executor_cls(future)

        with (
            patch("concurrent.futures.ThreadPoolExecutor", return_value=executor),
            patch("concurrent.futures.wait", side_effect=[(set(), {future}), (set(), set())]),
        ):
            with pytest.raises(ValueError, match="timeout"):
                processor.generate_summary_preview("tenant-1", preview_texts, {"enable": True}, session=self.session)

        future.cancel.assert_called_once()
