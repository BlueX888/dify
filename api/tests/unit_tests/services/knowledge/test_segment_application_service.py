from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast
from unittest.mock import Mock, create_autospec

import pytest

from libs.helper import generate_text_hash
from machinery.context import RequestContext
from services.errors.dataset import DatasetAccessDeniedError
from services.knowledge.dataset_access import (
    AccessibleDataset,
    DatasetAccess,
    DatasetAccessReader,
    DatasetAccessRecord,
    DatasetAccessService,
    DatasetAccessSnapshot,
    WorkspaceRoleReader,
)
from services.knowledge.entities.resource_scope import DatasetRef
from services.knowledge.entities.segments import (
    SegmentFileAccess,
    SegmentReadSnapshot,
    SegmentRecord,
    SegmentStateIntent,
    SegmentUpdateArgs,
)
from services.knowledge.segments.application import (
    ChildChunkNotFoundError,
    DatasetSegmentApplicationService,
    SegmentBatchImportDispatcher,
    SegmentBatchImportDispatchError,
    SegmentBatchImportNotFoundError,
    SegmentDatasetModelUnavailableError,
    SegmentDatasetNotFoundError,
    SegmentDatasetRecord,
    SegmentDetail,
    SegmentDocumentIndexingError,
    SegmentDocumentNotFoundError,
    SegmentDocumentRecord,
    SegmentEmbeddingModelUnavailableError,
    SegmentIndex,
    SegmentIndexingState,
    SegmentIndexTarget,
    SegmentInvalidFileTypeError,
    SegmentListFilter,
    SegmentModelGuard,
    SegmentModelProviderError,
    SegmentMutationScope,
    SegmentNotFoundError,
    SegmentPage,
    SegmentPermissionDeniedError,
    SegmentPresenter,
    SegmentScope,
    SegmentScopeReader,
    SegmentStateDispatcher,
    SegmentStateTransitions,
    SegmentStatusUpdateError,
    SegmentStore,
    SegmentUpdateState,
    SegmentUploadCatalog,
    SegmentUploadFileNotFoundError,
)


@dataclass
class SegmentLimitsStub:
    SINGLE_CHUNK_ATTACHMENT_LIMIT: int = 10


def _context() -> RequestContext:
    return RequestContext(
        request_id="request-1",
        trace_id=None,
        account_id="account-1",
        active_workspace_id="workspace-1",
    )


def _dataset(
    *,
    indexing_technique: str = "economy",
) -> SegmentDatasetRecord:
    return SegmentDatasetRecord(
        id="dataset-1",
        workspace_id="workspace-1",
        indexing_technique=indexing_technique,
        embedding_model_provider="provider",
        embedding_model="model",
    )


def _document() -> SegmentDocumentRecord:
    return SegmentDocumentRecord(
        id="document-1",
        dataset_id="dataset-1",
        workspace_id="workspace-1",
        doc_form="text_model",
    )


def _snapshot() -> DatasetAccessSnapshot:
    return DatasetAccessSnapshot(
        DatasetAccessRecord("dataset-1", "workspace-1", "account-1", "all_team_members"), False
    )


def _scope(
    *,
    dataset: SegmentDatasetRecord | None = None,
    document: SegmentDocumentRecord | None = None,
) -> SegmentScope:
    return SegmentScope(
        dataset=_dataset() if dataset is None else dataset,
        document=_document() if document is None else document,
        access_snapshot=_snapshot(),
    )


type ServiceFixture = tuple[DatasetSegmentApplicationService, Mock, Mock, Mock, Mock, Mock, Mock]


def _service(
    scope: SegmentScope | None = None,
    *,
    dataset_access_override: DatasetAccess | None = None,
    uploads: SegmentUploadCatalog | None = None,
    presenter: SegmentPresenter | None = None,
) -> ServiceFixture:
    dataset_access = create_autospec(DatasetAccess, instance=True, spec_set=True)
    dataset_access.check_access.return_value = AccessibleDataset(id="dataset-1", workspace_id="workspace-1")
    store = create_autospec(SegmentStore, instance=True, spec_set=True)
    scopes = create_autospec(SegmentScopeReader, instance=True, spec_set=True)
    scopes.get_segment_scope.return_value = scope or _scope()
    model_guard = create_autospec(SegmentModelGuard, instance=True, spec_set=True)
    indexing_state = create_autospec(SegmentIndexingState, instance=True, spec_set=True)
    indexing_state.is_document_indexing.return_value = False
    indexing_state.is_segment_indexing.return_value = False
    dispatcher = create_autospec(SegmentBatchImportDispatcher, instance=True, spec_set=True)
    state_transitions = create_autospec(SegmentStateTransitions, instance=True, spec_set=True)
    state_transitions.request.return_value = ()
    state_dispatcher = create_autospec(SegmentStateDispatcher, instance=True, spec_set=True)
    if presenter is None:
        default_presenter = create_autospec(SegmentPresenter, instance=True, spec_set=True)
        default_presenter.present.return_value = ()
        resolved_presenter: SegmentPresenter = default_presenter
    else:
        resolved_presenter = presenter
    service = DatasetSegmentApplicationService(
        dataset_access=dataset_access_override or dataset_access,
        store=store,
        presenter=resolved_presenter,
        limits=SegmentLimitsStub(),
        text_hash=generate_text_hash,
        index=create_autospec(SegmentIndex, instance=True, spec_set=True),
        uploads=uploads or create_autospec(SegmentUploadCatalog, instance=True, spec_set=True),
        scopes=scopes,
        model_guard=model_guard,
        indexing_state=indexing_state,
        batch_dispatcher=dispatcher,
        state_transitions=state_transitions,
        state_dispatcher=state_dispatcher,
        job_id_factory=lambda: "job-1",
    )
    return service, dataset_access, store, scopes, model_guard, indexing_state, dispatcher


def test_list_segments_uses_complete_owner_scope() -> None:
    service, dataset_access, store, scopes, _, _, _ = _service()
    expected = SegmentPage(items=(), total=0, total_pages=0, page=1, limit=20)
    store.list_segments.return_value = expected

    result = service.list_segments(
        _context(),
        dataset_id="dataset-1",
        document_id="document-1",
        query=SegmentListFilter(keyword="needle"),
    )

    assert result == expected
    assert scopes.get_segment_scope.call_args.kwargs == {
        "dataset_ref": DatasetRef("workspace-1", "dataset-1"),
        "document_id": "document-1",
        "actor_id": "account-1",
    }
    assert store.list_segments.call_args.args[0] == _document().ref
    dataset_access.check_access.assert_called_once_with(_context(), _snapshot())


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        (SegmentScope(None, None), SegmentDatasetNotFoundError),
        (SegmentScope(_dataset(), None, _snapshot()), SegmentDocumentNotFoundError),
    ],
)
def test_mutations_reject_invalid_scope_before_store_operation(
    scope: SegmentScope,
    expected: type[Exception],
) -> None:
    service, _, store, scopes, _, _, _ = _service(scope)

    with pytest.raises(expected):
        service.delete_segment(
            _context(),
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
        )

    store.delete_segments.assert_not_called()


@pytest.mark.parametrize(
    ("access_error", "expected"),
    [
        (DatasetAccessDeniedError(), SegmentPermissionDeniedError),
    ],
)
def test_dataset_access_errors_are_translated(
    access_error: Exception,
    expected: type[Exception],
) -> None:
    service, dataset_access, store, scopes, _, _, _ = _service()
    dataset_access.check_access.side_effect = access_error

    with pytest.raises(expected):
        service.delete_segment(
            _context(),
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
        )

    store.delete_segments.assert_not_called()


@pytest.mark.parametrize(
    ("kind", "method", "expected", "message"),
    [
        ("bad_request", "create_segment", SegmentEmbeddingModelUnavailableError, "No Embedding Model"),
        ("token", "create_segment", SegmentEmbeddingModelUnavailableError, "token missing"),
        ("bad_request", "update_segment", SegmentDatasetModelUnavailableError, "No Embedding Model"),
        ("token", "update_segment", SegmentDatasetModelUnavailableError, "dataset is unavailable"),
    ],
)
def test_model_failures_are_translated_per_use_case(
    kind: Literal["bad_request", "token"],
    method: str,
    expected: type[Exception],
    message: str,
) -> None:
    service, _, store, scopes, model_guard, _, _ = _service(_scope(dataset=_dataset(indexing_technique="high_quality")))
    model_guard.check.side_effect = SegmentModelProviderError(kind=kind, description="token missing")
    store.get_segment.return_value = None

    if method == "create_segment":

        def invoke() -> object:
            return service.create_segment(
                _context(),
                dataset_id="dataset-1",
                document_id="document-1",
                values={"content": "content"},
            )

    else:

        def invoke() -> object:
            return service.update_segment(
                _context(),
                dataset_id="dataset-1",
                document_id="document-1",
                segment_id="segment-1",
                values={"content": "changed"},
            )

    with pytest.raises(expected, match=message):
        invoke()


@pytest.mark.parametrize("operation", ["delete", "delete_many", "disable", "disable_patch"])
@pytest.mark.parametrize("kind", ["bad_request", "token"])
def test_cleanup_commands_remain_available_without_embedding_credentials(
    operation: str, kind: Literal["bad_request", "token"]
) -> None:
    service, _, store, _, model_guard, _, _ = _service(_scope(dataset=_dataset(indexing_technique="high_quality")))
    model_guard.check.side_effect = SegmentModelProviderError(kind=kind, description="revoked credentials")
    store.get_segments.return_value = (SegmentIndexTarget("segment-1", "node-1", True),)
    store.delete_segments.return_value = (SegmentStateIntent("cleanup-1"),)
    store.get_segment_update_state.return_value = SegmentUpdateState("original", True, None)
    detail = _segment_detail()
    store.get_segment.return_value = SegmentDetail(
        SegmentReadSnapshot(
            file_access=SegmentFileAccess(frozenset(), frozenset()),
            values=detail.data.model_dump(exclude={"sign_content", "child_chunks", "attachments", "summary"}),
            child_chunks=(),
            attachments=(),
            summary=None,
        ),
        "text_model",
    )
    presenter = cast(Mock, vars(service)["_presenter"])
    presenter.present.return_value = (detail.data,)

    if operation == "delete":
        service.delete_segment(_context(), dataset_id="dataset-1", document_id="document-1", segment_id="segment-1")
    elif operation == "delete_many":
        service.delete_segments(
            _context(), dataset_id="dataset-1", document_id="document-1", segment_ids=("segment-1",)
        )
    elif operation == "disable":
        service.change_segment_status(
            _context(),
            dataset_id="dataset-1",
            document_id="document-1",
            segment_ids=("segment-1",),
            action="disable",
        )
    else:
        service.update_segment(
            _context(),
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            values={"content": "original", "enabled": False},
        )

    model_guard.check.assert_not_called()
    if operation.startswith("delete"):
        store.delete_segments.assert_called_once()
    else:
        transitions = cast(Mock, vars(service)["_state_transitions"])
        transitions.request.assert_called_once_with(
            document_ref=_document().ref,
            segment_ids=["segment-1"],
            action="disable",
            actor_id="account-1",
        )


def test_enable_still_requires_embedding_credentials() -> None:
    service, _, _, _, model_guard, _, _ = _service(_scope(dataset=_dataset(indexing_technique="high_quality")))
    model_guard.check.side_effect = SegmentModelProviderError(kind="token", description="revoked credentials")

    with pytest.raises(SegmentDatasetModelUnavailableError, match="revoked credentials"):
        service.change_segment_status(
            _context(),
            dataset_id="dataset-1",
            document_id="document-1",
            segment_ids=("segment-1",),
            action="enable",
        )
    transitions = cast(Mock, vars(service)["_state_transitions"])
    transitions.request.assert_not_called()


def test_status_change_is_fenced_by_document_indexing_state() -> None:
    service, _, store, scopes, _, state, _ = _service()
    state.is_document_indexing.return_value = True

    with pytest.raises(SegmentDocumentIndexingError):
        service.change_segment_status(
            _context(),
            dataset_id="dataset-1",
            document_id="document-1",
            segment_ids=["segment-1"],
            action="disable",
        )

    store.save_segment.assert_not_called()


def test_status_change_translates_operation_failure() -> None:
    service, _, store, scopes, _, _, _ = _service()
    store.get_segments.side_effect = RuntimeError("failed")

    with pytest.raises(SegmentStatusUpdateError, match="failed"):
        service.change_segment_status(
            _context(),
            dataset_id="dataset-1",
            document_id="document-1",
            segment_ids=["segment-1"],
            action="enable",
        )


def test_status_change_dispatches_one_logical_batch() -> None:
    service, _, store, _, _, _, _ = _service()
    segment_ids = [f"segment-{position}" for position in range(1000)]
    store.get_segments.return_value = [_segment_detail(segment_id=segment_id).data for segment_id in segment_ids]
    intents = tuple(SegmentStateIntent(f"intent-{position}") for position in range(1000))
    transitions = cast(Mock, vars(service)["_state_transitions"])
    dispatcher = cast(Mock, vars(service)["_state_dispatcher"])
    transitions.request.return_value = intents

    service.change_segment_status(
        _context(),
        dataset_id="dataset-1",
        document_id="document-1",
        segment_ids=segment_ids,
        action="disable",
    )

    dispatcher.dispatch_many.assert_called_once_with([intent.id for intent in intents])
    dispatcher.dispatch.assert_not_called()


def test_update_and_delete_segment_map_missing_segment() -> None:
    service, _, store, scopes, _, _, _ = _service()
    store.get_segment_update_state.return_value = None
    store.get_segments.return_value = ()

    with pytest.raises(SegmentNotFoundError):
        service.update_segment(
            _context(),
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            values={"content": "content"},
        )
    with pytest.raises(SegmentNotFoundError):
        service.delete_segment(
            _context(),
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
        )


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (None, SegmentUploadFileNotFoundError),
        ("segments.txt", SegmentInvalidFileTypeError),
    ],
)
def test_batch_import_validates_tenant_scoped_upload(filename: str | None, expected: type[Exception]) -> None:
    uploads = create_autospec(SegmentUploadCatalog, instance=True, spec_set=True)
    uploads.get_file_name.return_value = filename
    service, _, store, scopes, _, state, dispatcher = _service(uploads=uploads)

    with pytest.raises(expected):
        service.start_batch_import(
            _context(),
            dataset_id="dataset-1",
            document_id="document-1",
            upload_file_id="file-1",
        )

    assert uploads.get_file_name.call_args.kwargs["workspace_id"] == "workspace-1"
    state.set_batch_waiting.assert_not_called()
    dispatcher.dispatch.assert_not_called()


def test_batch_import_rejects_private_dataset_non_owner_before_reading_upload() -> None:
    uploads = create_autospec(SegmentUploadCatalog, instance=True, spec_set=True)
    workspace_roles = create_autospec(WorkspaceRoleReader, instance=True, spec_set=True)
    workspace_roles.get_legacy_role.return_value = "normal"
    dataset_access = DatasetAccessService(
        datasets=create_autospec(DatasetAccessReader, instance=True, spec_set=True),
        workspace_roles=workspace_roles,
        legacy_permissions_enabled=True,
    )
    private_snapshot = DatasetAccessSnapshot(
        DatasetAccessRecord("dataset-1", "workspace-1", "maintainer-1", "only_me"),
        False,
    )
    service, _, _, scopes, _, state, dispatcher = _service(
        SegmentScope(dataset=_dataset(), document=_document(), access_snapshot=private_snapshot),
        dataset_access_override=dataset_access,
        uploads=uploads,
    )

    with pytest.raises(SegmentPermissionDeniedError):
        service.start_batch_import(
            _context(),
            dataset_id="dataset-1",
            document_id="document-1",
            upload_file_id="file-1",
        )

    scopes.get_segment_scope.assert_called_once_with(
        dataset_ref=DatasetRef("workspace-1", "dataset-1"),
        document_id="document-1",
        actor_id="account-1",
    )
    uploads.get_file_name.assert_not_called()
    state.set_batch_waiting.assert_not_called()
    dispatcher.dispatch.assert_not_called()


def test_batch_import_records_status_before_dispatch() -> None:
    uploads = create_autospec(SegmentUploadCatalog, instance=True, spec_set=True)
    uploads.get_file_name.return_value = "segments.CSV"
    service, _, store, scopes, _, state, dispatcher = _service(uploads=uploads)
    calls: list[str] = []
    state.set_batch_waiting.side_effect = lambda _job_id: calls.append("status")
    dispatcher.dispatch.side_effect = lambda **_kwargs: calls.append("dispatch")

    result = service.start_batch_import(
        _context(),
        dataset_id="dataset-1",
        document_id="document-1",
        upload_file_id="file-1",
    )

    assert result.job_id == "job-1"
    assert calls == ["status", "dispatch"]
    dispatcher.dispatch.assert_called_once_with(
        job_id="job-1",
        upload_file_id="file-1",
        dataset_id="dataset-1",
        document_id="document-1",
        workspace_id="workspace-1",
        actor_id="account-1",
    )


def test_batch_import_translates_dispatch_failure() -> None:
    uploads = create_autospec(SegmentUploadCatalog, instance=True, spec_set=True)
    uploads.get_file_name.return_value = "segments.csv"
    service, _, store, scopes, _, _, dispatcher = _service(uploads=uploads)
    dispatcher.dispatch.side_effect = RuntimeError("queue down")

    with pytest.raises(SegmentBatchImportDispatchError, match="queue down"):
        service.start_batch_import(
            _context(),
            dataset_id="dataset-1",
            document_id="document-1",
            upload_file_id="file-1",
        )


def test_batch_import_status_distinguishes_unknown_job() -> None:
    service, _, _, scopes, _, state, _ = _service()
    state.get_batch_status.return_value = None
    with pytest.raises(SegmentBatchImportNotFoundError):
        service.get_batch_import_status("missing")

    state.get_batch_status.return_value = "completed"
    assert service.get_batch_import_status("job-1").job_status == "completed"


def test_child_chunk_update_distinguishes_missing_parent_and_child() -> None:
    service, _, store, scopes, _, _, _ = _service()
    store.get_children.return_value = None
    with pytest.raises(SegmentNotFoundError):
        service.update_child_chunk(
            _context(),
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            child_chunk_id="child-1",
            content="content",
        )

    store.get_children.return_value = ()
    with pytest.raises(ChildChunkNotFoundError):
        service.update_child_chunk(
            _context(),
            dataset_id="dataset-1",
            document_id="document-1",
            segment_id="segment-1",
            child_chunk_id="child-1",
            content="content",
        )


def _segment_detail(*, segment_id: str = "segment-1", enabled: bool = True) -> SegmentDetail:
    return SegmentDetail(
        data=SegmentRecord(
            id=segment_id,
            position=1,
            document_id="document-1",
            content="content",
            sign_content="content",
            answer=None,
            word_count=7,
            tokens=0,
            keywords=None,
            index_node_id=None,
            index_node_hash=None,
            hit_count=0,
            enabled=enabled,
            disabled_at=None,
            disabled_by=None,
            status="completed",
            created_by="account-1",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
            updated_by=None,
            indexing_at=None,
            completed_at=None,
            error=None,
            stopped_at=None,
            child_chunks=(),
            attachments=(),
            summary=None,
        ),
        doc_form="text_model",
    )


def test_create_segment_returns_store_detail() -> None:
    detail = _segment_detail()
    snapshot = SegmentReadSnapshot(
        file_access=SegmentFileAccess(frozenset(), frozenset()),
        values=detail.data.model_dump(exclude={"sign_content", "child_chunks", "attachments", "summary"}),
        child_chunks=(),
        attachments=(),
        summary=None,
    )
    presenter = create_autospec(SegmentPresenter, instance=True, spec_set=True)
    presenter.present.return_value = (detail.data,)
    service, _, store, scopes, _, _, _ = _service(presenter=presenter)
    store.get_segment.return_value = SegmentDetail(data=snapshot, doc_form=detail.doc_form)

    result = service.create_segment(
        _context(),
        dataset_id="dataset-1",
        document_id="document-1",
        values={"content": "content"},
    )

    assert result == detail
    presenter.present.assert_called_once_with((snapshot,))
    persisted_batch = store.create_segments.call_args.args[1]
    assert persisted_batch[0].created_by == _context().account_id


def test_batch_create_carries_deduplicated_attachment_ids_into_persistence() -> None:
    service, _, store, _, _, _, _ = _service()
    scope = SegmentMutationScope(_dataset(), _document())

    refs = service.mutations.create_segments(
        scope,
        actor_id="author",
        args_list=(
            SegmentUpdateArgs(content="first", attachment_ids=["image-1", "image-1", "image-2"]),
            SegmentUpdateArgs(content="second", attachment_ids=["image-2"]),
        ),
        token_counts=(1, 2),
    )

    assert len(refs) == 2
    persisted_batch = store.create_segments.call_args.args[1]
    assert [segment.id for segment in persisted_batch] == [ref.segment_id for ref in refs]
    assert [segment.attachment_ids for segment in persisted_batch] == [
        ("image-1", "image-2"),
        ("image-2",),
    ]


@pytest.mark.parametrize(
    ("enabled", "indexing", "message"),
    [(False, False, "Can't update disabled segment"), (True, True, "Segment is indexing")],
)
def test_shared_update_rejects_disabled_or_indexing_segments(enabled: bool, indexing: bool, message: str) -> None:
    service, _, store, _, _, state, _ = _service()
    store.get_segment_update_state.return_value = SegmentUpdateState("content", enabled, None)
    state.is_segment_indexing.return_value = indexing
    with pytest.raises(ValueError, match=message):
        service.mutations.update_segment(
            SegmentMutationScope(_dataset(), _document()),
            actor_id="editor",
            segment_id="segment-1",
            args=SegmentUpdateArgs(content="changed"),
        )
    store.save_segment.assert_not_called()


def test_shared_disable_patch_does_not_replace_content() -> None:
    service, _, store, _, _, state, _ = _service()
    store.get_segment_update_state.return_value = SegmentUpdateState("content", True, None)
    service.mutations.update_segment(
        SegmentMutationScope(_dataset(), _document()),
        actor_id="editor",
        segment_id="segment-1",
        args=SegmentUpdateArgs(enabled=False),
    )
    transitions = cast(Mock, vars(service)["_state_transitions"])
    transitions.request.assert_called_once_with(
        document_ref=_document().ref,
        segment_ids=["segment-1"],
        action="disable",
        actor_id="editor",
    )
    store.save_segment.assert_not_called()
    store.get_segment.assert_not_called()


def test_attachment_preflight_rejects_partial_match_before_segment_write() -> None:
    uploads = create_autospec(SegmentUploadCatalog, instance=True, spec_set=True)
    uploads.get_existing_file_ids.return_value = frozenset({"valid-file"})
    service, _, store, _, _, _, _ = _service(uploads=uploads)
    store.get_segment_update_state.return_value = SegmentUpdateState("original", True, None)

    with pytest.raises(SegmentUploadFileNotFoundError, match="missing-file"):
        service.mutations.update_segment(
            SegmentMutationScope(_dataset(), _document()),
            actor_id="editor",
            segment_id="segment-1",
            args=SegmentUpdateArgs(
                content="changed",
                attachment_ids=["valid-file", "missing-file", "valid-file"],
            ),
        )

    uploads.get_existing_file_ids.assert_called_once_with(
        workspace_id="workspace-1",
        file_ids=("valid-file", "missing-file"),
    )
    store.save_segment.assert_not_called()
    index = cast(Mock, vars(service.mutations)["_index"])
    index.update.assert_not_called()
    index.update_attachments.assert_not_called()


def test_body_update_delegates_late_generation_fencing_to_index_gateway() -> None:
    service, _, store, _, _, _, _ = _service()
    store.get_segment_update_state.return_value = SegmentUpdateState("old", True, None)
    index = cast(Mock, vars(service.mutations)["_index"])
    index.count_tokens.return_value = 3

    service.mutations.update_segment(
        SegmentMutationScope(_dataset(indexing_technique="high_quality"), _document()),
        actor_id="editor",
        segment_id="segment-1",
        args=SegmentUpdateArgs(content="new"),
    )

    store.save_segment.assert_called_once()
    changes = store.save_segment.call_args.args[1]
    assert not {"enabled", "disabled_at", "disabled_by"} & changes.keys()
    store.lock_index_write.assert_not_called()
    index.update.assert_called_once_with(
        DatasetRef("workspace-1", "dataset-1").document("document-1").segment("segment-1"),
        keywords=None,
        regenerate_children=False,
    )


def test_shared_writes_do_not_build_segment_detail() -> None:
    service, _, store, _, _, _, _ = _service()
    store.get_segment.side_effect = RuntimeError("attachment URL signing unavailable")
    store.get_segment_update_state.return_value = SegmentUpdateState("content", True, None)
    scope = SegmentMutationScope(_dataset(), _document())

    created = service.mutations.create_segment(scope, actor_id="author", args=SegmentUpdateArgs(content="content"))
    service.mutations.update_segment(
        scope,
        actor_id="editor",
        segment_id=created.segment_id,
        args=SegmentUpdateArgs(content="updated"),
    )

    store.create_segments.assert_called_once()
    store.save_segment.assert_called_once()
    assert store.save_segment.call_args.args[1]["content"] == "updated"
    store.get_segment.assert_not_called()
