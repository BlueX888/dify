import logging
import time

import click
from celery import shared_task
from sqlalchemy import delete, select

from configs import dify_config
from core.db.session_factory import session_factory
from extensions.ext_storage import storage
from models import (
    AgentDebugConversation,
    Conversation,
    ConversationVariable,
    HumanInputForm,
    HumanInputFormUploadFile,
    HumanInputFormUploadToken,
    Message,
    MessageAgentThought,
    MessageAnnotation,
    MessageChain,
    MessageFeedback,
    MessageFile,
    PinnedConversation,
    SavedMessage,
)
from models.human_input import HumanInputDelivery, HumanInputFormRecipient
from models.tools import ToolConversationVariables, ToolFile
from repositories.upload_file_repository import register_upload_file_cleanup_candidates_in_session

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_DELAY_SECONDS = 30


def _delete_storage_object(file_key: str) -> None:
    try:
        storage.delete(file_key)
    except Exception:
        # A prior attempt may have deleted the object before its DB transaction
        # rolled back. Only suppress the retry when the backend confirms absence.
        if storage.exists(file_key):
            raise
        logger.info("Storage object %s was already absent", file_key)


def _cleanup_conversation_related_data(conversation_id: str) -> bool:
    """Physically remove a soft-deleted conversation and its owned resources.

    The storage object is deleted before its ``ToolFile`` row so a failed attempt
    retains the durable ``file_key`` needed by the next retry.
    """

    with session_factory.create_session() as session:
        conversation = session.scalar(select(Conversation).where(Conversation.id == conversation_id).with_for_update())
        if conversation is not None and not conversation.is_deleted:
            logger.warning("Skipped cleanup for active conversation %s", conversation_id)
            return False

        tool_files = list(
            session.scalars(
                select(ToolFile)
                .where(ToolFile.conversation_id == conversation_id)
                .order_by(ToolFile.id)
                .with_for_update()
            )
        )
        for tool_file in tool_files:
            _delete_storage_object(tool_file.file_key)
            session.delete(tool_file)

        # Persist file cleanup evidence before taking any UploadFile-owner
        # locks. This keeps the file-domain order outbox -> owner; no
        # UploadFile/outbox lock is acquired after MessageFile or Form rows are
        # changed. Existing writer receipts are preserved by the registration
        # helper.
        candidate_uploads = tuple(
            (tenant_id, upload_file_id)
            for tenant_id, upload_file_id in session.execute(
                select(HumanInputFormUploadFile.tenant_id, HumanInputFormUploadFile.upload_file_id)
                .join(HumanInputForm, HumanInputForm.id == HumanInputFormUploadFile.form_id)
                .where(HumanInputForm.conversation_id == conversation_id)
                .order_by(HumanInputFormUploadFile.tenant_id, HumanInputFormUploadFile.upload_file_id)
            )
        )
        candidate_ids_by_tenant: dict[str, list[str]] = {}
        for tenant_id, upload_file_id in candidate_uploads:
            candidate_ids_by_tenant.setdefault(tenant_id, []).append(upload_file_id)
        for tenant_id in sorted(candidate_ids_by_tenant):
            register_upload_file_cleanup_candidates_in_session(
                session,
                workspace_id=tenant_id,
                file_ids=candidate_ids_by_tenant[tenant_id],
            )

        message_ids = select(Message.id).where(Message.conversation_id == conversation_id)
        session.execute(delete(MessageAgentThought).where(MessageAgentThought.message_id.in_(message_ids)))
        session.execute(delete(MessageChain).where(MessageChain.message_id.in_(message_ids)))
        session.execute(delete(MessageFile).where(MessageFile.message_id.in_(message_ids)))
        session.execute(delete(SavedMessage).where(SavedMessage.message_id.in_(message_ids)))
        session.execute(delete(MessageAnnotation).where(MessageAnnotation.conversation_id == conversation_id))
        session.execute(delete(MessageFeedback).where(MessageFeedback.conversation_id == conversation_id))
        session.execute(
            delete(ToolConversationVariables).where(ToolConversationVariables.conversation_id == conversation_id)
        )
        session.execute(delete(ConversationVariable).where(ConversationVariable.conversation_id == conversation_id))

        # Upload publication locks the same aggregate in form -> recipient ->
        # token order before inserting its link. Take those owner locks before
        # deleting any links so a slow publisher cannot commit an orphan after
        # this cleanup has already passed the link table.
        forms = session.scalars(
            select(HumanInputForm)
            .where(HumanInputForm.conversation_id == conversation_id)
            .order_by(HumanInputForm.id)
            .with_for_update()
        ).all()
        form_ids = tuple(form.id for form in forms)
        if form_ids:
            session.scalars(
                select(HumanInputFormRecipient.id)
                .where(HumanInputFormRecipient.form_id.in_(form_ids))
                .order_by(HumanInputFormRecipient.id)
                .with_for_update()
            ).all()
            session.scalars(
                select(HumanInputFormUploadToken.id)
                .where(HumanInputFormUploadToken.form_id.in_(form_ids))
                .order_by(HumanInputFormUploadToken.id)
                .with_for_update()
            ).all()
            current_uploads = {
                (tenant_id, upload_file_id)
                for tenant_id, upload_file_id in session.execute(
                    select(HumanInputFormUploadFile.tenant_id, HumanInputFormUploadFile.upload_file_id)
                    .where(HumanInputFormUploadFile.form_id.in_(form_ids))
                    .order_by(HumanInputFormUploadFile.tenant_id, HumanInputFormUploadFile.upload_file_id)
                    .with_for_update()
                )
            }
            if not current_uploads.issubset(set(candidate_uploads)):
                raise RuntimeError("Human-input upload links changed while cleanup owners were being locked")
            session.execute(delete(HumanInputFormUploadFile).where(HumanInputFormUploadFile.form_id.in_(form_ids)))
            session.execute(delete(HumanInputFormUploadToken).where(HumanInputFormUploadToken.form_id.in_(form_ids)))
            session.execute(delete(HumanInputFormRecipient).where(HumanInputFormRecipient.form_id.in_(form_ids)))
            session.execute(delete(HumanInputDelivery).where(HumanInputDelivery.form_id.in_(form_ids)))
            session.execute(delete(HumanInputForm).where(HumanInputForm.id.in_(form_ids)))

        session.execute(delete(Message).where(Message.conversation_id == conversation_id))
        session.execute(delete(PinnedConversation).where(PinnedConversation.conversation_id == conversation_id))
        session.execute(delete(AgentDebugConversation).where(AgentDebugConversation.conversation_id == conversation_id))
        session.execute(
            delete(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.is_deleted.is_(True),
            )
        )
        session.commit()
        return True


@shared_task(queue="conversation", bind=True, max_retries=_MAX_RETRIES, default_retry_delay=_RETRY_DELAY_SECONDS)
def delete_conversation_related_data(self, conversation_id: str) -> None:
    """
    Delete related data conversation in correct order from database to respect foreign key constraints

    Args:
        conversation_id: conversation Id
    """

    logger.info(
        click.style(f"Starting to delete conversation data from db for conversation_id {conversation_id}", fg="green")
    )
    start_at = time.perf_counter()

    try:
        cleaned = _cleanup_conversation_related_data(conversation_id)
    except Exception as exc:
        logger.exception("Failed to delete data for conversation_id: %s", conversation_id)
        countdown = min(_RETRY_DELAY_SECONDS * (2**self.request.retries), 10 * 60)
        raise self.retry(exc=exc, countdown=countdown) from exc

    end_at = time.perf_counter()
    logger.info(
        click.style(
            (
                f"Finished cleaning data for conversation_id {conversation_id}, "
                f"cleaned={cleaned}, latency: {end_at - start_at}"
            ),
            fg="green",
        )
    )


@shared_task(queue="conversation")
def sweep_deleted_conversations() -> int:
    """Re-enqueue soft-deleted conversations whose immediate dispatch was lost."""

    with session_factory.create_session() as session:
        conversation_ids = list(
            session.scalars(
                select(Conversation.id)
                .where(Conversation.is_deleted.is_(True))
                .order_by(Conversation.updated_at, Conversation.id)
                .limit(dify_config.CONVERSATION_CLEANUP_BATCH_SIZE)
            )
        )

    dispatched = 0
    for conversation_id in conversation_ids:
        try:
            delete_conversation_related_data.delay(conversation_id)
            dispatched += 1
        except Exception:
            logger.exception("Failed to dispatch cleanup for conversation %s", conversation_id)
    return dispatched
