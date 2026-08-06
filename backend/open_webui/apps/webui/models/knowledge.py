import json
import logging
import time
from typing import Optional
import uuid

from open_webui.apps.webui.internal.db import Base, get_db
from open_webui.env import SRC_LOG_LEVELS

from open_webui.apps.webui.models.files import FileMetadataResponse
from open_webui.apps.webui.models.users import Users, UserResponse


from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Column, String, Text, JSON

from open_webui.utils.access_control import has_access

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

####################
# Knowledge DB Schema
####################


class Knowledge(Base):
    __tablename__ = "knowledge"

    id = Column(Text, unique=True, primary_key=True)
    user_id = Column(Text)

    name = Column(Text)
    description = Column(Text)

    data = Column(JSON, nullable=True)
    meta = Column(JSON, nullable=True)

    access_control = Column(JSON, nullable=True)  # Controls data access levels.


    created_at = Column(BigInteger)
    updated_at = Column(BigInteger)


class KnowledgeModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str

    name: str
    description: str

    data: Optional[dict] = None
    meta: Optional[dict] = None

    access_control: Optional[dict] = None

    created_at: int  # timestamp in epoch
    updated_at: int  # timestamp in epoch


####################
# Forms
####################


class KnowledgeUserModel(KnowledgeModel):
    user: Optional[UserResponse] = None


class KnowledgeResponse(KnowledgeModel):
    files: Optional[list[FileMetadataResponse | dict]] = None


class KnowledgeUserResponse(KnowledgeUserModel):
    files: Optional[list[FileMetadataResponse | dict]] = None


class KnowledgeForm(BaseModel):
    name: str
    description: str
    data: Optional[dict] = None
    access_control: Optional[dict] = None
    meta: Optional[dict] = None 


class KnowledgeTable:
    def insert_new_knowledge(self, user_id: str, form_data: KnowledgeForm) -> Optional[KnowledgeModel]:
        with get_db() as db:
            payload = form_data.model_dump(exclude_unset=True)

            # 规范化 meta
            meta = payload.get("meta") or {}
            if not isinstance(meta, dict):
                meta = {}

            tags = meta.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            if not isinstance(tags, list):
                tags = []
            tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]

            meta["tags"] = tags
            meta["tag_source"] = bool(meta.get("tag_source", False))

            payload["meta"] = meta

            knowledge = KnowledgeModel(
                **{
                    **payload,
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "created_at": int(time.time()),
                    "updated_at": int(time.time()),
                }
            )

            try:
                result = Knowledge(**knowledge.model_dump())
                db.add(result)
                db.commit()
                db.refresh(result)
                return KnowledgeModel.model_validate(result) if result else None
            except Exception as e:
                log.exception(e)
                return None


    def get_knowledge_bases(self) -> list[KnowledgeUserModel]:
        with get_db() as db:
            knowledge_bases = []
            for knowledge in (
                db.query(Knowledge).order_by(Knowledge.updated_at.desc()).all()
            ):
                user = Users.get_user_by_id(knowledge.user_id)
                knowledge_bases.append(
                    KnowledgeUserModel.model_validate(
                        {
                            **KnowledgeModel.model_validate(knowledge).model_dump(),
                            "user": user.model_dump() if user else None,
                        }
                    )
                )
            return knowledge_bases

    def get_knowledge_bases_by_user_id(
        self, user_id: str, permission: str = "write"
    ) -> list[KnowledgeUserModel]:
        knowledge_bases = self.get_knowledge_bases()
        return [
            knowledge_base
            for knowledge_base in knowledge_bases
            if knowledge_base.user_id == user_id
            or has_access(user_id, permission, knowledge_base.access_control)
        ]

    def get_knowledge_by_id(self, id: str) -> Optional[KnowledgeModel]:
        try:
            with get_db() as db:
                knowledge = db.query(Knowledge).filter_by(id=id).first()
                return KnowledgeModel.model_validate(knowledge) if knowledge else None
        except Exception:
            return None

    def update_knowledge_by_id(self, id: str, form_data: KnowledgeForm, overwrite: bool = False) -> Optional[KnowledgeModel]:
        try:
            with get_db() as db:
                # 只取用户实际传入的字段，避免覆盖
                payload = form_data.model_dump(exclude_unset=True)

                # 如果这次更新里包含 meta，才进行 meta 规范化
                if "meta" in payload:
                    meta = payload.get("meta")

                    if meta is None:
                        # 允许显式清空 meta：meta=null
                        payload["meta"] = None
                    else:
                        if not isinstance(meta, dict):
                            meta = {}

                        # tags 规范化
                        tags = meta.get("tags") or []
                        if isinstance(tags, str):
                            tags = [tags]
                        if not isinstance(tags, list):
                            tags = []
                        tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
                        meta["tags"] = tags

                        # 仅当 meta 里显式提供 tag_source 时才转换 bool，否则不动
                        if "tag_source" in meta:
                            meta["tag_source"] = bool(meta["tag_source"])

                        payload["meta"] = meta

                db.query(Knowledge).filter_by(id=id).update(
                    {
                        **payload,
                        "updated_at": int(time.time()),
                    }
                )
                db.commit()
                return self.get_knowledge_by_id(id=id)
        except Exception as e:
            log.exception(e)
            return None

    def update_knowledge_data_by_id(
        self, id: str, data: dict
    ) -> Optional[KnowledgeModel]:
        try:
            with get_db() as db:
                knowledge = self.get_knowledge_by_id(id=id)
                db.query(Knowledge).filter_by(id=id).update(
                    {
                        "data": data,
                        "updated_at": int(time.time()),
                    }
                )
                db.commit()
                return self.get_knowledge_by_id(id=id)
        except Exception as e:
            log.exception(e)
            return None

    def delete_knowledge_by_id(self, id: str) -> bool:
        try:
            with get_db() as db:
                db.query(Knowledge).filter_by(id=id).delete()
                db.commit()
                return True
        except Exception:
            return False

    def delete_all_knowledge(self) -> bool:
        with get_db() as db:
            try:
                db.query(Knowledge).delete()
                db.commit()

                return True
            except Exception:
                return False


Knowledges = KnowledgeTable()
