"""Select the canonical SynBioGPT dialogue prompts for a request."""

from open_webui.apps.retrieval.prompts import (
    NO_EVIDENCE_REFUSE_PROMPT,
    NO_EVIDENCE_SYSTEM_PROMPT,
    PLAIN_CHAT_SYSTEM_PROMPT,
    PLAIN_CHAT_SYSTEM_PROMPT_NO_GUIDE,
    PRODUCT_CAPABILITY_SYSTEM_PROMPT,
)

NO_EVIDENCE_MODE = "answer"


def is_first_user_message(messages) -> bool:
    """返回当前会话是否只有一条用户消息。"""

    return (
        sum(
            isinstance(message, dict) and message.get("role") == "user"
            for message in (messages or [])
        )
        == 1
    )


def get_plain_chat_prompt(messages) -> str:
    """按会话轮次选择默认闲聊提示词。"""

    if is_first_user_message(messages):
        return PLAIN_CHAT_SYSTEM_PROMPT
    return PLAIN_CHAT_SYSTEM_PROMPT_NO_GUIDE


def get_product_capability_prompt() -> str:
    """返回系统知识库与检索能力的固定事实提示词。"""

    return PRODUCT_CAPABILITY_SYSTEM_PROMPT


def get_no_evidence_prompt(mode: str) -> str:
    """返回对应无证据模式的固定提示词。"""

    if mode == "refuse":
        return NO_EVIDENCE_REFUSE_PROMPT
    return NO_EVIDENCE_SYSTEM_PROMPT


def get_no_evidence_mode() -> str:
    """返回规范化的无证据模式，非法配置安全降级为 answer。"""

    mode = NO_EVIDENCE_MODE.strip().casefold()
    return mode if mode in {"answer", "refuse"} else "answer"
