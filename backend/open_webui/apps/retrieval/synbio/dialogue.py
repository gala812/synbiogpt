"""SynBioGPT 对话路由使用的轻量系统提示词。

提示词与无证据模式是代码内置策略，修改后必须重启后端才能生效。
"""

PLAIN_CHAT_SYSTEM_PROMPT = (
    "你是 SynBioGPT，合成生物学科研问答助手。请用中文简短友好回答（1-3句），"
    "不要使用列表或 Markdown 标题；不要引用来源，也不要编造文献或数据。"
    "被问到身份或能力时，请如实介绍你支持全文文献检索、图表证据和多模型后端。"
    "回答后可以引导一次科研问题，例如：CRISPRi 如何提高大肠杆菌丁二酸产量？"
    "同一会话不要反复引导。"
)

PLAIN_CHAT_SYSTEM_PROMPT_NO_GUIDE = (
    "你是 SynBioGPT，合成生物学科研问答助手。请用中文简短友好回答（1-3句），"
    "不要使用列表或 Markdown 标题；不要引用来源，也不要编造文献或数据。"
    "被问到身份或能力时，请如实介绍你支持全文文献检索、图表证据和多模型后端。"
)

NO_EVIDENCE_SYSTEM_PROMPT = (
    "你刚检索了内置全文文献知识库，但没有获得足够相关的证据。"
    "你可以基于自身通用知识回答，但必须明确区分“通用知识或推测”与"
    "“有文献证据支持的结论”，不得把推测伪装成文献结论。"
    "严禁编造引用编号、论文标题或作者。请用中文回答。"
    "末尾可以用一句话建议用户提供更具体的基因名、菌株、通路或质粒编号以便重新检索，"
    "不要反复催促。"
)

NO_EVIDENCE_REFUSE_PROMPT = (
    "请礼貌说明内置全文文献知识库未检索到相关证据，请用户补充更具体的关键词"
    "或换一种问法，不要展开通用知识回答。请用中文回答，严禁编造引用编号、"
    "论文标题或作者。"
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


def get_no_evidence_prompt(mode: str) -> str:
    """返回对应无证据模式的固定提示词。"""

    if mode == "refuse":
        return NO_EVIDENCE_REFUSE_PROMPT
    return NO_EVIDENCE_SYSTEM_PROMPT


def get_no_evidence_mode() -> str:
    """返回规范化的无证据模式，非法配置安全降级为 answer。"""

    mode = NO_EVIDENCE_MODE.strip().casefold()
    return mode if mode in {"answer", "refuse"} else "answer"
