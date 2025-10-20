import streamlit as st
from tax_qa_system import TaxQASystem
import time
import os

# 页面配置
st.set_page_config(
    page_title="税小助 - 增值税法问答",
    page_icon="💼",
    layout="wide"
)

# 自定义样式
st.markdown("""
<style>
    .chat-user {background: #e3f2fd; padding: 12px; border-radius: 8px; margin: 8px 0;}
    .chat-assistant {background: #e8f5e9; padding: 12px; border-radius: 8px; margin: 8px 0;}
    .source-expander {font-size: 0.9em; color: #666;}
</style>
""", unsafe_allow_html=True)

# 缓存初始化（避免重复加载模型）
@st.cache_resource(show_spinner="正在初始化税小助...（首次启动约10秒）")
def init_tax_system():
    """初始化税小助系统"""
    try:
        tax_system = TaxQASystem()
        return tax_system, None
    except Exception as e:
        return None, str(e)

# 主交互逻辑
def main():
    st.title("💼 税小助 - 《中华人民共和国增值税法》专属问答")
    st.caption("📄 基于官方文档（2024年12月通过，2026年1月1日施行）")

    # 初始化系统
    tax_system, init_error = init_tax_system()
    if init_error:
        st.error(f"初始化失败：{init_error}")
        st.info("请检查：1. 模型文件是否完整 2. 已安装依赖（pip install reportlab transformers langchain_community）")
        return

    # 会话状态：保存聊天历史
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 显示聊天历史
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">你：{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">税小助：{msg["content"]}</div>', unsafe_allow_html=True)
            if msg["sources"]:
                with st.expander("📌 参考来源", expanded=False):
                    st.markdown(f'<div class="source-expander">{msg["sources"][0]}</div>', unsafe_allow_html=True)

    # 清除历史按钮
    if st.button("🗑️ 清除聊天历史"):
        st.session_state.chat_history = []
        st.rerun()

    # 用户输入与回答
    user_input = st.chat_input("请输入问题（例：增值税法什么时候施行？）")
    if user_input:
        # 添加用户问题到历史
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.markdown(f'<div class="chat-user">你：{user_input}</div>', unsafe_allow_html=True)

        # 生成回答（关键：直接传字符串，不包装字典）
        with st.spinner("正在查询文档..."):
            start_time = time.time()
            # 正确传参：直接传user_input字符串（tax_qa_system会自动处理）
            response = tax_system.answer_tax_question(user_input)
            end_time = time.time()

        # 显示回答
        st.markdown(f'<div class="chat-assistant">税小助：{response["answer"]}</div>', unsafe_allow_html=True)
        if response["sources"]:
            with st.expander("📌 参考来源", expanded=False):
                st.markdown(f'<div class="source-expander">{response["sources"][0]}</div>', unsafe_allow_html=True)
        st.caption(f"查询耗时：{end_time - start_time:.2f}秒 | 基于官方增值税法文档")

        # 添加助手回答到历史
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"]
        })

        # 限制历史长度（避免内存占用）
        if len(st.session_state.chat_history) > 15:
            st.session_state.chat_history = st.session_state.chat_history[-15:]

if __name__ == "__main__":
    main()