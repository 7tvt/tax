import streamlit as st
from tax_qa_system import TaxQASystem
import time

# 页面配置：专属《中华人民共和国增值税法》
st.set_page_config(
    page_title="税小助 - 中华人民共和国增值税法助手",
    page_icon="💼",
    layout="wide"
)


# 本地缓存初始化（避免重复加载）
@st.cache_resource(show_spinner="正在初始化税小助（仅处理《中华人民共和国增值税法》）...")
def init_local_tax_helper():
    """仅初始化针对目标税法文档的助手"""
    try:
        tax_helper = TaxQASystem()
        return tax_helper, None
    except Exception as e:
        return None, str(e)


# 主交互界面
def main():
    st.title("💼 税小助 - 《中华人民共和国增值税法》专属助手")
    st.markdown("""
    📄 仅基于文档：《中华人民共和国增值税法》全文发布！2026年1月1日起施行.pdf  
    🔍 支持提问类型：
    - 施行时间、立法目的
    - 纳税人范围、应税交易定义
    - 各项目税率（货物/服务/不动产等）
    - 税收优惠、征收管理规则
    """)

    # 初始化本地助手
    tax_helper, init_error = init_local_tax_helper()

    # 处理初始化错误（仅提示目标文档相关问题）
    if init_error:
        st.error(f"税小助初始化失败：\n{init_error}")
        st.info("请检查：1. data文件夹中是否有'tax_laws.pdf'（即目标税法文档） 2. 文档是否损坏")
        return

    # 会话状态：保存聊天历史（仅目标文档相关对话）
    if "tax_chat_history" not in st.session_state:
        st.session_state.tax_chat_history = []

    # 显示聊天历史
    for msg in st.session_state.tax_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("查看《中华人民共和国增值税法》参考来源"):
                    for src in msg["sources"]:
                        st.write(src)

    # 用户输入处理（仅接受目标税法相关问题）
    user_question = st.chat_input("请输入关于《中华人民共和国增值税法》的问题（例：该法什么时候施行？）")
    if user_question:
        # 添加用户问题到历史
        st.session_state.tax_chat_history.append({
            "role": "user",
            "content": user_question
        })
        with st.chat_message("user"):
            st.markdown(user_question)

        # 生成基于目标文档的回答
        with st.chat_message("assistant"):
            with st.spinner("税小助正在查询《中华人民共和国增值税法》文档..."):
                start_time = time.time()
                response = tax_helper.answer_tax_question(user_question)
                end_time = time.time()

                # 显示回答
                st.markdown(response["answer"])
                st.caption(
                    f"查询耗时：{end_time - start_time:.2f}秒 | 基于《中华人民共和国增值税法》全文发布！2026年1月1日起施行.pdf")

                # 显示参考来源
                if response["sources"]:
                    with st.expander("查看《中华人民共和国增值税法》参考来源"):
                        for src in response["sources"]:
                            st.write(src)

        # 添加助手回答到历史
        st.session_state.tax_chat_history.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"]
        })


if __name__ == "__main__":
    main()