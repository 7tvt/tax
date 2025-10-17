import argparse
from tax_qa_system import TaxQASystem


def main():
    print("📜 税小助 - 智能税法助手")
    print("------------------------")
    print("输入 'exit' 退出程序")
    print("示例问题: 增值税的税率有哪些？")
    print("------------------------")

    # 初始化税小助
    try:
        print("税小助正在初始化...请稍候...")
        tax_helper = TaxQASystem()
        print("税小助已准备就绪！")
    except Exception as e:
        print(f"税小助初始化失败: {str(e)}")
        return

    # 交互循环
    while True:
        try:
            question = input("\n请输入您的问题: ")
            if question.lower() in ["exit", "quit"]:
                print("感谢使用税小助，再见！")
                break

            if not question.strip():
                continue

            print("税小助正在处理您的问题...")
            response = tax_helper.answer_tax_question(question)  # 修正方法名

            print("\n税小助的回答:")
            print(response["answer"])

            if response["sources"]:
                print("\n参考来源:")
                for i, src in enumerate(response["sources"], 1):
                    print(f"{i}. {src}")

        except KeyboardInterrupt:
            print("\n程序已终止")
            break
        except Exception as e:
            print(f"处理过程中出错: {str(e)}")


if __name__ == "__main__":
    main()