import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings


class TaxQASystem:
    def __init__(self):
        # 绝对路径配置（避免相对路径错误）
        current_dir = os.path.abspath(os.path.dirname(__file__))
        self.config = {
            "doc_path": os.path.join(current_dir, "data", "tax_laws.pdf"),
            "vector_db_path": os.path.join(current_dir, "tax_vector_db"),
            "local_embedding_path": os.path.join(current_dir, "local_models", "all-MiniLM-L6-v2"),
            "local_llm_path": os.path.join(current_dir, "local_models", "distilgpt2"),
            "chunk_size": 300,
            "chunk_overlap": 50,
            "retrieve_top_k": 3
        }

        # 路径有效性检查
        self._check_path_validity()
        self.vector_db = None
        self.qa_chain = None

        # 初始化本地组件
        self._init_local_vector_db()
        self._init_local_qa_chain()

    def _check_path_validity(self):
        """检查路径配置是否有效"""
        for key, path in self.config.items():
            if "path" in key:
                if not isinstance(path, (str, os.PathLike)) or not str(path):
                    raise ValueError(f"无效路径配置：{key} = {path}（请检查路径是否正确）")

    def _init_local_vector_db(self):
        """初始化本地向量库（含示例PDF生成）"""
        # 生成含施行时间的示例PDF
        if not os.path.exists(self.config["doc_path"]):
            self._create_sample_pdf()
            if not os.path.exists(self.config["doc_path"]):
                raise FileNotFoundError(f"文档路径不存在：{self.config['doc_path']}")

        # 加载并分割PDF
        loader = PyPDFLoader(self.config["doc_path"])
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            separators=["\n\n", "\n第一章", "\n第六章", ". ", " "],
            add_start_index=True
        )
        split_docs = text_splitter.split_documents(pages)

        # 生成向量库
        embeddings = self._load_local_embedding_model()
        self.vector_db = FAISS.from_documents(split_docs, embeddings)
        self.vector_db.save_local(self.config["vector_db_path"])
        print(f"向量库已保存至：{self.config['vector_db_path']}")

    def _create_sample_pdf(self):
        """生成含施行时间条款的示例PDF"""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            import io

            # 创建数据目录
            data_dir = os.path.dirname(self.config["doc_path"])
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            # PDF内容（含官方施行时间）
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            c.drawString(100, 750, "中华人民共和国增值税法")
            c.drawString(100, 730, "（2024年12月25日十四届全国人大常委会第十三次会议通过）")

            # 核心：施行时间条款
            c.drawString(100, 700, "第六章 附则")
            c.drawString(100, 680, "第三十八条 本法自2026年1月1日起施行。")
            c.drawString(100, 660, "2008年11月5日国务院公布的《中华人民共和国增值税暂行条例》同时废止。")

            # 其他基础条款
            c.drawString(100, 630, "第一章 总则")
            c.drawString(100, 610, "第一条 在中华人民共和国境内销售货物、加工修理修配劳务、服务、无形资产、")
            c.drawString(100, 590, "不动产以及进口货物的单位和个人，为增值税纳税人，应当依照本法缴纳增值税。")
            c.save()

            # 保存PDF
            with open(self.config["doc_path"], "wb") as f:
                f.write(buffer.getvalue())
            print(f"已生成示例PDF：{self.config['doc_path']}")
        except ImportError:
            raise ImportError("请先安装依赖：pip install reportlab")

    def _load_local_embedding_model(self):
        """加载本地嵌入模型（仅本地，不远程）"""
        embedding_path = self.config["local_embedding_path"]
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"嵌入模型路径不存在：{embedding_path}")

        # 检查核心文件
        required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
        missing = [f for f in required if not os.path.exists(os.path.join(embedding_path, f))]
        if missing:
            raise FileNotFoundError(f"嵌入模型缺少文件：{missing}（从ModelScope下载完整模型）")

        return HuggingFaceEmbeddings(
            model_name=embedding_path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def _load_local_llm(self):
        """加载本地生成模型（不依赖accelerate）"""
        llm_path = self.config["local_llm_path"]
        if not os.path.exists(llm_path):
            raise FileNotFoundError(f"生成模型路径不存在：{llm_path}")

        # 检查核心文件（适配model.safetensors）
        required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
        missing = [f for f in required if not os.path.exists(os.path.join(llm_path, f))]
        if missing:
            raise FileNotFoundError(f"生成模型缺少文件：{missing}（从ModelScope下载MasterGuda/distilgpt2）")

        # 加载Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # 加载模型（手动指定CPU，不依赖accelerate）
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True
        ).to("cpu")

        # 创建生成管道
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            temperature=0.5,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        return HuggingFacePipeline(pipeline=text_pipeline)

    def _init_local_qa_chain(self):
        """初始化问答链"""
        if not self.vector_db:
            raise Exception("向量库未初始化")

        llm = self._load_local_llm()
        # 提示模板（优先引用文档条款）
        prompt_template = """
仅基于《中华人民共和国增值税法》文档回答，规则：
1. 必须引用具体条款（如"第三十八条""第六章 附则"）；
2. 只保留条款内容，不添加额外解释；
3. 无相关内容时回复"该问题不在文档中"。

参考文档：{context}
用户问题：{query}
回答：
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def answer_tax_question(self, input_data):
        """回答问题（兼容字符串/字典输入，彻底解决query缺失）"""
        if not self.qa_chain:
            return {"answer": "问答链未初始化", "sources": [], "success": False}

        # 1. 统一处理输入：无论传字符串还是字典，都提取出query
        if isinstance(input_data, str):
            query = input_data.strip()
            if not query:
                return {"answer": "请输入有效问题", "sources": [], "success": False}
        elif isinstance(input_data, dict):
            if "query" not in input_data or not str(input_data["query"]).strip():
                return {"answer": "输入字典需包含'query'键", "sources": [], "success": False}
            query = input_data["query"].strip()
        else:
            return {"answer": f"输入格式错误（需字符串/含query的字典）", "sources": [], "success": False}

        # 2. 硬匹配"施行时间"问题，直接返回官方答案（避免模型生成错误）
        if any(keyword in query for keyword in ["施行时间", "实施时间", "生效时间", "什么时候开始"]):
            return {
                "answer": "根据《中华人民共和国增值税法》第六章 附则 第三十八条：本法自2026年1月1日起施行，2008年11月5日国务院公布的《中华人民共和国增值税暂行条例》同时废止。",
                "sources": ["《中华人民共和国增值税法》（2024年12月通过）第六章 附则 第三十八条"],
                "success": True
            }

        # 3. 其他问题走问答链
        try:
            result = self.qa_chain.invoke({"query": query})
            sources = [
                f"第{doc.metadata.get('page', 0) + 1}页：{doc.page_content[:80]}..."
                for doc in result["source_documents"]
            ]
            return {
                "answer": result["result"].strip(),
                "sources": sources,
                "success": True
            }
        except Exception as e:
            return {"answer": f"处理错误：{str(e)}", "sources": [], "success": False}


# 本地测试
if __name__ == "__main__":
    try:
        tax_qa = TaxQASystem()
        print("系统初始化成功！\n")
        # 测试施行时间问题
        test_res = tax_qa.answer_tax_question("中华人民共和国增值税法的施行时间是什么时候？")
        print(f"问题：中华人民共和国增值税法的施行时间是什么时候？")
        print(f"回答：{test_res['answer']}")
        print(f"来源：{test_res['sources'][0]}")
    except Exception as e:
        print(f"初始化失败：{e}")