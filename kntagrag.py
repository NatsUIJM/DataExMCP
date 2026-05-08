import os
import asyncio
import pandas as pd
from typing import List
from openai import AsyncOpenAI
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务
mcp = FastMCP("tag_rag_server")

# 初始化异步 OpenAI 客户端
client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def find_project_root():
    """
    参照示例代码逻辑：从当前脚本路径向上查找，
    排除虚拟环境干扰，直到找到包含 tags.csv 的目录。
    """
    # 获取当前脚本的绝对路径
    current_path = os.path.dirname(os.path.abspath(__file__))

    # 排除虚拟环境目录的影响
    virtual_env_indicators = ['venv', '.venv', 'Scripts', 'bin']
    while any(part in virtual_env_indicators for part in current_path.split(os.sep)):
        new_path = os.path.dirname(current_path)
        if new_path == current_path:
            break
        current_path = new_path

    # 向上遍历查找 tags.csv
    search_path = current_path
    while search_path != os.path.dirname(search_path):
        if os.path.exists(os.path.join(search_path, "tags.csv")):
            return search_path
        search_path = os.path.dirname(search_path)
    
    return current_path

# 定位根目录及 tags.csv 路径
PROJECT_ROOT = find_project_root()
TAGS_CSV_PATH = os.path.join(PROJECT_ROOT, "tags.csv")

def read_document_content(file_path):
    """读取文档内容，支持相对路径转绝对路径"""
    # 如果是相对路径，则相对于 PROJECT_ROOT 拼接
    abs_path = file_path if os.path.isabs(file_path) else os.path.join(PROJECT_ROOT, file_path)
    try:
        with open(abs_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception:
        return ""

def chunk_list(lst, n):
    """将列表分成大小为 n 的块"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

async def process_group_task(group_docs, user_query):
    """异步处理单个文档组的信息提取"""
    doc_contents = []
    for doc_path in group_docs:
        content = read_document_content(doc_path)
        if content:
            doc_contents.append(f"Source: {doc_path}\nContent: {content[:99999]}")
    
    if not doc_contents:
        return ""

    docs_text = "\n\n---\n\n".join(doc_contents)
    prompt = f"""
    User Query: {user_query}
    
    Below are contents from a group of related documents:
    {docs_text}
    
    Please extract and summarize key information relevant to the user query from these documents.
    Maintain accuracy and cite the source paths.
    """
    
    try:
        completion = await client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return f"--- Analysis of a document group ---\n{completion.choices[0].message.content}"
    except Exception:
        return ""

@mcp.tool()
async def search_knowledge(tags: List[str], description: str) -> str:
    """
    根据标签检索文档并提取相关信息。
    
    Args:
        tags: 标签列表，例如 ["数学复习", "时间管理"]。
        description: 用户的原始查询描述。
    """
    # 检查 tags.csv 是否存在
    if not os.path.exists(TAGS_CSV_PATH):
        return (f"Error: tags.csv not found.\n"
                f"Target Path: {os.path.abspath(TAGS_CSV_PATH)}\n"
                f"Project Root: {PROJECT_ROOT}\n"
                f"Current Working Directory: {os.getcwd()}")

    # 1. 检索匹配的文档路径
    try:
        tags_df = pd.read_csv(TAGS_CSV_PATH)
    except Exception as e:
        return f"Error reading tags.csv: {str(e)}"

    all_documents = set()
    # 获取 CSV 中所有的唯一文档路径，作为备选
    fallback_documents = set(tags_df['文件路径'].unique().tolist())
    
    if not tags:
        # 如果没有提供标签，默认使用全部文档
        all_documents = fallback_documents
    else:
        # 尝试根据提供的标签进行匹配
        for tag in tags:
            mask = tags_df['标签'].str.contains(tag, na=False)
            matching_docs = tags_df.loc[mask, '文件路径'].tolist()
            all_documents.update(matching_docs)
        
        # 如果提供的标签在库中没有任何匹配项，则使用全部文档
        if not all_documents:
            all_documents = fallback_documents
    
    document_paths = list(all_documents)
    if not document_paths:
        return "No documents found in the database."

    # 2. 分组并并发处理 (每组 30 个文档)
    groups = list(chunk_list(document_paths, 30))
    
    tasks = [
        process_group_task(group, description) 
        for group in groups
    ]
    
    # 并发执行所有组的任务
    results = await asyncio.gather(*tasks)
    
    final_output = "\n\n".join([r for r in results if r])
    return final_output if final_output else "No relevant information extracted from documents."

if __name__ == "__main__":
    mcp.run(transport='stdio')
