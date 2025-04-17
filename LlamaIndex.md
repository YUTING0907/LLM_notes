### 0 LlamaIndex 总述
LlamaIndex 是一个将大语言模型（Large Language Models, LLMs，后简称大模型）和外部数据连接在一起的工具。大模型依靠上下文学习（Context Learning）来推理知识，针对一个输入（或者是prompt），根据其输出结果。因此Prompt的质量很大程度上决定了输出结果的质量，因此提示工程（Prompt engineering）现在也很受欢迎。目前大模型的输入输出长度因模型结构、显卡算力等因素影响，都有一个长度限制（以Token为单位，ChatGPT限制长度为4k个，GPT-4是32k等，Claude最新版有个100k的）。当我们外部知识的内容超过这个长度时，就无法同时将有效的信息传递给大模型。因此就诞生了 LlamaIndex 等项目。

假设有一个10w的外部数据，我们的原始输入Prompt长度为100，长度限制为4k，通过查询-检索的方式，我们能将最有效的信息提取集中在这4k的长度中，与Prompt一起送给大模型，从而让大模型得到更多的信息。此外，还能通过多轮对话的方式不断提纯外部数据，达到在有限的输入长度限制下，传达更多的信息给大模型。这部分知识可参考：

如何让 ChatGPT(LLMs) 学习更多的私有数据知识？（一）
如何为 ChatGPT(LLMs) 学习更多的私有数据知识？（二）
LLamaIndex的任务是通过查询、检索的方式挖掘外部数据的信息，并将其传递给大模型，因此其主要由x部分组成：

数据连接。首先将数据能读取进来，这样才能挖掘。
索引构建。要查询外部数据，就必须先构建可以查询的索引，llamdaIndex将数据存储在Node中，并基于Node构建索引。索引类型包括向量索引、列表索引、树形索引等；
查询接口。有了索引，就必须提供查询索引的接口。通过这些接口用户可以与不同的 大模型进行对话，也能自定义需要的Prompt组合方式。查询接口会完成 检索+对话的功能，即先基于索引进行检索，再将检索结果和之前的输入Prompt进行（自定义）组合形成新的扩充Prompt，对话大模型并拿到结果进行解析。
### 1 数据连接器（Data Connectors）
数据连接器，读取文档的工具，最简单的就是读取本地文件。
LLamaIndex 的数据连接器包括

本地文件、Notion、Google 文档、Slack、Discord
具体可参考Data Connectors。

### 2 索引结构（Index Structures）
LlamaIndex 的核心其实就是 索引结构的集合，用户可以使用索引结构或基于这些索引结构自行建图。

2.1 索引如何工作
两个概念：

* Node（节点）：即一段文本（Chunk of Text），LlamaIndex读取文档（documents）对象，并将其解析/划分（parse/chunk）成 Node 节点对象，构建起索引。
* Response Synthesis（回复合成）：LlamaIndex 进行检索节点并响应回复合成，不同的模式有不同的响应模式（比如向量查询、树形查询就不同），合成不同的扩充Prompt。

索引方式包括

* List Index：Node顺序存储，可用关键字过滤Node
* Vector Store Index：每个Node一个向量，查询的时候取top-k相似
* Tree Index：树形Node，从树根向叶子查询，可单边查询，或者双边查询合并。
* Keyword Table Index：每个Node有很多个Keywords链接，通过查Keyword能查询对应Node。
不同的索引方式决定了Query选择Node方式的不同。

回复合成方式包括：

创建并提纯（Create and Refine)，即线性依次迭代；
树形总结（Tree Summarize）：自底向上，两两合并，最终合并成一个回复。
### 3 查询接口（Query Inference）
#### 3.1 LlamaIndex 使用模板
LlamaIndex 常用使用模版：

读取文档 (手动添加or通过Loader自动添加)；
将文档解析为Nodes；
构建索引（从文档or从Nodes，如果从文档，则对应函数内部会完成第2步的Node解析）
[可选，进阶] 在其他索引上构建索引，即多级索引结构
查询索引并对话大模型
#### 3.1.1 读取文档
使用data loaders读取
```
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(input_dir='./data').load_data()

documents = SimpleDirectoryReader(input_files=['./data/file.txt']).load_data()
```
或者直接把自己的text改为document文档
```
from llama_index import Document

text_list = [text1, text2, ...]
documents = [Document(t) for t in text_list]
```
文档是轻量化的数据源容器，可以将文档：

解析为 Node 对象 (见3.1.2)
直接喂入 Index (见3.1.3)，函数内部会完成转化Node过程
#### 3.1.2 解析文档为Node
Node以数据 Chunks 的形式呈现文档，同时 Node 保留与其他 Node 和 索引结构 的关系。

直接解析文档
```
from llama_index.node_parser import SimpleNodeParser

parser = SimpleNodeParser()

nodes = parser.get_nodes_from_documents(documents)
```
或者跳过 3.1.1 节文档创建操作，直接手动构建 Node
```
from llama_index.data_structs.node_v2 import Node, DocumentRelationship

node1 = Node(text="<text_chunk>", doc_id="<node_id>")
node2 = Node(text="<text_chunk>", doc_id="<node_id>")

node1.relationships[DocumentRelationship.NEXT] = node2.get_doc_id()
node2.relationships[DocumentRelationship.PREVIOUS] = node1.get_doc_id()
```
#### 3.1.3 Index 构建
可以直接将文档构建为 Index，这种简单构建的方式是在 Index 初始化时直接加载 文档

这种方式可以跳过 Node 构建（3.1.2）
```
from llama_index import GPTSimpleVectorIndex

index = GPTSimpleVectorIndex.from_documents(documents)
```
或者从 Node 构建 Index（3.1.2的续）
```
from llama_index import GPTSimpleVectorIndex

index = GPTSimpleVectorIndex(nodes)
```
多个索引（Index）结构复用 Node
当想在多个索引中，复用一个 Node 时，可以通过定义 DocumentStore 结构，并在添加Nodes时指定 DocumentStore
```
from gpt_index.docstore import SimpleDocumentStore

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

index1 = GPTSimpleVectorIndex(nodes, docstore=docstore)
index2 = GPTListIndex(nodes, docstore=docstore)
```
如果没指定 docstore，则会在创建 Index 时隐式创建一个。

索引中插入文档
也可以将文档插入到索引
```
from llama_index import GPTSimpleVectorIndex

index = GPTSimpleVectorIndex([])
for doc in documents:
    index.insert(doc)
自定义 LLMs
默认情况，llamaIndex 使用text-davinci-003，也可以用别的构建 Index

from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext
from langchain import OpenAI

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

max_input_size = 4096

num_output = 256

max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

index = GPTSimpleVectorIndex.from_documents(
    documents, service_context=service_context
)
```
自定义 Prompts
基于使用的Index，llamaIndex 会使用默认的 prompt 模板进行构建 Index（插入 or 创建）, 也可以自定义link。

自定义 Embeddings
对于自定义 embedding 的模型，也可以自定义 embedding link。

消费 Predictor
创建 Index、Insert 和 Query 时也会消耗 tokens，link。

存储 Index 下次用
```
import os.path as osp
index_file = "data/indices/index.json"
if not osp.isfile(index_file):
    # 判断是否存在，不存在则创建
    index = GPTSimpleVectorIndex.from_documents(documents)
    index.save_to_disk(index_file, encoding='utf-8')
else:
    # 存在则 load
    index = GPTSimpleVectorIndex.load_from_disk(index_file)
```
#### 3.1.4 [可选，进阶] 在索引上继续构建索引
可参考官方教程第4节；

#### 3.1.5 查询索引
默认使用索引为 问答形式，可以不指定额外的参数：
```
response = index.query("What did the author do growing up?")
print(response)

response = index.query("Write an email to the user given their background information.")
print(response)
```
也可以额外使用参数，取决于使用的索引类型，见link。

设置模式（mode）
通过加参数可以指定模型，以 ListIndex 为例，可选default默认格式和embedding嵌入特征模式。

如果是default，则是 创建并提纯（create and refine）的顺序迭代通过 Node；
如果是embedding，则根据 top-k 相似的 nodes 进行回复合成。
```
index = GPTListIndex.from_documents(documents)

response = index.query("What did the author do growing up?", mode="default")

response = index.query("What did the author do growing up?", mode="embedding")
```
具体可参考link。

设置回复模式（response_mode）
注意：此选项在GPTreeIndex中不可用/使用。

索引还可以通过response_mode具有以下响应模式：

default：对于给定的索引，“创建和完善”通过顺序浏览每个节点的答案；每个节点进行单独的LLM调用。有益于更详细的答案。
compact：对于给定的索引，通过填充可以适合最大提示大小的许多节点文本块来“紧凑”在每个LLM调用过程中的提示。如果有太多的块在一个提示中塞满了东西，请通过多个提示来“创建和完善”答案。
Tree_summarize：给定一组节点和查询，递归构造树并将根节点作为响应返回。有益于摘要目的。
```
index = GPTListIndex.from_documents(documents)

response = index.query("What did the author do growing up?", response_mode="default")

response = index.query("What did the author do growing up?", response_mode="compact")

response = index.query("What did the author do growing up?", response_mode="tree_summarize")
设置必需_keywords和dubl_keywords
可以在大多数索引上设置required_keywords和exclude_keywords（除了GPTTreeIndex）。
这能预先滤除不包含 required_keywords或 包含exclude_keywords 的节点，从而减少搜索空间，从而减少LLM调用/成本的时间/数量。

index.query(
    "What did the author do after Y Combinator?", required_keywords=["Combinator"], 
    exclude_keywords=["Italy"]
)
```
#### 3.1.5 解析回复
query的回复解析，包含回复的text和回复的 sources 来源
```
response = index.query("<query_str>")

str(response)

response.source_nodes

response.get_formatted_sources()
```
### 代码简述


```
import os
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from llama_index import (
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext
)
from langchain import OpenAI


documents = SimpleDirectoryReader('data').load_data()

os.environ['OPENAI_API_KEY'] = '设置自己的Key'

"""
直接这样可以运行
"""

index = GPTSimpleVectorIndex.from_documents(documents)
response = index.query("What did the author do growing up?")
print(response)
"""
Response(response='\n\nGrowing up, the author wrote short stories, programmed on an IBM 1401, wrote simple games and a word processor on a TRS-80, studied philosophy in college, learned Lisp, reverse-engineered SHRDLU, wrote a book about Lisp hacking, took art classes at Harvard, and painted still lives in his bedroom at night. He also attended an Accademia where he painted still lives on leftover scraps of ...
tover scraps of canvas, which was all I could afford at the time. Painting still lives is different', doc_id='5ba2ade0-0b8c-4ef7-906d-1ca434606232', embedding=None, doc_hash='6a5d8e0ae90c969305717b2ba8d4bc6296336ef595104d8d474abfff99ed64e3', extra_info=None, node_info={'start': 0, 'end': 15198}, relationships={<DocumentRelationship.SOURCE: '1'>: '5c89fa41-6bf8-4181-a9fa-57b66c3aecc1', <DocumentRelationship.NEXT: '3'>: 'a6444b3f-5fc5-4593-8b3d-f04ffb39a6a6'}), score=0.8242781513247005)], extra_info={'5ba2ade0-0b8c-4ef7-906d-1ca434606232': None})
"""


"""
自定义模型
"""
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

index = GPTKeywordTableIndex.from_documents(documents, service_context=service_context)

response = index.query("What did the author do growing up?")
print(response)

```
