# RAG 실습 (Reranker 기반 사내 문서 QA)

> 사용한 책 내용: 심화편 CHAPTER 01 리랭커(교차 인코더 리랭커) + 기본편의 PDF 로더·텍스트 분할·임베딩·FAISS·리트리버.
> Reranker를 뺀 기본편 버전은 [rag_one_branch](https://github.com/TechieMoon/rag_one_branch)에 있습니다.

가상의 사내 문서인 `가상Tech_업무가이드.pdf`를 대상으로, FAISS 벡터 검색 + Cross Encoder Reranker를 조합한 RAG(Retrieval-Augmented Generation) 파이프라인을 실습했다. [`rag_v1_reranker_search_only.ipynb`](rag_v1_reranker_search_only.ipynb)는 직접 작성한 첫 버전이고, [`rag_v2_reranker_answer_generation.ipynb`](rag_v2_reranker_answer_generation.ipynb)는 그 코드를 기반으로 클로드와 함께 보강한 버전이다. 두 버전을 비교하면서 무엇을, 왜 개선했는지를 정리한다.

## 1. 공통 준비: 청크 설정과 질문 세트

두 노트북 모두 같은 설정과 같은 10개 질문으로 검색 품질을 비교할 수 있게 했다.

```python
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

```python
query_list = [
    "연차를 사용하려면 최소 며칠 전에 신청해야 하나요?",
    "서울 출장 시 숙박비는 최대 얼마까지 지원되나요?",
    "출장을 다녀온 후 비용 정산은 언제까지 완료해야 하나요?",
    "회사 노트북을 분실했을 경우 어떤 절차로 신고해야 하나요?",
    "재택근무를 신청하려면 어떤 조건을 충족해야 하나요?",
    "직원이 업무 관련 교육을 받을 경우 회사에서 지원하는 비용 기준은 어떻게 되나요?",
    "출장 중 택시비를 회사 경비로 처리할 수 있는 경우는 언제인가요?",
    "개인 사정으로 출근 시간이 늦어질 경우 어떤 방식으로 보고해야 하나요?",
    "업무 중 시스템 장애가 발생했을 때 가장 먼저 해야 하는 조치는 무엇인가요?",
    "제주도로 2박 3일 출장을 가는 경우 숙박비 지원 한도와 출장비 정산 기한을 각각 알려주세요.",
]
```

10개 중 일부는 문서에 명확한 근거가 없는 질문을 일부러 섞어서, RAG가 "모르면 모른다고 답하는지"까지 검증할 수 있게 설계했다.

## 2. rag_v1_reranker_search_only.ipynb — 직접 작성한 첫 버전

PDF를 로드하고 `CharacterTextSplitter`(500자, 오버랩 100자)로 분할한 뒤 FAISS에 임베딩하고, 기본 retriever를 `CrossEncoderReranker`(`BAAI/bge-reranker-v2-m3`, top_n=3)로 한 번 더 재정렬해서 질문마다 상위 3개 문서를 출력했다.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader("data/가상Tech_업무가이드.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
split_docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(split_docs, embeddings)

retriever = db.as_retriever()
```

```python
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=model, top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)

for i, query in enumerate(query_list):
    compressed_docs = compression_retriever.invoke(query)
    print(f"\n질문{i+1}. {query}\n")
    pretty_print_docs(compressed_docs)
```

이 버전은 "검색"까지만 확인하는 실습이었다. 재정렬된 문서 3개를 눈으로 확인할 수는 있지만, 실제로 LLM이 답을 생성하지도 않고, 검색된 내용이 질문에 정말 답이 되는지 검증하는 과정도 없다.

## 3. rag_v2_reranker_answer_generation.ipynb — 클로드와 함께 보강한 버전

같은 구조를 기반으로 다음을 보강했다.

- `CharacterTextSplitter` → `RecursiveCharacterTextSplitter`로 교체. 구분자를 순서대로 시도하며 자르기 때문에 문장/문단 경계를 더 자연스럽게 지킨다.
- retriever가 1차로 가져오는 후보 수를 기본값(4개)에서 `k=7`로 늘렸다. Reranker가 최종 3개를 고르기 전에, 후보 풀을 더 넉넉하게 줘서 진짜 정답 문서가 애초에 후보에서 빠지는 걸 줄이기 위함이다.
- Cross Encoder 모델을 매번 새로 만들지 않고 한 번만 로드해서 재사용하도록 분리했다.
- 문서 출력 시 `page_label` metadata를 함께 표시해서, 나중에 답변에 근거 페이지를 붙일 수 있게 했다.
- 검색으로 끝내지 않고, 실제로 LLM이 답을 생성하는 체인을 추가했다.

```python
retriever = db.as_retriever(search_kwargs={"k": 7})
```

```python
# 허깅페이스 모델 한 번만 불러오기
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
```

### 답변 생성

검색된 문서를 그대로 붙여넣는 게 아니라, 시스템 프롬프트로 "문서에 있는 내용만으로 답하고, 없으면 모른다고 답하라"는 규칙을 명시한 뒤 LLM에게 답변을 생성시켰다.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    context = f"\n{'-' * 100}\n".join(
        [f"Document {i+1} (p.{d.metadata.get('page_label')}):\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    return context

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    1. 당신은 사용자의 질문에 대답을 하는 친절한 AI agent이다.
    2. 반드시 문서에 있는 내용만으로 답한다.
    3. 사용자의 질문을 문서에서 찾을 수 없으면 문서에서 찾지 못하여 대답을 할 수 없다고 답변한다.
    4. 질문이 여러 항목이면 항목마다 문서에 있는지 판단하고, 없는 항목은 '문서에 없음'이라고 밝힌다.
    5. 답변에 근거 페이지를 밝힌다.

    문서:
    {context}"""),
    ("human", "{question}"),
])

llm = ChatOpenAI(model="gpt-5.6-luna", temperature=0)
rag_chain = prompt | llm | StrOutputParser()

for i, query in enumerate(query_list):
    compressed_docs = compression_retriever.invoke(query)
    context = format_docs(compressed_docs)
    answer = rag_chain.invoke({"context": context, "question": query})
    print(f"질문 {i+1}: {query}\n\n답변: {answer}\n\n{'-'*80}\n\n")
```

## 4. 왜 유사도 임계값(threshold) 대신 시스템 프롬프트로 판단하게 했는가

처음에는 "관련성 점수가 일정 기준(threshold) 밑이면 답변을 거부한다"는 방식도 고려했지만, 실제로 검색 결과를 확인해보니 이 방식은 통하지 않았다. 예를 들어 질문 3("출장 비용 정산 기한")의 경우, reranker가 상위로 뽑은 문서들은 정산 기한을 직접 답하지는 못해도 "비용 처리" 챕터라는 점에서 주제상 관련이 높아 유사도 점수 자체는 높게 나온다. 즉 "관련은 있지만 실제 답은 아닌" 문서가 threshold로는 걸러지지 않는다.

그래서 점수 기준으로 자동 필터링하는 대신, 시스템 프롬프트에 "문서에 있는 내용만으로 답하고, 없으면 모른다고 답하라"는 규칙을 명시해서 LLM이 매번 스스로 판단하도록 위임했다.

## 5. 실행 결과로 확인한 것

10개 질문 중 실제로 문서에서 명확한 근거를 찾을 수 없었던 것은 3번(출장 비용 정산 기한), 4번(노트북 분실 신고 절차), 8번(지각 시 보고 방식)이었다. rag_02는 이 세 질문 모두에서 "문서에 명시되어 있지 않다"고 정확히 구분해서 답했고, 대신 관련은 있는 다른 정보(예: 법인카드 영수증 등록 기한, 보안사고 신고 연락처, 유연근무 출근 시간대)를 참고로 덧붙였다.

반대로 6번 질문("업무 관련 교육 비용 지원 기준")은 문서의 "성장 지원 제도(연 120만원, 교육·도서·컨퍼런스)" 항목에 그대로 나와 있어서, rag_02가 근거 페이지까지 명시하며 자신 있게 답변했다. 즉 이 실습에서 "문서에 답이 없는 질문"은 3·4·8번이고, 6번은 오히려 잘 답변되는 질문이었다.

```text
질문 3: 출장을 다녀온 후 비용 정산은 언제까지 완료해야 하나요?
답변: 문서에는 출장 후 비용 정산을 완료해야 하는 별도 기한은 명시되어 있지 않습니다.
다만 법인카드 사용 내역은 지출일로부터 5영업일 이내에 영수증과 사용 목적을 ExpenseHub에 등록해야 합니다.
근거: p.8

질문 4: 회사 노트북을 분실했을 경우 어떤 절차로 신고해야 하나요?
답변: 문서에는 회사 노트북 분실 시 신고 절차가 명시되어 있지 않아 정확한 절차를 답변할 수 없습니다.
다만 계정 탈취가 의심되면 즉시 네트워크를 끊고 #security-incident로 신고한다는 내용은 있습니다.

질문 6: 직원이 업무 관련 교육을 받을 경우 회사에서 지원하는 비용 기준은 어떻게 되나요?
답변: 업무 관련 교육은 성장 지원 제도를 통해 지원되며, 연 120만원 한도로 교육·도서·컨퍼런스 비용을
LearnHub에서 사전 승인받아 사용할 수 있습니다.
근거: Document 1, p.8

질문 8: 개인 사정으로 출근 시간이 늦어질 경우 어떤 방식으로 보고해야 하나요?
답변: 문서에는 개인 사정으로 출근이 늦어질 경우의 별도 보고 방식이 없습니다.
다만 유연근무 기준상 출근 시간은 08:00~11:00 사이입니다.
```

## 6. PDF 원문 대조 평가 (Ground Truth 검증)

rag_02가 생성한 10개 답변이 실제로 맞는지 확인하기 위해, `가상Tech_업무가이드.pdf` 9쪽 전체를 처음부터 끝까지 직접 읽고 질문마다 정답을 독립적으로 도출한 뒤 rag_02의 답변과 하나씩 대조했다.

| # | 질문 | PDF 원문 기준 정답 | rag_02 답변 | 판정 |
|---|---|---|---|---|
| 1 | 연차 신청 기한 | 1일 전(3일 이상은 2주 전 협의), 팀장 승인 — p.8 | 동일하게 답변 | 정확 |
| 2 | 서울 출장 숙박비 한도 | "서울" 전용 규정은 없고 국내 출장 공통으로 1박 15만원(세금 포함) — p.8 | 1박 15만원으로 답변(단, "서울 한정 규정이 아니라 국내 출장 공통"이라는 점은 언급하지 않음) | 정확(경미한 보완 여지) |
| 3 | 출장 후 비용 정산 기한 | 문서에 명시 없음. 가장 근접한 규정은 법인카드 5영업일 이내 등록 — p.8 | "명시된 기한 없음"이라 정확히 인정하고 같은 대체 규정을 근거로 제시 | 정확 |
| 4 | 노트북 분실 신고 절차 | 문서에 없음. 근접 규정은 계정 탈취 의심 시 신고(02-555-0142/#security-incident) — p.3 | "절차 없음"이라 인정하고 같은 대체 정보 제시 | 정확 |
| 5 | 재택근무 신청 조건 | 주 2회, 매주 월요일까지 캘린더에 WFH 표시, 이월 불가, 예외는 팀장+People Ops 승인 — p.2, p.9 | 동일 내용을 근거 페이지까지 포함해 답변 | 정확 |
| 6 | 업무 교육 비용 지원 기준 | 성장 지원 제도, 연 120만원, LearnHub 사전 승인 — p.8 | 동일하게 답변 | 정확 |
| 7 | 출장 중 택시비 처리 조건 | 23:00 이후 퇴근·대중교통 미운행·고객 일정 등 업무상 필요 시 실비 지원 — p.8 | 동일하게 답변 | 정확 |
| 8 | 지각 시 보고 방식 | 문서에 없음. 참고로 유연근무 08:00~11:00 출근 규정만 존재 — p.2 | "보고 방식 없음"이라 인정하고 같은 참고 정보 제시 | 정확 |
| 9 | 시스템 장애 최초 조치 | #help-urgent에 증상·시작시각·영향·관측 링크 게시 후 PagerDuty 호출 — p.7 | 동일하게 답변(첫 응답자가 IC를 맡는다는 후속 내용까지 포함) | 정확 |
| 10 | 제주 2박 3일 — 숙박비 한도/정산 기한 | 1박 15만원 한도이므로 2박은 단순 계산상 최대 30만원(문서에 "2박 한도"가 별도 명시된 건 아님). 정산 기한은 3번과 동일하게 명시 없음, 법인카드 5영업일 규정만 적용 | 동일하게 "2박 기준 최대 30만원"으로 계산하고, 정산 기한도 같은 대체 규정을 근거로 제시 | 정확(추론 방식도 동일) |

**총평**: 10문제 모두 PDF 원문과 일치했다(10/10). 특히 문서에 답이 없는 3·4·8번을 모두 정확히 "문서에 없음"으로 구분해내면서도 답변을 포기하지 않고 관련 규정을 참고로 덧붙인 점, 9번처럼 절차형 질문에서 채널명(#help-urgent)과 도구(PagerDuty)까지 정확히 인용한 점이 눈에 띈다. 2번과 10번에서는 문서에 없는 정보(서울 특례 여부, 2박 한도)를 "국내 출장 공통 규정"과 "1박 한도의 단순 배수"로 합리적으로 추론했는데, 이 추론 과정 자체를 답변에 명시하지 않은 점은 아쉬운 부분이다 — 실무에서는 "명시된 규정을 추정 적용했다"는 식의 단서를 붙이는 게 더 안전하다.

## 정리

| 항목 | rag_01 (직접 작성) | rag_02 (클로드와 함께 보강) |
|---|---|---|
| 텍스트 분할 | `CharacterTextSplitter` | `RecursiveCharacterTextSplitter` |
| 1차 검색 후보 수 | 기본값 (k=4) | `k=7`로 확대 |
| Cross Encoder 모델 | 매번 새로 생성 | 한 번만 로드해서 재사용 |
| 근거 페이지 표시 | 없음 | `page_label` metadata로 표시 |
| 답변 생성 | 없음 (재정렬된 문서만 출력) | 시스템 프롬프트 기반 RAG 체인으로 답변 생성 |
| 답 없는 질문 처리 | 검증 없음 | 시스템 프롬프트로 "문서에 없음" 판별 |
| 필터링 방식 | (해당 없음) | similarity threshold 대신 LLM이 매번 판단 — 관련은 있지만 오답인 문서도 점수가 높게 나와 threshold로는 못 거르기 때문 |

핵심은 "검색만 하는 파이프라인"에서 "검색 결과를 근거로 답을 생성하고, 근거가 없으면 모른다고 말할 줄 아는 파이프라인"으로 넘어간 것이다. Reranker로 관련도 높은 문서를 추려도 그 문서가 질문에 대한 진짜 답을 담고 있는지는 또 다른 문제이며, 이 판단을 점수 임계값이 아니라 LLM의 문맥 이해에 맡긴 것이 이번 실습의 핵심 설계 결정이었다.
