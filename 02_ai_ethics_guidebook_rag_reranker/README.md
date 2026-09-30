# 생성형 AI 윤리 가이드북 QA (Reranker)

> 사용한 책 내용: 심화편 CHAPTER 01 교차 인코더 리랭커 + 기본편의 PDF 로더·텍스트 분할·임베딩·FAISS·리트리버.

01 프로젝트에서 만든 **FAISS + Cross Encoder Reranker + 시스템 프롬프트** RAG 파이프라인을 새로운 문서인 `생성형_AI윤리_가이드북_RAG시험용_30쪽.pdf`에 적용했다. 질문은 코드에 직접 쓰지 않고, 학생별로 나눠 받은 질문 파일(`students/student_11_questions.json`)에서 불러온다.

| 항목 | 01 프로젝트 (사내 업무 가이드) | 02 프로젝트 (AI 윤리 가이드북) |
|---|---|---|
| 문서 | 사내 규정 PDF 9쪽 | 법·윤리 서술형 PDF 30쪽 |
| 질문 | 코드에 직접 작성한 10개 | JSON 파일에서 불러온 3개 (난이도 하·중·상) |
| 1차 검색 후보 수 | `k=7` | `k=8` |
| Reranker 최종 문서 수 | `top_n=3` | `top_n=4` |

> 코드: [`rag_reranker_answer_generation.ipynb`](rag_reranker_answer_generation.ipynb) · 파이프라인 자체의 설계 이유(유사도 threshold 대신 시스템 프롬프트로 "문서에 없음"을 판단하게 한 이유 등)는 [01 프로젝트 정리](../01_company_guide_rag_reranker/README.md)에 있다.

## 1. 질문을 파일에서 불러오기

질문 파일은 `question`(질문), `difficulty`(난이도), `reference`(참고 답안), `source_pages`(정답 근거 페이지) 등을 담고 있다. RAG에는 질문만 넣고, 나머지는 채점할 때 쓴다.

```python
# PDF를 자를 청크 사이즈와 오버랩 사이즈 결정
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# 학생 번호(내 번호는 11번)
STUDENT_NUMBER=11
```

```python
# 학생 번호 11번 질문 데이터 불러오기 
import json
with open(f"students/student_{STUDENT_NUMBER}_questions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# JSON에서 질문 리스트만 추출
query_list = [question.get("question") for question in data["questions"]]
```

## 2. 문서가 길어져서 바꾼 검색 설정

문서가 9쪽에서 30쪽으로 길어져서 1차 검색 후보를 `k=8`로, Reranker가 최종으로 고르는 문서를 `top_n=4`로 늘렸다. 나머지 파이프라인(PDF 로드 → `RecursiveCharacterTextSplitter` 500/100 → `text-embedding-3-small` → FAISS → Cross Encoder Reranker → 시스템 프롬프트 기반 답변 생성)은 01 프로젝트와 같다.

```python
# 리트리버 생성
retriever = db.as_retriever(search_kwargs={"k":8})
```

```python
from langchain_classic.retrievers import ContextualCompressionRetriever 
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker 

# 리랭커 생성
compressor = CrossEncoderReranker(model=model, top_n=4)

# 리트리버, 리랭커 합체
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)
```

## 3. 결과 확인

질문 파일에 들어 있는 참고 답안(`reference`)과 정답 근거 페이지(`source_pages`)를 기준으로 채점했다.

| # | 난이도 | 질문 요지 | 정답 근거 페이지 | Reranker 상위 4개 페이지 | 답변 | 판정 |
|---|---|---|---|---|---|---|
| 1 | 중 | 가짜 뉴스 배포 시 법적 책임 | p.16 | 16, 16, 17, 10 | 명예훼손 시 형사 처벌·민사 손해배상 (p.16) + 초상권 침해 등 추가 사례 (p.17) | 정확 |
| 2 | 하 | 콜롬비아 판사의 챗GPT 활용에 대한 교수의 입장 | p.14 | 14, 26, 27, 26 | "확실히 비윤리적이며 무책임한 행동", 같은 질문에 다른 답변을 받았다고 지적 (p.14) | 정확 |
| 3 | 상 | 개인정보·기밀 입력 시 보안 문제 | p.16, p.23 | 23, 20, 23, 14 | 서버 저장, 학습 재이용, 외부 유출 위험 (p.20, p.23) | 정확 |

- 세 질문 모두 Reranker가 고른 **1위 문서가 정답 근거 페이지**였고, 답변 내용도 참고 답안과 일치했다(3/3).
- 3번은 정답 근거가 p.16과 p.23 두 곳인데 p.16은 상위 4개에 들지 못했다. 그래도 p.23과 p.20에 같은 내용(서버 저장·학습 재이용·유출 위험)이 있어서 답변에는 문제가 없었다. 근거가 여러 페이지에 흩어진 질문에서는 `top_n`을 늘리거나 청크 크기를 조절해볼 여지가 있다.
- 1번 답변은 참고 답안보다 넓게 초상권 침해(p.17)까지 덧붙였는데, 이것도 문서에 있는 내용이라 환각(hallucination)은 아니다.

## 정리

- 같은 파이프라인을 **성격이 다른 문서**(사내 규정 → 법·윤리 가이드북)와 **3배 긴 문서**(9쪽 → 30쪽)에 적용해도 질문 3개 모두 정확하게 답했다.
- 질문을 JSON 파일에서 불러오게 바꿔서, `STUDENT_NUMBER`만 바꾸면 다른 질문 세트로 바로 다시 실험할 수 있다.
- 질문 파일에 참고 답안과 근거 페이지가 함께 있어서, 이번에는 PDF를 직접 읽지 않고도 **검색(근거 페이지를 찾았는가)과 답변(내용이 맞는가)을 따로 채점**할 수 있었다.
