# RAG 비법노트 심화편 — 프로젝트 실습

[rag_two](https://github.com/TechieMoon/rag_two)(테디노트의 랭체인을 활용한 RAG 비법노트 심화편 복습 노트)에서 배운 내용을 적용한 프로젝트 실습 모음입니다.
심화편 기술 없이 기본편 내용만으로 만든 버전은 [rag_one_branch](https://github.com/TechieMoon/rag_one_branch)에 있습니다.

## 프로젝트 목록

| # | 프로젝트 | 내용 | 사용한 책 내용 |
|---|---|---|---|
| 01 | [Reranker 기반 사내 문서 QA](01_company_guide_rag_reranker/) | FAISS로 후보 7개를 검색하고 Cross Encoder Reranker(`BAAI/bge-reranker-v2-m3`)로 3개를 골라 답변 생성. 10개 질문을 PDF 원문과 대조해 10/10 정답 확인 | 심화편 CH01 교차 인코더 리랭커 + 기본편 CH09~13 |

## 실행 환경 (uv)

이 레포지토리는 [uv](https://docs.astral.sh/uv/)로 패키지를 관리합니다. `pyproject.toml`에 의존성이, `uv.lock`에 정확한 버전이 고정되어 있어서 어느 컴퓨터에서든 같은 환경을 만들 수 있습니다.

```powershell
# 1. uv 설치 (처음 한 번만, Windows PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 레포 루트에서 가상환경 생성 + 패키지 설치 (.venv 폴더가 생김)
uv sync

# 3. API 키 설정: .env.example을 복사해서 .env를 만들고 값 채우기
copy .env.example .env
```

- VS Code에서 노트북을 열고 커널로 `.venv`(Python 3.12)를 선택하면 됩니다.
- 패키지를 추가할 때는 `pip install` 대신 `uv add 패키지명`을 쓰면 `pyproject.toml`과 `uv.lock`이 함께 갱신됩니다.
- 스크립트는 `uv run python 파일명.py`로 실행합니다.
- 노트북은 각 프로젝트 폴더 안에서 실행하는 것을 기준으로 작성했습니다. (`data/` 상대 경로 사용, `.env`는 루트에 두면 자동으로 찾습니다)

> Cross Encoder 모델(`BAAI/bge-reranker-v2-m3`)은 처음 실행할 때 Hugging Face에서 자동으로 내려받습니다. `sentence-transformers`가 PyTorch를 함께 설치하므로 첫 `uv sync`는 시간이 조금 걸립니다.
