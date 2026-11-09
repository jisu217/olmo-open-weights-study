# Paper Card: OLMo — Accelerating the Science of Language Models

## 논문 정보
- **제목**: OLMo: Accelerating the Science of Language Models
- **저자**: Groeneveld et al. (Allen Institute for AI)
- **연도**: ACL 2024 / arXiv:2402.00838
- **모델**: OLMo-1B, OLMo-7B

## 관련 링크
### 모델
- [OLMo-7B](https://huggingface.co/allenai/OLMo-7B)
- [OLMo-1B](https://huggingface.co/allenai/OLMo-1B)
- [OLMo-7B-SFT](https://huggingface.co/allenai/OLMo-7B-SFT)
- [OLMo-7B-Instruct (OLMo-7B+SFT+DPO)](https://huggingface.co/allenai/OLMo-7B-Instruct)

### 데이터 및 코드
- [Dolma 데이터셋](https://huggingface.co/datasets/allenai/dolma)
- [SFT 학습 코드 (Open-Instruct)](https://github.com/allenai/open-instruct)
- [TÜLU SFT 데이터](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture)

## 핵심 주장
- **한 줄 요약:** 기존 언어 모델의 폐쇄성으로 인한 과학적 연구의 한계를 해결하기 위해 전체 파이프라인을 100% 공개하는 개방형 언어 모델을 제안한다.
- **문제 정의:** 기존 언어 모델들이 닫혀 있어 성능의 강점/약점, 편향성, 위험성을 과학적으로 연구하기 어려움.
- **핵심 아이디어:** 안정성을 높인 아키텍처(편향 제거, SwiGLU, RoPE 등)와 3조 토큰 규모의 Dolma를 학습시키고, SFT와 DPO를 적용하여 인간의 지시를 따르는 성능을 극대화함

## 모델/알고리즘
###
### SFT
지도학습 기반 파인튜닝 단계. instruction과 output 쌍으로 된 데이터를 이용해 모델이 기본적인 응답을 할 수 있도록 훈련.
- 예시:
  ```json
  {"instruction": "육아휴직은 얼마나 쓸 수 있나요?", "output": "최대 1년까지 가능합니다. 회사 내규를 참고하세요."}

### DPO
직접 선호도 최적화 단계. 동일한 지시문에 대해 인간이 선호하는 '좋은 답변(chosen)'과 선호하지 않는 '나쁜 답변(rejected)' 쌍을 제공하여, 모델이 더 안전하고 품질 높은 답변을 하도록 정렬.
- 예시: 
  ```json
  {
    "instruction": "육아휴직은 얼마나 쓸 수 있나요?",
    "chosen": "남녀 근로자 모두 최대 1년까지 사용할 수 있습니다. 자세한 사항은 소속된 회사의 내규를 참고해 주시기 바랍니다.",
    "rejected": "1년요. 알아서 회사 인사팀에 물어보세요."
  }

## 실험
| 항목 | 내용 |
|---|---|
| PART 1 | 체크포인트 목록 출력 |
| PART 2 | Base→SFT→Instruct 패턴  |
| PART 3 | TÜLU LoRA SFT 학습 |
| PART 4 | 도메인SFT + DPO 5개 모델 비교 |

## 한계
- 영어 중심 (다른 언어 지원X)
- 편향성, 환각현상 존재
- 논문 원본 7B → 우리 실험 1B (스케일 차이)
- Full FT → LoRA 0.07% 경량화

## 프로젝트 활용
- OLMo가 가중치를 공개했기 때문에 발달장애 직장 도메인 데이터로 직접 파인튜닝 후 LLM직접 활용
