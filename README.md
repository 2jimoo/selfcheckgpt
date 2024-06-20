# Advanced experiments
SelfCheckGPT에 대한 추가 아이디어 실험입니다.

- /demo/experiments/advanced_experiments.ipynb
- demo/experiments/advanced_experiments_checkpoint
- 실험 이미지와 내용 요약에 대한 자료는 https://drive.google.com/file/d/10MnN6i49k-wkT9cIiU5bjWkDQNLFzOTO/view?usp=drive_link 애서 확인하실 수 있습니다.


# Table
- [README-KO](#README-KO)
- [README-EN](#README-EN)

------

# README-KO
### Abstract
본 연구에서 저희는 대형 언어 모델(LLM)의 텍스트 응답 일관성을 측정하여 할루시네이션을 탐지하는 방법을 연구하였습니다. 그 중 최근 소개된 SelfCheckGPT 방법론을 깊이 있게 탐구 하였고, 해당 방법론의 성능 강화를 위해 세 가지 개선된 방법론을 개발하여 적용하고 평가하였습니다. 우리가 적용한 방법론의 세가지 모델은 다음과 같습니다. 첫 번째는 SBERT Scoring 모델로 문장 임베딩 성능이 뛰어난 SBERT를 활용하여 기존 BERT 기반 평가 방식을 개선했습니다. 두번째는 HHEM(Hughes Hallucination Evaluation Model) Scoring 모델입니다. 이 모델은 기존 NLI 모델에 FEVER, Vitamin C, PAWS 데이터셋으로Fine Tuning을 수행하였으며, specific 한 데이터에 대해 학습 기능을 개선한 모델입니다. 세번째는 OpenIE Scoring 모델로 도메인 지식 모델링을 통해 비문맥적 정보를 활용하는 OpenIE(Open Information Extraction)를 적용합니다.
실험 결과, NLI Scoring 방식이 부정확한 텍스트 감지에서 가장 높은 성능을 보였으며, AUC 값이 91.60으로 다른 모델을 크게 상회했습니다. NLI 모델을 기반으로 한 HHEM의 경우 추가 데이터셋으로 Fine Tuning을 진행하였지만 빠른 추론과 학습 기능에 초점을 맞춰 할루시네이션 탐지 성능에 타협을 이룬 것으로 보입니다. NLI 모델은 부정확한 텍스트를 효과적으로 감지하며, 다양한 할루시네이션 유형을 감지하는 데 적합하다는 것이 입증되었습니다. 연구 결과 SelfCheckGPT의 할루시네이션 감지에서 NLI 모델이 중요한 역할을 할 수 있음을 시사하며, 이를 기반으로 한 추가적인 연구가 진행된다면, LLM의 할루시네이션 탐지 및 개선에 긍정적인 영향을 미칠 것으로 기대됩니다. 

### 1. Introduction

최근 생성형 AI Large Language Model(LLM)은 빠르게 발전하며 그 성능을 매일 같이 갱신하고 있습니다. 특히GPT(Generative Pre-trained Transformer)와 같은 모델은 텍스트, 이미지, 음성 등 다양한 형태의 콘텐츠 생성에 활용되고 있으며, 이외 다양한 오픈소스 LLM 들 또한 여러 산업분야의 소프트웨어에 적용하여 고객의 편의성 개선, 업무 효율성 향상을 위한 시도가 지속되는 상황입니다. 하지만 LLM에는 할루시네이션(Hallucination) 이라는 치명적인 문제가 존재하고, 다양한 분야에서 생성형 AI를 올바르게 도입하고 활용하기 위해서는 해당 문제를 개선하고 해결하는 것이 중요한 과제 중 하나입니다.
할루시네이션은 LLM이 일관성 없는 답변, 거짓 정보 또는 잘못된 정보로 응답하는 현상을 말하는데, 이는 엄격한 제어가 필요한 자율주행, 로봇, 의료분야와 같은 경우 치명적인 악영향을 야기할 수 있기 때문에 LLM의 할루시네이션을 완화는 매우 중요한 Task라고 할 수 있습니다.    
 할루시네이션 완화를 위해서는 LLM이 생성이 하는 Text에서 이를 탐지할 수 있는 알고리즘의 연구가 선행되어야합니다. 우리는 본 연구에서 LLM의 할루시네이션을 탐지하는 알고리즘들을 조사하였고, 기존의 방법론들과 차별화되며 높은 성능을 자랑하는 SelfCheckGPT 알고리즘에 대해 연구를 진행하였습니다.

기존의 할루시네이션 탐지 방법론은 텍스트에서의 토큰 출현 확률, 엔트로피와 같은 불확실성 메트릭을 활용하며, 공개가 제한된 데이터 시스템의 경우 활용이 어렵습니다. 이에 대한 대안은 생성된 텍스트에 사실 검증을 진행하는 것으로 외부 데이터에서 주장의 진실성을 평가하는 방법론이 있습니다. 그러나 이러한 방법론은 데이터베이스가 특정 도메인에 치중된 경우 정량적인 평가가 어려울 수 있고, 탐지하고자 하는 도메인의 데이터베이스가 부족하거나 없는 경우 이를 구축 해야하는 번거로움이 있습니다. 기존의 이러한 알고리즘들은 크게 화이트 박스, 그레이 박스, 블랙박스 접근법으로 분류할 수 있습니다. 화이트 박스 접근법은 신경망의 전 계층에 걸친 가중치를 취득할 수 있을 때 사용할 수 있는 방식으로 Hidden States를 활용하는 SAPLMA[2] 또는 LUNA[3]가 있습니다. 그리고 Attention Weights를 활용하는 방식으로 OPERA[4]가 있으며, Honesty Alignment를 활용하는 방법으로는 정직성 학습 프레임워크[5]들이 있습니다. 

 그레이 박스 접근법은 모든 계층의 정보는 알 수 없으나 최종 출력의 토큰 확률 분포를 얻을 수 있을 때 사용하는 방식으로 토큰 확률 분포의 불확실성을 기반으로 LVLM의 객체환각을 보정하는 KnowNo[6]과 플래닝 시 토큰 확률 불확정성 기준으로 수행할 다음 작업을 선택하는 HERACLEs[7] 등이 있습니다.

 블랙 박스 접근법은 입력 프롬프트와 출력 텍스트에만 의존하는 방식으로 Self-Consistency와 CoT를 결합하여 여러 지식 기반에서 증거를 검색하여 정확한 답변을 생성하는 CoK[8]가 있습니다. 본 연구에서 우리가 할루시네이션 탐지 방법론으로 최종 선정한 알고리즘인 SelfCheckGPT는 블랙박스 접근방식으로 동일 요청에 여러 응답 샘플간의 일관성을 비교하여 할루시네이션을 판단하게 됩니다.[9] SelfCheckGPT는 블랙박스 접근방법론에서 데이터의 외부 자원 없이 할루시네이션을 탐지하는 최초의 솔루션으로 데이터베이스가 제한된 다양한 영역에서 활용이 가능 합니다. SelfCheckGPT에서 제안하는 할루시네이션 Scoring Method 를 구현하고 이와 유사한 추가적인 Method를 연구하고 테스트하여 SelfCheckGPT 알고리즘을 통한 할루시네이션 탐지 전략을 제시합니다.

### 2. Related Work

블랙 박스 접근법에서 할루시네이션 감지에 사용하는 대표적인 메트릭으로 일관성 점수와 모순성 점수가 있습니다. 일관성 점수는 LLM의 응답을 여러 번 샘플링하여 응답 간 일관성을 비교합니다. 모델이 문장에서 환각을 적게 느낄수록 일관성은 높아집니다. 모순성 점수는 자연어 추론(Natural Language Inference, NLI)모델이 전제로 주어지는 텍스트와 가설로 주어지는 텍스트 사이의 관계를 함의(entailment), 모순(contradiction), 또는 중립(neutral)으로 분류합니다. 사실적 일관성은 모순이 없음을 의미하므로 모순성 점수 또한 환각 점수로써 유의미합니다.

### 2.1 Sentence-BERT

Sentence-BERT(SBERT)[10] BERT(Bidirectional Encoder Representations from Transformers)를 기반으로 한 문장 임베딩을 위한 모델로, Reimers와 Gurevych(2019)에 의해 제안되었다. BERT는 자연어 처리(NLP)의 다양한 작업에서 뛰어난 성능을 보여주었으나, 표준 BERT 모델을 사용한 문장 임베딩 추출은 큰 계산 비용과 비효율성을 야기합니다. 이를 해결하기 위해 SBERT는 시메즈(Siamese) 및 트리플렛(Triplet) 네트워크 구조를 활용하여 효율적인 문장 임베딩을 생성합니다. SBERT는 두 문장 간의 의미적 유사성을 효과적으로 측정할 수 있게 하며, 이는 대규모 데이터의 의미 유사성 비교, 클러스터링, 의미 검색과 같은 작업에 적합합니다. 기존 BERT 모델과 달리, SBERT는 문장을 독립적으로 처리하여 고정된 크기의 벡터로 변환한 후, 이 벡터들을 비교함으로써 훨씬 빠른 성능을 제공합니다. 이는 특히 대규모 데이터셋에서 효율적인 문장 비교를 가능하게 합니다.

SBERT는 자연어 추론(NLI) 데이터를 사용하여 미세 조정되며, 이를 통해 생성된 문장 임베딩은 InferSent, Universal Sentence Encoder와 같은 기존의 상태-아트 문장 임베딩 방법보다 우수한 성능을 보여주었습니다. 또한, SBERT는 다양한 의미적 텍스트 유사성(Semantic Textual Similarity, STS) 작업에서 높은 성능을 달성하였습니다. 특히 대규모 데이터셋에서의 응용에 있어서 기존 방법들과 비교하여 상당한 계산 효율성을 보여줍니다. 
2.2 HHEM(Hughes Hallucination Evaluation Model)

Hughes Hallucination Evaluation Model[11]은 생성된 텍스트가 원본 데이터에 충실한지 평가하는 데 사용되는 Vectara사의 상용 모델로 사실성 일관성 확률을 출력합니다. 0은 환각이고 1은 사실과 일치함을 나타냅니다. 이 모델은 사전 학습된 microsoft의 deberta-v3-base에 대해 FEVER, Vitamin C, PAWS 데이터셋으로 추가적인 fine-tuning을 수행하였고, 추론 속도를 개선하였습니다. 그리고 domain specific한 데이터에 활용하기 위해 추가 데이터의 학습 기능을 개선하였습니다.

### 2.3 Open Information Extraction

Open Information Extraction (OpenIE)은 자연어 텍스트로부터 구조화된 정보를 자동으로 추출하는 연구 분야입니다. 이 기술은 텍스트 내의 개체와 그들 사이의 관계를 식별하고 이를 트리플(주체, 관계, 객체) 형태로 변환하여 정보를 구조화하는 데 중점을 둡니다. OpenIE는 다양한 NLP 작업에서 중요한 기초 기술로 활용되며, 지식 그래프 생성, 요약, 질문 응답 시스템 및 정보 검색 등에 응용됩니다. 초기 OpenIE 시스템은 주로 패턴 또는 통계 기반 접근 방식을 사용했습니다. 예를 들어, Stanford OpenIE[12]는 의존성 구문 트리를 반복적으로 순회하면서 문장을 구문 및 의미론적으로 독립적인 절 단위로 나눕니다. KBP 데이터셋에 대하여 동사에 대해 관계 추출 완성도를 훈련시킨 다항로지스틱분류기를 통해 각 절이 핵심 내용을 유지하면서 최대로 단축하도록 병합합니다. 이 논문에서는 n-gram과 하드코딩을 통해 암시적으로 의미론적 내용을 학습했습니다. 이 과정으로 간결화된 문장들에서 트리플을 생성합니다.

최근의 OpenIE 연구는 신경망 모델을 채택하고 있습니다. SpanOIE[13]은 BiLSTM을 Encoder로 하여 토큰 레벨이 아닌 span 레벨로 tagging task를 지도기반 학습을 수행합니다. 이후 BiLSTM의 출력벡터를 LSTM 모델에 넣어 predicate 및 argument 구분자를 포함한 sequence를 생성하는 Seq2Seq 프레임워크가 제안되었고[14], IMoJIE[15]는 이를 개선하여 Seq2Seq모델과 BERT를 결합하였습니다.

### 3. Experiment

우리는 기존 SelfCheckGPT의 성능 향상을 목적으로 앞서 소개한 세 가지 모델을 SelfCheckGPT에 적용하였고, 성능을 평가하였습니다. 저희가 연구한 세 가지 모델에서 각각의 평가 알고리즘과 수식은 다음과 같습니다.

3.1 SelfcheckGPT with SBERT

기존 논문에서 제안한 BERTScore는 각 문장 S=\{s_1,s_2,…,s_m \}에 대해 샘플링된 문장들 P=\{p_1,p_2,…,p_n \}과 BERTScore의 평균 F1을 사용하여 아래와 같은 수식으로 일관성 평가를 수행합니다.
bertscore mean per 〖sent〗_i=1/N ∑_(j=1)^n▒〖F1〗_(i,j) 

 위 수식에서 〖F1〗_(i,j)은 문장 s_i, p_j 사이의 F1 score를 의미합니다. SBERT는 기존의 BERT를 활용하여 문장 임베딩의 코사인 유사도를 구하여 일관성을 평가하기 때문에 문장 단위 임베딩에서 의미적 유사도를 더 잘 나타낼 수 있습니다. SBERT score의 수식은 아래와 같습니다.

〖"score" 〗_i=1-(1/N ∑_(j=1)^N▒max┬k⁡(cos⁡(e_i,e_jk ) ) )

위 수식에서 N은 샘플링 문장의 전체 수이며, 문장 S=\{s_1,s_2,…,s_m \}과 샘플링된 문장 P=\{p_1,p_2,…,p_n \} 이 주어질 때, e_i 는 문장 s_i 의 임베딩 결과이고, e_jk 는 샘플링 문장 p_j 에서 k 번째 문장의 임베딩 결과입니다. 이 후 각 임베딩 결과를 통해 cos⁡(e_i,e_jk ) 를 산출합니다.

3.2 SelfcheckGPT with HHEM

기존 논문에서 제안하는 NLI Score는 문장과 샘플링된 통과 문장 사이의 평균 모순 확률을 사용하여 일관성을 평가한다. 이를 표현하는 수식은 다음과 같습니다. 

█(〖"scores per sentence" 〗_i=1/N ∑_(j=1)^N▒〖"prob" 〗_(i,j) )

위 수식에서 〖"prob" 〗_(i,j) 은 문장 s_i 와 샘플링 문장 p_j 사이의 모순 확률을 의미합니다. 
본 연구에서 저희는 기존 NLI 모델과 동일한 DeBERTa-v3모델에FEVER, Vitamin C, PAWS 데이터셋으로 Fine Tuning 수행 후 도메인 데이터의 학습기능과 추론 속도를 개선한HHEM 모델을 연구하였습니다. HHEM Score는 문장이 사실에 가까울수록 1, 환각에 가까울 수록 0을 출력하며, 이를 표현하는 수식은 다음과 같습니다.
    
█(〖"score" 〗_i=1-(1/N ∑_(j=1)^N▒"similarity" (s_i,p_j ) ) )

N은 전체 문장의 개수이고, "similarity" (s_i,p_j ) 는 문장 s_i 와 샘플링 문장 p_j 사이의 유사도를 의미합니다. 이는 CrossEncoder model을 통해 예측됩니다.
3.3 SelfcheckGPT with OpenIE

기존 논문에서 제안하는 N-gram score는 각 문장과 샘플링된 문장들과의 N-gram 유사도를 사용하여 일관성을 정의합니다. 기본적으로 N=1로 수행되며, 수식은 아래와 같습니다.
█("cosine similarity" (s,p)=(v_s⋅v_p)/(\|v_s \|\|v_p \|)#(1) )

█("average " 〖"similarity" 〗_s=1/n ∑_(i=1)^n▒〖"cosine " 〖"similarity" 〗_i 〗#(2) )

█(〖"final score" 〗_s=1-"average " 〖"similarity" 〗_s#(3) )


위 수식(1)에서 v_s 와 v_p 는 각각 sentences(S) 와 sampled passages(P)의 embedding vector이며 이를 통해 문장간의 cosine simlilarity를 산출합니다. 
기존 실험결과 BERT Score보다 N-gram, 특히 Unigram score가 성능이 더 좋았던 것에서 착안하여, 본 논문에서는 문맥적 정보를 의도적으로 제외하고 도메인 지식의 모델링에 집중하는 아이디어를 제안합니다. 도메인 지식 모델링에는 OpenIE를 사용하였으며, OpenIE는 통계적 접근법과 신경망적 접근법이 있는데 신경망 접근법은 내부에서 BERT계열 모델을 활용하고 있기 때문에 통계적 접근법 중 n-gram을 내부적으로 사용하는 Stanford OpenIE를 사용하여 triple들을 추출합니다. 추출된 triple들의 각 원소를 사전훈련된 단어 임베딩 모델을 사용하여 벡터로 변환하였고, 벡터로 표현된 triple간 유사도를 평가합니다. 실험에서는 메모리의 제약으로 Glove나 FastText와 같은 전통적 단어 임베딩 모델 대신 BERT를 사용하였습니다. OpenIE score수식은 아래와 같습니다.

█("similarity" (t_s,t_p )=(v_(t_s )⋅v_(t_p ))/(\|v_(t_s ) \|\|v_(t_p ) \|)#(1) )

█(〖"score" 〗_s=1/|P|  ∑_(p∈P)▒〖max┬(t_s∈T_s,t_p∈T_p )⁡"similarity"  (t_s,t_p ) 〗#(2) )

█(〖"final score" 〗_s=1-〖"score" 〗_s#(3) )

위 수식(1)은 sentences (S)와 sampled passages (P)에서 추출된 triple 간의 similarity를 구하는 수식으로 각각의 v_ts 와 v_tp 는 S, P 원소의 embedding vector를 의미합니다. Score를 산출하는 수식 또한 기존의 N-gram과 다르게 위 (2)수식과 같이 sentences(S)와 sampled passages(P)의 triples의 similarity에서 최댓값을 활용합니다.  

| Datasets/Scoring Method      | Bert Scores AUC | SBert Scores AUC | NIL Scores AUC | OpenIE Score AUC | Ngram Scores AUC | HHEM Scores AUC |
|------------------------------|-----------------|------------------|----------------|------------------|------------------|-----------------|
| Detect False (count 1843)    | 80.49           | 79.28            | 91.60          | 77.14            | 73.83            | 85.50           |
| Detect False* (count 1577)   | 42.53           | 40.27            | 41.98          | 35.21            | 29.46            | 33.79           |
| Detect True (count 1843)     | 21.31           | 22.06            | 16.61          | 25.21            | 26.29            | 18.83           |




앞서 소개한 Fig1,2 와 이를 통해 산출한 Table 1과 같은 지표 값을 산출하였습니다. 실험을 통해 분석한 결과 NLI 모델은 부정확한 텍스트 탐지 테스트에AUC 값이 91.60으로, 다른 모델보다 높은 성능을 나타냈습니다. 회귀선 분석에서 기울기가 0.46, 절편이 0.50으로, Human Score 와 비례성이 뛰어났습니다.

### 4. Conclusion

본 연구는 LLM의 할루시네이션 탐지에 효과적인 방법론으로 연구된 SelfCheckGPT의 성능을 향상시키기 위해 다양한 할루시네이션 탐지 모델을 평가하고 분석하였습니다. 특히, NLI (Natural Language Inference) 모델이 부정확한 텍스트 감지에서 가장 우수한 성능을 보였습니다. 진행한 실험을 통해 분석한 결과 NLI 모델은 다음과 같은 이유로 가장 효율적이고 신뢰할 수 있는 모델로 평가됩니다. NLI는 다양한 할루시네이션 유형 감지에 높은 성능을 나타내며, 다양한 도메인의 데이터에서 상대적으로 높은 신뢰성을 제공합니다. NLI 모델을 기반으로 한 HHEM의 경우 추가 데이터셋으로 Fine Tuning을 진행하였지만 빠른 추론과 학습 기능에 집중하여 일부 성능을 희생한 것으로 보입니다. 그러므로 상용 모델 검증 및 상용 적용 시 NLI 사용을 권장합니다. 그리고 N-Gram의 성능 또한 좋은 편에 속하였는데, 이는 토큰 분포를 모사하여 높은 성능을 나타낸 것으로 추정됩니다. 그러므로 할루시네이션 탐지에서 토큰 분포 불확실성과 텍스트 일관성 비교가 모두 수행된다면 더욱 높은 탐지 성능을 발휘할 수 있을 것입니다.  

본 연구를 통해 기존 논문에서 제안한 일관성 측정 방법을 강화하고 새로운 아이디어를 도입하여 실용성과 성능을 모두 향상시킬 수 있었고, 다양한 방법론을 통합하여 LLM의 할루시네이션을 더욱 효과적으로 탐지할 수 있는 새로운 접근 방식을 제안하였습니다. 추가적인 연구에서는 더욱 다양한 데이터셋과 환경에서 제안된 방법론을 검증하고, 실시간 적용 가능성을 높이기 위한 최적화와 개선이 필요할 것으로 보입니다. 하지만 모든 Scoring Method는 기본적으로 외부 데이터베이스 없이 샘플링 기반으로 LLM의 할루시네이션을 탐지하기 때문에 기존의 방법론들 대비 활용성의 자유도가 높다고 볼 수 있습니다. 이러한 이유로 저희가 연구한 할루시네이션 탐지 방법론은Active Learning 과 Hard Negative Mining Task에서 또한 유용하게 활용할 수 있다고 판단되며 관련 연구가 이어진다면 LLM의 할루시네이션 개선과 성능향상에도 많은 기여를 할 수 있을 것으로 보입니다.



### 5. Reference

[1] Chakraborty, N., Ornik, M., & Driggs-Campbell, K. (2024). Hallucination Detection in Foundation Models for Decision-Making: A Flexible Definition and Review of the State of the Art. arXiv preprint arXiv:2403.16527.
[2] Azaria, A., & Mitchell, T. (2023). The internal state of an llm knows when its lying. arXiv preprint arXiv:2304.13734.
[3] Song, D., Xie, X., Song, J., Zhu, D., Huang, Y., Juefei-Xu, F., & Ma, L. (2023). LUNA: A Model-Based Universal Analysis Framework for Large Language Models. arXiv preprint arXiv:2310.14211.
[4] Huang, Q., Dong, X., Zhang, P., Wang, B., He, C., Wang, J., ... & Yu, N. (2023). Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. arXiv preprint arXiv:2311.17911.
[5] Yang, Y., Chern, E., Qiu, X., Neubig, G., & Liu, P. (2023). Alignment for honesty. arXiv preprint arXiv:2312.07000.
[6] Liang, K., Zhang, Z., & Fisac, J. F. (2024). Introspective Planning: Guiding Language-Enabled Agents to Refine Their Own Uncertainty. arXiv preprint arXiv:2402.06529.
[7] Wang, J., Tong, J., Tan, K., Vorobeychik, Y., & Kantaros, Y. (2023). Conformal temporal logic planning using large language models: Knowing when to do what and when to ask for help. arXiv preprint arXiv:2309.10092.
[8] Li, X., Zhao, R., Chia, Y. K., Ding, B., Bing, L., Joty, S., & Poria, S. (2023). Chain of knowledge: A framework for grounding large language models with structured knowledge bases. arXiv preprint arXiv:2305.13269.
[9] Manakul, P., Liusie, A., & Gales, M. J. (2023). Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. arXiv preprint arXiv:2303.08896.
[10] Reimers, N., & Gurevych, I. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.
[11] Hong, G., Gema, A. P., Saxena, R., Du, X., Nie, P., Zhao, Y., ... & Minervini, P. (2024). The Hallucinations Leaderboard--An Open Effort to Measure Hallucinations in Large Language Models. arXiv preprint arXiv:2404.05904.
[12] Angeli, G., Premkumar, M. J. J., & Manning, C. D. (2015, July). Leveraging linguistic structure for open domain information extraction. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 344-354).
[13] Zhan, J., & Zhao, H. (2020, April). Span model for open information extraction on accurate corpus. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 05, pp. 9523-9530).
[14] Cui, L., Wei, F., & Zhou, M. (2018). Neural open information extraction. arXiv preprint arXiv:1805.04270.
[15] Kolluru, K., Aggarwal, S., Rathore, V., & Chakrabarti, S. (2020). Imojie: Iterative memory-based joint open information extraction. arXiv preprint arXiv:2005.08178.

# README-EN

### SelfcheckGPT with SBERT
In previous research, BERTScore was used to assess consistency by calculating the average F1 BERTScore between each sentence and sampled passages, as depicted in fig(1). This paper proposes replacing BERT with SBERT, which provides faster performance by utilizing sentence-level embeddings to better capture semantic similarities. The SBERT score formula remains unchanged from the one used in BERTScore.

### SelfcheckGPT with HHEM
The existing approach uses NLI Score to evaluate consistency by calculating the average probability of contradiction between a sentence and sampled passages, as shown in fig(2). This paper suggests replacing the existing DeBERTa-v3 model with HHEM, a model trained on additional NLI datasets. The HHEM score formula is consistent with the existing metric, maintaining its approach to evaluating consistency.

### SelfcheckGPT with OpenIE
Building on findings that N-gram, particularly Unigram score, outperformed BERT Score in previous experiments, this paper proposes a method that focuses on domain knowledge modeling using OpenIE. This approach deliberately excludes contextual information and utilizes n-grams within Stanford OpenIE for generating triples. The extracted triples are then vectorized using pre-trained word embedding models, and the similarity between these vectorized triples is assessed. Due to memory constraints, BERT was used instead of traditional word embedding models like Glove or FastText in experiments. The OpenIE score formula can be represented as shown in fig(6).

### Figures
- **fig(1)**: Demonstrates the evaluation of consistency using BERTScore.
- **fig(2)**: Shows the method for calculating consistency using the average probability of contradiction with the NLI Score.
- **fig(5)**: Illustrates the process of evaluating consistency using N-gram score, specifically focusing on Unigram.
- **fig(6)**: Describes the OpenIE score formula used to assess the similarity between extracted triples.


# Background and Related Works

## Hallucination Detection
Hallucination in Language Models (LMs) refers to the phenomenon where LMs produce inconsistent answers, false information, or incorrect information. This can have critical implications in areas requiring stringent control such as autonomous driving, robotics, and healthcare. To mitigate hallucination, research into detection algorithms is essential. Hallucination detection algorithms can be broadly categorized into white-box, gray-box, and black-box approaches.

### White-Box Approach
The white-box approach is applicable when it is possible to obtain weights across all layers of a neural network. Techniques such as SAPLMA and LUNA utilize Hidden States, and OPERA uses Attention Weights. Methods employing Honesty Alignment include various honesty training frameworks.

### Gray-Box Approach
The gray-box approach is used when it is not possible to know all layer information but it is possible to obtain the final output's token probability distribution. Techniques like KnowNo adjust object hallucination based on the uncertainty of the token probability distribution, and HERACLEs choose the next action based on token probability uncertainty during planning.

### Black-Box Approach
The black-box approach relies solely on the input prompts and output text. Techniques like CoK combine Self-Consistency and Chains of Thought (CoT) to search for evidence across various knowledge bases to generate accurate answers. SelfCheckGPT compares consistency among multiple response samples to the same request to detect hallucinations.

In the black-box approach, typical metrics for detecting hallucinations include consistency scores and contradiction scores. Consistency scores compare the consistency among multiple responses sampled from the LM—the less the model hallucinates, the higher the consistency. Contradiction scores use Natural Language Inference (NLI) models to classify the relationship between the given text (premise) and the hypothesized text into entailment, contradiction, or neutrality. Factual consistency implies the absence of contradiction, making contradiction scores a meaningful metric for hallucination.

## Sentence-BERT (SBERT)
SBERT, proposed by Reimers and Gurevych in 2019, is a model for sentence embedding based on BERT. While BERT has shown excellent performance in various NLP tasks, using the standard BERT model for sentence embedding extraction incurs high computational costs and inefficiencies. To address this, SBERT utilizes Siamese and Triplet network architectures to produce efficient sentence embeddings.

SBERT effectively measures semantic similarity between two sentences, making it suitable for large-scale semantic similarity comparison, clustering, and semantic search. Unlike traditional BERT models, SBERT processes sentences independently, converting them into fixed-size vectors and comparing these vectors to provide much faster performance, especially in large datasets.

SBERT is fine-tuned using Natural Language Inference (NLI) data, and the resulting sentence embeddings have shown superior performance compared to existing state-of-the-art sentence embedding methods like InferSent and Universal Sentence Encoder. SBERT also achieves high performance in various Semantic Textual Similarity (STS) tasks.

## Hughes Hallucination Evaluation Model
The Hughes Hallucination Evaluation Model is a commercial model used to evaluate how faithful generated text is to the original data, producing a probability score between 0 and 1. A score of 0 indicates hallucination, and 1 indicates factual accuracy. The model, based on Microsoft's DeBERTa-v3-base, has been trained on additional NLI datasets after training on SNLI and Multi-NLI datasets and then on summary tasks using FEVER, Vitamin C, and PAWS datasets to enhance NLI task performance.

## Open Information Extraction (OpenIE)
OpenIE is a research field that automatically extracts structured information from natural language text. This technology focuses on identifying entities within the text and their relationships, converting this information into triples (subject, relation, object) to structure the information. OpenIE is a fundamental technology used in various NLP tasks, including knowledge graph creation, summarization, question-answering systems, and information retrieval.

Initial OpenIE systems primarily used pattern-based or statistical approaches. For instance, Stanford OpenIE traverses dependency syntax trees iteratively, dividing sentences into semantically and syntactically independent clauses. These clauses are then merged to maximize content retention while minimizing length, using a multinomial logistic classifier trained on verb relation extraction completeness on the KBP dataset. Recent OpenIE research has adopted neural network models, with approaches like SpanOIE using a BiLSTM as an encoder to perform tagging tasks at the span level rather than the token level. Subsequent models like IMoJIE have improved upon this by combining Seq2Seq models with BERT.

## Proposed Method: SelfcheckGPT with SBERT
The proposed method involves replacing BERT with SBERT in the existing framework to assess sentence consistency using cosine similarity of sentence embeddings, which better represents semantic similarity at the sentence level. The SBERT score formula remains the same as shown in the existing literature.

## SelfcheckGPT with HHEM
The proposed method involves replacing the existing DeBERTa-v3 model with HHEM, trained on additional NLI datasets. The HHEM score formula remains consistent with existing metrics for evaluating consistency based on average contradiction probability.

## SelfcheckGPT with OpenIE
Based on the observation that N-gram, particularly Unigram score, performed better than BERT Score in prior studies, the proposed method focuses on domain knowledge modeling using OpenIE. This approach intentionally excludes contextual information and utilizes statistically approached n-grams within OpenIE for generating triples. The extracted triples are then vectorized using pre-trained word embedding models, and the similarity between these vectorized triples is assessed. Due to memory constraints, traditional word embedding models like Glove or FastText were replaced with BERT in experiments. The OpenIE score formula is expressed as shown in the literature.


---

# References

1. Chakraborty, N., Ornik, M., & Driggs-Campbell, K. (2024). *Hallucination Detection in Foundation Models for Decision-Making: A Flexible Definition and Review of the State of the Art*. arXiv preprint arXiv:2403.16527.
2. Azaria, A., & Mitchell, T. (2023). *The internal state of an llm knows when it's lying*. arXiv preprint arXiv:2304.13734.
3. Song, D., Xie, X., Song, J., Zhu, D., Huang, Y., Juefei-Xu, F., & Ma, L. (2023). *LUNA: A Model-Based Universal Analysis Framework for Large Language Models*. arXiv preprint arXiv:2310.14211.
4. Huang, Q., Dong, X., Zhang, P., Wang, B., He, C., Wang, J., ... & Yu, N. (2023). *Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation*. arXiv preprint arXiv:2311.17911.
5. Yang, Y., Chern, E., Qiu, X., Neubig, G., & Liu, P. (2023). *Alignment for honesty*. arXiv preprint arXiv:2312.07000.
6. Liang, K., Zhang, Z., & Fisac, J. F. (2024). *Introspective Planning: Guiding Language-Enabled Agents to Refine Their Own Uncertainty*. arXiv preprint arXiv:2402.06529.
7. Wang, J., Tong, J., Tan, K., Vorobeychik, Y., & Kantaros, Y. (2023). *Conformal temporal logic planning using large language models: Knowing when to do what and when to ask for help*. arXiv preprint arXiv:2309.10092.
8. Li, X., Zhao, R., Chia, Y. K., Ding, B., Bing, L., Joty, S., & Poria, S. (2023). *Chain of knowledge: A framework for grounding large language models with structured knowledge bases*. arXiv preprint arXiv:2305.13269.
9. Manakul, P., Liusie, A., & Gales, M. J. (2023). *Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models*. arXiv preprint arXiv:2303.08896.
10. Reimers, N., & Gurevych, I. (2019). *Sentence-bert: Sentence embeddings using siamese bert-networks*. arXiv preprint arXiv:1908.10084.
11. Hong, G., Gema, A. P., Saxena, R., Du, X., Nie, P., Zhao, Y., ... & Minervini, P. (2024). *The Hallucinations Leaderboard--An Open Effort to Measure Hallucinations in Large Language Models*. arXiv preprint arXiv:2404.05904.
12. Angeli, G., Premkumar, M. J. J., & Manning, C. D. (2015, July). *Leveraging linguistic structure for open domain information extraction*. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 344-354).
13. Zhan, J., & Zhao, H. (2020, April). *Span model for open information extraction on accurate corpus*. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 05, pp. 9523-9530).
14. Cui, L., Wei, F., & Zhou, M. (2018). *Neural open information extraction*. arXiv preprint arXiv:1805.04270.
15. Kolluru, K., Aggarwal, S., Rathore, V., & Chakrabarti, S. (2020). *Imojie: Iterative memory-based joint open information extraction*. arXiv preprint arXiv:2005.08178.