# Advanced experiments
- /demo/experiments/advanced_experiments.ipynb
- demo/experiments/advanced_experiments_checkpoint
- https://drive.google.com/file/d/1PtsXKbnIxlccUmbvIWxBualbp9oTJzIq/view?usp=sharing 

# Table
[README-EN](#README-EN)
[README-KO](#README-KO)
---
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

# README-KO

# Purpose
 SelfCheckGPT는 LLM의 최종 텍스트만 얻을 수 있고 학습이나 RAG에 사용된 데이터를 모르는 경우 응답 텍스트간 일관성을 측정하여 할루시네이션을 감지하는 방법을 제시합니다.  
 논문에서는 GPT-3의 logProb와 llama를 활용한 Proxy-tuning과 비교해본 결과 최종 토큰의 확률분포를 활용하는보다 응답 텍스트간 일관성 비교가 할루시네이션 감지 정확도가 높았다고 보고합니다. 
 또한 대부분의 서비스들이 직접 LLM을 운영하지 않을 것이기 때문에 항상 토큰 확률 분포를 얻을 수 없습니다. 이 두 가지 이유에서 해당 논문이 다른 할루시네이션 감지 논문 대비 프로덕션 실용성이 높다고 판단해 선택하게 되었습니다.

# Literature review
 동일 쿼리에 대한 출력 응답들의 일관성이 낮을 수록 할루시네이션일 확률이 높다는 가정으로 시작합니다.
검증하고자 하는 응답 R, 동일 쿼리에 대한 응답 샘플들 S의 문장 유사도를 일관성 점수로 정의합니다.
문장 유사도의 평균으로 문단의 일관성 점수로 정의합니다.
 WikiBio longtail 데이터로 인한 할루시네이션 현상을 배제하기 위해 빈도 상위 20%중 랜덤 238개에 대해 GPT에게 기사를 작성시켰습니다. 기사의 문장마다 수동 레이블(Major Inaccurate/Minor Inaccurate/Accurate)을 매깁니다.
 문장 단위 평가(PR-AUC), 문단 단위 평가(PCC)를 수행합니다. 논문에서는 PCC 인간 레이블과 평균 문장 점수간 피어슨 상관 계수와 스피어먼 순위 상관 계수값를 확인합니다.


# Proposed Method
총 5개의 일관성 정의방법이 제안 되어있습니다.
1- (문장 Ri와 최대로 일치하는 문장Si의 임베딩 유사도 평균) : RoBERTa-Large 사용
MQAG프레임워크로 쿼리에 대한 선택지 생성 후 다른 선택지가 선택되는 비율
R과 S들로 N-gram model을 훈련하여 토큰 확률 모방, 가장 낮은 또는 평균 토큰 확률값
전제 S들이 주어졌을 때, 문장 Ri가 모순일 확률: DeBERTa-v3-large 사용
S들이 프롬프트에 컨텍스트로 주어졌을 때 LLM이 Ri가 거짓이라고 응답하는 비율


# Experiments
## GreyBox approach보다 평균적으로 성능이 좋다
GreyBox: 최종 layer의 token probabilty(또는 logits)를 알고 있을 때
Proxy-tuning: LLM의 logits에 완전 접근 및 학습 가능한 LM의 logits를 더해서 decoding 되는 확률 분포를 조작하는 방식
프록시 LLM은 엔트로피 H 측정치가 확률 측정치보다 더 나은 성능을 보이고 있고, 성능이 안 좋은데 논문에서는 그 이유가 LLM들이 서로 다른 생성 패턴을 가지기 때문인 것 같다고 추측합니다.

## Unigram이 BERTScore보다 성능이 좋다
흥미롭게도, 가장 계산 비용이 적게 드는 방법임에도 불구하고, unigram (max)을 사용하는 방식이 성능이 좋은데  샘플 전체에서 가장 낮은 발생 빈도를 가진 토큰을 선택하기 때문으로 추측합니다.

## 성능과 비용을 절충했을 때 NLI가 좋다
LLM Prompt가 제일 성능이 좋기는 하지만 Passage level에서도 성능이 좋은 NLI을 추천하고 있습니다.


# Proposal
 기존 방식 2개를 강화하고, 새로운 아이디어 하나를 추가하여 총 3가지 방법론을 제안합니다.
추가 데이터셋으로 fine-tuning된 NLI 모델인 HHEM 사용하여 NLI Scoring을 개선한 HHEM Scoring을 제안합니다.
BERT보다 문장 임베딩 성능이 뛰어난 SBERT사용하여 BERT Scoring을 개선한 SBERT Scoring을 제안합니다.
BERT보다 Unigram model이 성능이 좋았던 점에서 착안하여 도메인 지식 모델링하여 비문맥적 정보를 활용하도록 OpenIE를 적용한 OpenIE Scoring을 제안합니다.


# HHEM Scoring
 HHEM Scoring은 HHEM(Hughes Hallucination Evaluation Model)을 사용합니다. HHDEM의 basemodel은 논문에서 사용한 microsoft deBERTa-v3와 동일하나 FEVER, Vitamin C, PAWS와 같은 Factuality 데이터셋에 대해 fine-tuning하여 NLI 성능을 향상
[0, 1] 사이 값 출력하는데 0일수록 환각이고 1일 수록 사실과 일치합니다.
 문장 Ri에 대해 Samples가 전제로 주어졌을 때 HHEM의 평균 출력값을 O라고 했을 때 1-O을 일관성 점수로 사용했습니다.


# SBERT Scoring
SBERT(Sentence BERT)는 BERT Siamese(또는 Triplet) 네트워크를 활용하여 BERT pooling보다 의미론적 문장 임베딩을 생성하는 모델입니다. 해당 모델은 SNLI 데이터셋과 Multi-Genre NLI 데이터셋으로 파인튜닝됩니다.
우선 문장 Ri의 SBERT 임베딩과 Sample의 문장 Si의 SBERT 임베딩의 코사인 유사도를 구하고, Sample별 최대 유사도 값들의 평균 M을 구했습니다. 1-M을 일관성 점수로 사용합니다.


# OpenIE Scoring
OpenIE(Open Information Extraction)는  자연어 텍스트로부터 triples(주체, 관계, 객체) 형태로 구조화된 정보 추출하는 기술로 통계적 방식(ex. Stanford OpenIE) 또는 신경망적 방식(ex.IMoJIE)이 있습니다. 이번 실험에서는 Stanford OpenIE내부에서 N-gram이 사용되고 있기 때문에 Stanford OpenIE를 사용했습니다.

OpenIE Scoring
문장 Ri과 Sample에서 triples 추출하고 triple의 각 단어를 임베딩했습니다. triple간 최대 유사도 T를 구하고 1-T를 일관성 점수로 사용헸습니다.


# Experiments 
## Detecting False
동일한 데이터 셋과 레이블에 대해 PR Curve AUC 값을 비교한 결과입니다.
문장 전체가 잘못된 데이터에 대해  NLI Scoring이 가장 높은 성능을 보여주었습니다.

## Detecting False*
문장 일부가 잘못된 데이터에 대해  BERT Scoring이 가장 높은 성능을 보여주었습니다.

## Detecting True
문장 전체가 사실인 데이터에 대해 N-gram Scoring이 가장 높은 성능을 보여주었습니다.

## PCC
문단 단위로 예측 레이블과 인간 레이블과의 선형관계성을 비교한 결과 NLI가 가장 인간과 유사했습니다.

# Conclusion
.

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