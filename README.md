# Advanced experiments
- /demo/experiments/advanced_experiments.ipynb

# Proposed Method

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