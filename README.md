# Quoran

The case-study utlizes use of SOTA LLMs like BERT, RoBERTa, T5, GPT-2, GPT 3.5 turbo on a dataset sourced from Quora which contains 56k Question Answer pairs scraped from Quora.
You can view the [Quora Question Answer Dataset](https://huggingface.co/datasets/toughdata/quora-question-answer-dataset/embed/viewer/default/train) on Hugging Face.

Major area of study has been a closed book as well as augmented Question Answering, along with various other NLP tasks like topic modelling, semantic analysis, seiamese network implementation.
Various methodologies have been deployed in Exploratory Data Analysis (EDA) phase to extract out important insights from dataset.

A further work is under pipeline to check how usage of frameworks like RAG, finetuning-decomposition using LoRA and that of quantization of NLP tasks.

Machine used: Nvdia A6000 (instance taken from Paperspace VM)

## [Report Link](https://drive.google.com/file/d/1PKkLPsku66JQB5RiRav0MoZDXBO9BK13/view?usp=drive_link)
## [PPT Link](https://docs.google.com/presentation/d/1DKb2sXhlhjuSZq9vIPKq1EH4c2QNjlvd/edit?usp=drive_link&ouid=103154953792064305029&rtpof=true&sd=true)

# 🔥 **MORE UPDATES COMING IN** 🔥
(Git commit has been slowed down due to time out issues from paper space remote VM)

### **Evaluation Metrics For BERT QA Model Causal-run**

| **Metric**                      | **Value**        |
|---------------------------------|------------------|
| eval_loss                       | 1.896943         |
| eval_runtime                    | 0.0968           |
| eval_samples_per_second         | 123.911          |
| eval_steps_per_second           | 20.652           |
| epoch                           | 14.0             |

**ROUGE Metrics**

| **Metric**                      | **Value**        |
|---------------------------------|------------------|
| rouge1_precision_low            | 0.713288         |
| rouge1_recall_low               | 0.045921         |
| rouge1_fmeasure_low             | 0.068042         |
| rouge1_precision_mid            | 0.724761         |
| rouge1_recall_mid               | 0.048894         |
| rouge1_fmeasure_mid             | 0.071487         |
| rouge1_precision_high           | 0.735619         |
| rouge1_recall_high              | 0.051794         |
| rouge1_fmeasure_high            | 0.074884         |
| rouge2_precision_low            | 0.196496         |
| rouge2_recall_low               | 0.027044         |
| rouge2_fmeasure_low             | 0.037401         |
| rouge2_precision_mid            | 0.206170         |
| rouge2_recall_mid               | 0.029530         |
| rouge2_fmeasure_mid             | 0.040387         |
| rouge2_precision_high           | 0.216236         |
| rouge2_recall_high              | 0.032244         |
| rouge2_fmeasure_high            | 0.043399         |
| rougeL_precision_low            | 0.702902         |
| rougeL_recall_low               | 0.043070         |
| rougeL_fmeasure_low             | 0.064185         |
| rougeL_precision_mid            | 0.714032         |
| rougeL_recall_mid               | 0.045818         |
| rougeL_fmeasure_mid             | 0.067366         |
| rougeL_precision_high           | 0.725406         |
| rougeL_recall_high              | 0.048594         |
| rougeL_fmeasure_high            | 0.070413         |
| rougeLsum_precision_low         | 0.703049         |
| rougeLsum_recall_low            | 0.043074         |
| rougeLsum_fmeasure_low          | 0.064148         |
| rougeLsum_precision_mid         | 0.714155         |
| rougeLsum_recall_mid            | 0.045892         |
| rougeLsum_fmeasure_mid          | 0.067428         |
| rougeLsum_precision_high        | 0.725879         |
| rougeLsum_recall_high           | 0.048696         |
| rougeLsum_fmeasure_high         | 0.070737         |

**BLEU Metrics**

| **Metric**                      | **Value**        |
|---------------------------------|------------------|
| bleu                            | 1.385355e-25     |
| bleu_precisions_1               | 0.745941         |
| bleu_precisions_2               | 0.676722         |
| bleu_precisions_3               | 0.632481         |
| bleu_precisions_4               | 0.592542         |
| brevity_penalty                 | 2.100587e-25     |
| length_ratio                    | 0.017294         |
| translation_length              | 15091            |
| reference_length                | 872598           |

 **Exact Match and F1 Score**

| **Metric**                      | **Value**        |
|---------------------------------|------------------|
| exact_match                     | 0.0              |
| f1_score                        | 7.024128         |

### Evaluation metrics for T5

**Evaluation Metrics**

| **Metric**                      | **Value**        |
|---------------------------------|------------------|
| eval_loss                       | 1.896943         |
| eval_runtime                    | 0.0968           |
| eval_samples_per_second         | 123.911          |
| eval_steps_per_second           | 20.652           |
| epoch                           | 14.0             |

**ROUGE Metrics**

| **Metric**                      | **Value**        |
|---------------------------------|------------------|
| rouge1_precision_low            | 0.713288         |
| rouge1_recall_low               | 0.045921         |
| rouge1_fmeasure_low             | 0.068042         |
| rouge1_precision_mid            | 0.724761         |
| rouge1_recall_mid               | 0.048894         |
| rouge1_fmeasure_mid             | 0.071487         |
| rouge1_precision_high           | 0.735619         |
| rouge1_recall_high              | 0.051794         |
| rouge1_fmeasure_high            | 0.074884         |
| rouge2_precision_low            | 0.196496         |
| rouge2_recall_low               | 0.027044         |
| rouge2_fmeasure_low             | 0.037401         |
| rouge2_precision_mid            | 0.206170         |
| rouge2_recall_mid               | 0.029530         |
| rouge2_fmeasure_mid             | 0.040387         |
| rouge2_precision_high           | 0.216236         |
| rouge2_recall_high              | 0.032244         |
| rouge2_fmeasure_high            | 0.043399         |
| rougeL_precision_low            | 0.702902         |
| rougeL_recall_low               | 0.043070         |
| rougeL_fmeasure_low             | 0.064185         |
| rougeL_precision_mid            | 0.714032         |
| rougeL_recall_mid               | 0.045818         |
| rougeL_fmeasure_mid             | 0.067366         |
| rougeL_precision_high           | 0.725406         |
| rougeL_recall_high              | 0.048594         |
| rougeL_fmeasure_high            | 0.070413         |
| rougeLsum_precision_low         | 0.703049         |
| rougeLsum_recall_low            | 0.043074         |
| rougeLsum_fmeasure_low          | 0.064148         |
| rougeLsum_precision_mid         | 0.714155         |
| rougeLsum_recall_mid            | 0.045892         |
| rougeLsum_fmeasure_mid          | 0.067428         |
| rougeLsum_precision_high        | 0.725879         |
| rougeLsum_recall_high           | 0.048696         |
| rougeLsum_fmeasure_high         | 0.070737         |

**BLEU Metrics**

| **Metric**                      | **Value**        |
|---------------------------------|------------------|
| bleu                            | 1.385355e-25     |
| bleu_precisions_1               | 0.745941         |
| bleu_precisions_2               | 0.676722         |
| bleu_precisions_3               | 0.632481         |
| bleu_precisions_4               | 0.592542         |
| brevity_penalty                 | 2.100587e-25     |
| length_ratio                    | 0.017294         |
| translation_length              | 15091            |
| reference_length                | 872598           |

 **Exact Match and F1 Score**

| **Metric**                      | **Value**        |
|---------------------------------|------------------|
| exact_match                     | 0.0              |
| f1_score                        | 7.024128         |


