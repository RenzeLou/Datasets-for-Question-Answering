# Datasets for Question Answering (QA)

*A collection of datasets used in QA tasks. Solely for Natural Language Processing (NLP). Categorization based on the language. The datasets are sorted by year of publication.*

## English

#### 1. MCTEST

- **Dataset**: https://mattr1.github.io/mctest/data.html
- **Publication:** https://aclanthology.org/D13-1020.pdf (2013, EMNLP)
- **Abstract**: MCTest is a freely available set of stories and associated questions intended for research on the machine comprehension of text. MCTest requires machines to answer multiple-choice reading comprehension questions about fictional stories, directly tackling the high-level goal of open-domain machine comprehension.

#### 2. WikiQA

- **Dataset**: https://www.microsoft.com/en-us/download/confirmation.aspx?id=52419
- **Publication:** https://aclanthology.org/D15-1237.pdf (2015, EMNLP)
- **Abstract**: For research on open-domain question answering. Bing query logs were used as the question source. The corpus includes 3,047 questions and 29,258 sentences, where 1,473 sentences were labeled as answer sentences to their corresponding questions.
#### 3. SQuAD (v1.0)

- **Dataset**: https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/
- **Publication:** https://arxiv.org/pdf/1606.05250.pdf (2016, EMNLP)
- **Abstract**: MRC dataset consisting of 100,000+ questions posed by crowdworkers on a set of Wikipedia articles. 

#### 4. CNN/Daily Mail

- **Dataset**: https://github.com/abisee/cnn-dailymail
- **Publication:** https://arxiv.org/pdf/1602.06023v5.pdf (2016, CONLL)
- **Abstract**: CNN/Daily Mail is a dataset for text summarization. The corpus has 286,817 training pairs, 13,368 validation pairs and 11,487 test pairs, as defined by their scripts. The source documents in the training set have 766 words spanning 29.74 sentences on an average while the summaries consist of 53 words and 3.72 sentences.

#### 5. CHILDREN’S BOOK TEST (CBT)

- **Dataset**: https://research.facebook.com/downloads/babi/
- **Publication:** https://arxiv.org/pdf/1511.02301v4.pdf (2016, ICLR)
- **Abstract**: Children’s Book Test (CBT) is designed to measure directly how well language models can exploit wider linguistic context. The CBT is built from books that are freely available thanks to Project Gutenberg.


#### 6. BOOK TEST (BT)

- **Dataset**: https://ibm.ent.box.com/v/booktest-v1
- **Publication:** https://arxiv.org/pdf/1610.00956v1.pdf (2016, arXiv)
- **Abstract**: BookTest is a new dataset similar to the popular Children’s Book Test (CBT), however more than 60 times larger.

#### 7. TriviaQA

- **Dataset**: http://nlp.cs.washington.edu/triviaqa/
- **Publication:** https://arxiv.org/pdf/1705.03551v2.pdf (2017, ACL)
- **Abstract**: TriviaQA includes 95K question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per question on average, that provide high quality distant supervision for answering the questions.

#### 8. RACE

- **Dataset**: https://www.cs.cmu.edu/~glai1/data/race/
- **Publication:** https://arxiv.org/pdf/1704.04683v5.pdf (2017, EMNLP)
- **Abstract**: Consists of 27,933 passages and 97,867 questions from English exams, targeting Chinese students aged 12-18. RACE consists of two subsets, RACE-M and RACE-H, from middle school and high school exams, respectively. RACE-M has 28,293 questions and RACE-H has 69,574. Each question is associated with 4 candidate answers, one of which is correct. 


#### 9. NewsQA

- **Dataset**: https://www.microsoft.com/en-us/research/project/newsqa-dataset/download/
- **Publication:** https://arxiv.org/pdf/1611.09830v3.pdf (2017, WS)
- **Abstract**: The NewsQA dataset is a crowd-sourced machine reading comprehension dataset of 120,000 question-answer pairs. NewsQA is collected using a 3-stage, siloed process.

#### 10. SearchQA

- **Dataset**: https://github.com/nyu-dl/dl4ir-searchQA
- **Publication:** https://arxiv.org/pdf/1704.05179v3.pdf (2017, arXiv)
- **Abstract**: SearchQA was built using an in-production, commercial search engine. It closely reflects the full pipeline of a (hypothetical) general question-answering system, which consists of information retrieval and answer synthesis.

####  11. NarrativeQA

- **Dataset**: https://github.com/deepmind/narrativeqa
- **Publication:** https://arxiv.org/pdf/1712.07040v1.pdf (2018, TACL)
- **Abstract**: The NarrativeQA dataset includes a list of documents with Wikipedia summaries, links to full stories, and questions and answers.

#### 12. SQuAD (v2.0)

- **Dataset**: https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/
- **Publication:** https://arxiv.org/pdf/1806.03822.pdf (2018, ACL)
- **Abstract**: SQuAD 2.0 combines existing SQuAD data with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. 

#### 13. AI2 Reasoning Challenge (ARC)

- **Dataset**: https://allenai.org/data/arc
- **Publication:** https://arxiv.org/pdf/1803.05457v1.pdf (2018, arXiv)
- **Abstract**: A multiple-choice question-answering dataset, containing questions from science exams from grade 3 to grade 9. The dataset is split in two partitions: Easy and Challenge. Most of the questions have 4 answer choices, with <1% of all the questions having either 3 or 5 answer choices. ARC includes a supporting KB of 14.3M unstructured text passages.


#### 14. Natural Questions

- **Dataset**: https://ai.google.com/research/NaturalQuestions
- **Publication:** https://storage.googleapis.com/pub-tools-public-publication-data/pdf/b8c26e4347adc3453c15d96a09e6f7f102293f71.pdf (2019, TACL)
- **Abstract**: Contains 307,373 training examples, 7,830 development examples, and 7,842 test examples. Each example is comprised of a google.com query and a corresponding Wikipedia page.  Finally 1% of the documents have a passage annotated with a short answer that is “yes” or “no”, instead of a list of short spans.



#### 15. MS Marco

- **Dataset**: https://microsoft.github.io/msmarco/
- **Publication:** https://arxiv.org/pdf/1611.09268v3.pdf (2019, NeurIPS)
- **Abstract**: The first dataset was a question answering dataset featuring 100,000 real Bing questions and a human generated answer. Over time the collection was extended with a 1,000,000 question dataset, a natural language generation dataset, a passage ranking dataset, keyphrase extraction dataset, crawling dataset, and a conversational search.


#### 16. CoQA

- **Dataset**: https://stanfordnlp.github.io/coqa/
- **Publication:** https://arxiv.org/pdf/1808.07042v2.pdf (2019, TACL)
- **Abstract**: For Conversational Question Answering systems CoQA contains 127,000+ questions with answers collected from 8000+ conversations. Each conversation is collected by pairing two crowdworkers to chat about a passage in the form of questions and answers. 


## Chinese
#### 1. ORCD

- **Dataset**: https://github.com/DRCKnowledgeTeam/DRCD
- **Publication:** https://arxiv.org/ftp/arxiv/papers/1806/1806.00920.pdf (2018, arXiv)
- **Abstract**: An open domain traditional Chinese machine reading comprehension (MRC) dataset. The dataset contains 10,014 paragraphs from 2,108 Wikipedia articles and 30,000+ questions generated by annotators.

#### 2. MATINF

- **Dataset**: https://github.com/WHUIR/MATINF
- **Publication:** https://arxiv.org/pdf/2004.12302v2.pdf (2020, ACL)
- **Abstract**: Jointly labeled for classification, question answering and summarization in the domain of maternity and baby caring in Chinese. An entry in the dataset includes four fields: question (Q), description (D), class (C) and answer (A).


#### 3. LiveQA

- **Dataset**: https://github.com/PKU-TANGENT/LiveQA
- **Publication:** https://arxiv.org/pdf/2010.00526.pdf (2020, CCL)
- **Abstract**: It contains 117k multiple-choice questions written by human commentators for over 1,670 NBA games, which are collected from the Chinese Hupu website. In LiveQA, the questions require understanding the timeline, tracking events or doing mathematical computations.
## Other Languages

#### 1. JaQuAD

- **Dataset**: https://github.com/SkelterLabsInc/JaQuAD
- **Publication:** https://arxiv.org/pdf/2202.01764v1.pdf (2022, arXiv)
- **Coverage**: Japanese
- **Abstract**: JaQuAD (Japanese Question Answering Dataset) is a question answering dataset in Japanese that consists of 39,696 extractive question-answer pairs on Japanese Wikipedia articles.
#### 2. GermanQuAD

- **Dataset**: https://www.deepset.ai/datasets
- **Publication:**  https://arxiv.org/pdf/2104.12741v1.pdf (2021, MRQA)
- **Coverage**: German
- **Abstract**: GermanQuAD is a Question Answering (QA) dataset of 13,722 extractive question/answer pairs in German.

#### 3. DaNetQA

- **Dataset**: https://github.com/RussianNLP/RussianSuperGLUE
- **Publication:** https://arxiv.org/pdf/2010.15925v2.pdf (2020, EMNLP)
- **Coverage**: Russian
- **Abstract**: DaNetQA is a question answering dataset for yes/no questions. These questions are naturally occurring ---they are generated in unprompted and unconstrained settings.



#### 4. MuSeRC

- **Dataset**: https://github.com/RussianNLP/RussianSuperGLUE
- **Publication:** https://arxiv.org/pdf/2010.15925v2.pdf (2020, EMNLP)
- **Coverage**: Russian
- **Abstract**: The dataset is the first to study multi-sentence inference at scale, with an open-ended set of question types that requires reasoning skills.



#### 5. RuCoS

- **Dataset**: https://github.com/RussianNLP/RussianSuperGLUE
- **Publication:** https://arxiv.org/pdf/2010.15925v2.pdf (2020, EMNLP)
- **Coverage**: Russian
- **Abstract**: Requires commonsense reasoning. RuCoS consists of queries automatically generated from CNN/Daily Mail news articles; the answer to each query is a text span from a summarizing passage of the corresponding news.



#### 6. HeadQA

- **Dataset**: https://github.com/aghie/head-qa
- **Publication:** https://arxiv.org/pdf/1906.04701v1.pdf (2019, ACL)
- **Coverage**: Spanish
- **Abstract**: Multi-choice question answering . The questions come from exams to access a specialized position in the Spanish healthcare system, and are challenging even for highly specialized humans.

#### 7. FQuAD

- **Dataset**: https://fquad.illuin.tech/
- **Publication:** https://arxiv.org/pdf/2002.06071v2.pdf (2020, EMNLP)
- **Coverage**: French
- **Abstract**: A French Native Reading Comprehension dataset of questions and answers on a set of Wikipedia articles that consists of 25,000+ samples for the 1.0 version and 60,000+ samples for the 1.1 version.


#### 8. KLEJ

- **Dataset**: https://klejbenchmark.com/
- **Publication:** https://arxiv.org/pdf/2005.00630v1.pdf (2020, ACL)
- **Coverage**: Polish
- **Abstract**: The KLEJ benchmark (Kompleksowa Lista Ewaluacji Językowych) is a set of nine evaluation tasks for the Polish language understanding task.

#### 9. MilkQA

- **Dataset**: http://nilc.icmc.usp.br/nilc/index.php/milkqa
- **Publication:** https://arxiv.org/pdf/1801.03460v1.pdf (2017, BRACIS)
- **Coverage**: Portuguese
- **Abstract**: A question answering dataset from the dairy domain dedicated to the study of consumer questions. The dataset contains 2,657 pairs of questions and answers, written in the Portuguese language.

#### 10. PersianQA

- **Dataset**: https://github.com/sajjjadayobi/PersianQA
- **Publication:** None 
- **Coverage**: Persian
- **Abstract**: Based on Persian Wikipedia. The crowd-sourced the dataset consists of more than 9,000 entries. Each entry can be either an impossible-to-answer or a question with one or more answers spanning in the passage (the context) from which the questioner proposed the question.


## Multi-lingual

#### 1. FM-IQA

- **Dataset**: http://research.baidu.com/Downloads 
- **Publication:**  https://arxiv.org/pdf/1505.05612v3.pdf (2015, NeurIPS)
- **Coverage**: English, Chinese
- **Abstract**: A question-answering dataset containing over 150,000 images and 310,000 freestyle Chinese question-answer pairs and their English translations.

#### 2. SuperGLUE

- **Dataset**: https://super.gluebenchmark.com/
- **Publication:** https://arxiv.org/pdf/1905.00537v3.pdf (2019, NeurIPS)
- **Coverage**: eight language
- **Abstract**: A new benchmark styled after GLUE with a new set of more difficult language understanding tasks, a software toolkit, and a public leaderboard.

#### 3. XQA

- **Dataset**: https://github.com/thunlp/XQA
- **Publication:** https://aclanthology.org/P19-1227.pdf (2019, ACL)
- **Coverage**: nine languages
- **Abstract**: XQA is a data which consists of a total amount of 90k question-answer pairs in nine languages for cross-lingual open-domain question answering.



#### 4. XQuAD

- **Dataset**: https://github.com/deepmind/xquad
- **Publication:** https://arxiv.org/pdf/1910.11856v3.pdf (2020, ACL)
- **Coverage**: Spanish, German, Greek, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, and Hindi. 
- **Abstract**: XQuAD (Cross-lingual Question Answering Dataset) is a benchmark dataset for evaluating cross-lingual question answering performance. The dataset consists of a subset of 240 paragraphs and 1190 question-answer pairs from the development set of SQuAD v1.1 

#### 5. MLQA

- **Dataset**: https://github.com/facebookresearch/mlqa
- **Publication:**  https://arxiv.org/pdf/1910.07475v3.pdf (2020, ACL)
- **Coverage**: English, Arabic, German, Spanish, Hindi, Vietnamese, Chinese.
- **Abstract**: MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages. MLQA is highly parallel, with QA instances parallel between 4 different languages on average.


#### 6. RELX

- **Dataset**: https://github.com/boun-tabi/RELX
- **Publication:** https://arxiv.org/pdf/2010.09381v1.pdf (2020, EMNLP findings)
- **Coverage**: English, French, German, Spanish and Turkish.
- **Abstract**: RELX is a benchmark dataset for cross-lingual relation classification in English, French, German, Spanish and Turkish.



#### 7. MKQA

- **Dataset**: https://github.com/apple/ml-mkqa/
- **Publication:** https://arxiv.org/pdf/2007.15207v2.pdf (2020, arXiv)
- **Coverage**: 26 languages
- **Abstract**: Multilingual Knowledge Questions and Answers (MKQA) is an open-domain question answering evaluation set comprising 10k question-answer pairs aligned across 26 typologically diverse languages (260k question-answer pairs in total). 



