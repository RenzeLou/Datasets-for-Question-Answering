# Dataset for Question Answering (QA)

*A collection of datasets used in QA tasks. Solely for Natural Language Processing (NLP). Categorization based on the language.*

## English

####  Stanford Question Answering Dataset (SQuAD)

- **Dataset**:https://rajpurkar.github.io/SQuAD-explorer/
- **Publication:** https://arxiv.org/pdf/1606.05250v3.pdf  （EMNLP 2016 ）
- **Abstract**: We present the Stanford Question Answering Dataset (SQuAD), a new reading comprehension dataset consisting of 100,000+ questions posed by crowdworkers on a set of Wikipedia articles.We build a strong logistic regression model, which achieves an F1 score of 51.0%, a significant improvement over a simple baseline (20%). However, human performance (86.8%) is much higher, indicating that the dataset presents a good challenge problem for future research. 



#### WikiQA

- **Dataset**:https://www.microsoft.com/en-us/download/confirmation.aspx?id=52419
- **Publication:** https://aclanthology.org/D15-1237.pdf  （EMNLP 2015 ）
- **Abstract**: The WikiQA corpus is a publicly available set of question and sentence pairs, collected and annotated for research on open-domain question answering. In order to reflect the true information need of general users, Bing query logs were used as the question source. The corpus includes 3,047 questions and 29,258 sentences, where 1,473 sentences were labeled as answer sentences to their corresponding questions.



#### Natural Questions

- **Dataset**:https://ai.google.com/research/NaturalQuestions
- **Publication:** https://storage.googleapis.com/pub-tools-public-publication-data/pdf/b8c26e4347adc3453c15d96a09e6f7f102293f71.pdf （ TACL2019）
- **Abstract**: The Natural Questions corpus is a question answering dataset containing 307,373 training examples, 7,830 development examples, and 7,842 test examples. Each example is comprised of a google.com query and a corresponding Wikipedia page.  Finally 1% of the documents have a passage annotated with a short answer that is “yes” or “no”, instead of a list of short spans.



#### MS Marco

- **Dataset**:https://microsoft.github.io/msmarco/
- **Publication:**https://arxiv.org/pdf/1611.09268v3.pdf (NULL)
- **Abstract**: The MS MARCO (Microsoft MAchine Reading Comprehension) is a collection of datasets focused on deep learning in search. The first dataset was a question answering dataset featuring 100,000 real Bing questions and a human generated answer. Over time the collection was extended with a 1,000,000 question dataset, a natural language generation dataset, a passage ranking dataset, keyphrase extraction dataset, crawling dataset, and a conversational search.



#### NewsQA

- **Dataset**:https://www.microsoft.com/en-us/research/project/newsqa-dataset/download/
- **Publication:** https://arxiv.org/pdf/1611.09830v3.pdf  （WS2017）
- **Abstract**: The NewsQA dataset is a crowd-sourced machine reading comprehension dataset of 120,000 question-answer pairs.NewsQA is collected using a 3-stage, siloed process.


#### CNN/Daily Mail

- **Dataset**: https://github.com/abisee/cnn-dailymail
- **Publication:** https://arxiv.org/pdf/1602.06023v5.pdf (CONLL 2016)
- **Abstract**: CNN/Daily Mail is a dataset for text summarization. The corpus has 286,817 training pairs, 13,368 validation pairs and 11,487 test pairs, as defined by their scripts. The source documents in the training set have 766 words spanning 29.74 sentences on an average while the summaries consist of 53 words and 3.72 sentences.


#### CoQA

- **Dataset**:https://stanfordnlp.github.io/coqa/
- **Publication:** https://arxiv.org/pdf/1808.07042v2.pdf  （TACL 2019 ）
- **Abstract**: CoQA is a large-scale dataset for building Conversational Question Answering systems. The goal of the CoQA challenge is to measure the ability of machines to understand a text passage and answer a series of interconnected questions that appear in a conversation.CoQA contains 127,000+ questions with answers collected from 8000+ conversations. Each conversation is collected by pairing two crowdworkers to chat about a passage in the form of questions and answers. 



#### SearchQA

- **Dataset**:https://github.com/nyu-dl/dl4ir-searchQA
- **Publication:** https://arxiv.org/pdf/1704.05179v3.pdf  （NULL）
- **Abstract**: SearchQA was built using an in-production, commercial search engine. It closely reflects the full pipeline of a (hypothetical) general question-answering system, which consists of information retrieval and answer synthesis.



####  NarrativeQA

- **Dataset**:https://deepmind.com/research?filters_and=%7B%22tags%22:%5B%22Datasets%22%5D%7D
- **Publication:** https://arxiv.org/pdf/1712.07040v1.pdf  （TACL 2018）
- **Abstract**: The NarrativeQA dataset includes a list of documents with Wikipedia summaries, links to full stories, and questions and answers.



#### TriviaQA

- **Dataset**:https://aclanthology.org/D13-1020.pdf 
- **Publication:** https://arxiv.org/pdf/1705.03551v2.pdf  （ACL 2017）
- **Abstract**: TriviaQA is a realistic text-based question answering dataset which includes 950K question-answer pairs from 662K documents collected from Wikipedia and the web. This dataset is more challenging than standard QA benchmark datasets such as Stanford Question Answering Dataset (SQuAD), as the answers for a question may not be directly obtained by span prediction and the context is very long. TriviaQA dataset consists of both human-verified and machine-generated QA subsets.



#### MCTEST

- **Dataset**:https://mattr1.github.io/mctest/data.html
- **Publication:** https://aclanthology.org/D13-1020.pdf  （EMNLP 2013  ）
- **Abstract**: MCTest is a freely available set of stories and associated questions intended for research on the machine comprehension of text.MCTest requires machines to answer multiple-choice reading comprehension questions about fictional stories, directly tackling the high-level goal of open-domain machine comprehension.



#### CHILDREN’S BOOK TEST

- **Dataset**:https://research.facebook.com/downloads/babi/
- **Publication:** https://arxiv.org/pdf/1511.02301v4.pdf （NULL）
- **Abstract**: Children’s Book Test (CBT) is designed to measure directly how well language models can exploit wider linguistic context. The CBT is built from books that are freely available thanks to Project Gutenberg.



#### BOOK TEST

- **Dataset**:https://ibm.ent.box.com/v/booktest-v1
- **Publication:**https://arxiv.org/pdf/1610.00956v1.pdf （NULL）
- **Abstract**: BookTest is a new dataset similar to the popular Children’s Book Test (CBT), however more than 60 times larger.



#### RACE

- **Dataset**:https://www.cs.cmu.edu/~glai1/data/race/
- **Publication:**https://arxiv.org/pdf/1704.04683v5.pdf （EMNLP 2017 ）
- **Abstract**: The ReAding Comprehension dataset from Examinations (RACE) dataset is a machine reading comprehension dataset consisting of 27,933 passages and 97,867 questions from English exams, targeting Chinese students aged 12-18. RACE consists of two subsets, RACE-M and RACE-H, from middle school and high school exams, respectively. RACE-M has 28,293 questions and RACE-H has 69,574. Each question is associated with 4 candidate answers, one of which is correct. 



#### ARC (AI2 Reasoning Challenge)

- **Dataset**:https://allenai.org/data/arc
- **Publication:** https://arxiv.org/pdf/1803.05457v1.pdf （NULL）
- **Abstract**: The AI2’s Reasoning Challenge (ARC) dataset is a multiple-choice question-answering dataset, containing questions from science exams from grade 3 to grade 9. The dataset is split in two partitions: Easy and Challenge, where the latter partition contains the more difficult questions that require reasoning. Most of the questions have 4 answer choices, with <1% of all the questions having either 3 or 5 answer choices. ARC includes a supporting KB of 14.3M unstructured text passages.




## Other Languages


#### MATINF (Maternal and Infant Dataset)

- **Dataset**: https://github.com/WHUIR/MATINF
- **Publication:**https://arxiv.org/pdf/2004.12302v2.pdf （ACL 2020 ）
- **Coverage**: Chinese
- **Abstract**: Maternal and Infant (MATINF) Dataset is a large-scale dataset jointly labeled for classification, question answering and summarization in the domain of maternity and baby caring in Chinese.



#### ODSQA (Open-Domain Spoken Question Answering)

- **Dataset**:https://github.com/chiahsuan156/ODSQA
- **Publication:**https://arxiv.org/pdf/1808.02280v1.pdf （NULL）
- **Coverage**: Chinese
- **Abstract**: The ODSQA dataset is a spoken dataset for question answering in Chinese. It contains more than three thousand questions from 20 different speakers.



#### DaNetQA (Yes/no Question Answering Dataset for the Russian)

- **Dataset**:https://github.com/RussianNLP/RussianSuperGLUE
- **Publication:**https://arxiv.org/pdf/2010.15925v2.pdf （EMNLP 2020 ）
- **Coverage**: Russian
- **Abstract**: DaNetQA is a question answering dataset for yes/no questions. These questions are naturally occurring ---they are generated in unprompted and unconstrained settings.



#### MuSeRC (Russian Multi-Sentence Reading Comprehension)

- **Dataset**: https://github.com/RussianNLP/RussianSuperGLUE
- **Publication:**https://arxiv.org/pdf/2010.15925v2.pdf EMNLP 2020 ）
- **Coverage**: Russian
- **Abstract**: The dataset is the first to study multi-sentence inference at scale, with an open-ended set of question types that requires reasoning skills.



#### RuCoS (Russian Reading Comprehension with Commonsense Reasoning)

- **Dataset**: https://github.com/RussianNLP/RussianSuperGLUE
- **Publication:**https://arxiv.org/pdf/2010.15925v2.pdf （EMNLP 2020 ）
- **Coverage**: Russian
- **Abstract**: Russian reading comprehension with Commonsense reasoning (RuCoS) is a large-scale reading comprehension dataset that requires commonsense reasoning. RuCoS consists of queries automatically generated from CNN/Daily Mail news articles; the answer to each query is a text span from a summarizing passage of the corresponding news. The goal of RuCoS is to evaluate a machine`s ability of commonsense reasoning in reading comprehension.



#### HeadQA

- **Dataset**: https://github.com/aghie/head-qa
- **Publication:**https://arxiv.org/pdf/1906.04701v1.pdf （ACL 2019）
- **Coverage**: Spanish
- **Abstract**: HeadQA is a multi-choice question answering testbed to encourage research on complex reasoning. The questions come from exams to access a specialized position in the Spanish healthcare system, and are challenging even for highly specialized humans.



#### FQuAD (French Question Answering Dataset)

- **Dataset**: https://fquad.illuin.tech/
- **Publication:**https://arxiv.org/pdf/2002.06071v2.pdf （EMNLP 2020）
- **Coverage**: French
- **Abstract**: A French Native Reading Comprehension dataset of questions and answers on a set of Wikipedia articles that consists of 25,000+ samples for the 1.0 version and 60,000+ samples for the 1.1 version.



#### KLEJ

- **Dataset**: https://klejbenchmark.com/
- **Publication:** https://arxiv.org/pdf/2005.00630v1.pdf （ACL 2020）
- **Coverage**:Polish 
- **Abstract**: The KLEJ benchmark (Kompleksowa Lista Ewaluacji Językowych) is a set of nine evaluation tasks for the Polish language understanding task.



#### MilkQA

- **Dataset**: http://nilc.icmc.usp.br/nilc/index.php/milkqa
- **Publication:** https://arxiv.org/pdf/1801.03460v1.pdf （NULL）
- **Coverage**:Portuguese
- **Abstract**: A question answering dataset from the dairy domain dedicated to the study of consumer questions. The dataset contains 2,657 pairs of questions and answers, written in the Portuguese language.



#### JaQuAD

- **Dataset**: https://github.com/SkelterLabsInc/JaQuAD
- **Publication:** https://arxiv.org/pdf/2202.01764v1.pdf (NULL)
- **Coverage**:Japanese
- **Abstract**: JaQuAD (Japanese Question Answering Dataset) is a question answering dataset in Japanese that consists of 39,696 extractive question-answer pairs on Japanese Wikipedia articles.







## TODO

#### PersianQA (Persian Question Answering Dataset)

- **Dataset**: https://github.com/sajjjadayobi/PersianQA
- **Publication:**TODO
- **Coverage**:Persian
- **Abstract**: Persian Question Answering (PersianQA) Dataset is a reading comprehension dataset on Persian Wikipedia. The crowd-sourced the dataset consists of more than 9,000 entries. 








#### COPA-HR

- **Dataset**: https://www.clarin.si/repository/xmlui/handle/11356/1404
- **Publication:** https://arxiv.org/pdf/2104.09243v1.pdf （NULL）
- **Coverage**:Croatian
- **Abstract**: The COPA-HR dataset (Choice of plausible alternatives in Croatian) is a translation of the English COPA dataset by following the XCOPA dataset translation methodology. 



#### GermanQuAD

- **Dataset**: https://www.deepset.ai/datasets
- **Publication:**  https://arxiv.org/pdf/2104.12741v1.pdf (MRQA 2021)
- **Coverage**: German
- **Abstract**: GermanQuAD is a Question Answering (QA) dataset of 13,722 extractive question/answer pairs in German.



## Multi-lingual


#### SuperGLUE

- **Dataset**: https://super.gluebenchmark.com/
- **Publication:** https://arxiv.org/pdf/1905.00537v3.pdf (NeurIPS 2019 )
- **Coverage**:eight language
- **Abstract**: It consists of a public leaderboard built around eight language understanding tasks, drawing on existing data, accompanied by a single-number performance metric, and an analysis toolkit. 

#### XQA

- **Dataset**: https://github.com/thunlp/XQA
- **Publication:** https://aclanthology.org/P19-1227.pdf (ACL 2019 )
- **Coverage**:nine languages
- **Abstract**: XQA is a data which consists of a total amount of 90k question-answer pairs in nine languages for cross-lingual open-domain question answering.



#### XQuAD

- **Dataset**:https://github.com/deepmind/xquad
- **Publication:** https://arxiv.org/pdf/1910.11856v3.pdf (ACL 2020 )
- **Coverage**:Spanish, German, Greek, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, and Hindi. 
- **Abstract**:XQuAD (Cross-lingual Question Answering Dataset) is a benchmark dataset for evaluating cross-lingual question answering performance. The dataset consists of a subset of 240 paragraphs and 1190 question-answer pairs from the development set of SQuAD v1.1 



#### RELX

- **Dataset**: https://github.com/boun-tabi/RELX
- **Publication:** https://arxiv.org/pdf/2010.09381v1.pdf (NULL)
- **Coverage**:English, French, German, Spanish and Turkish.
- **Abstract**: RELX is a benchmark dataset for cross-lingual relation classification in English, French, German, Spanish and Turkish.



#### MKQA (Multilingual Knowledge Questions and Answers)

- **Dataset**: https://github.com/apple/ml-mkqa/
- **Publication:**https://arxiv.org/pdf/2007.15207v2.pdf (NULL)
- **Coverage**:26 languages
- **Abstract**: Multilingual Knowledge Questions and Answers (MKQA) is an open-domain question answering evaluation set comprising 10k question-answer pairs aligned across 26 typologically diverse languages (260k question-answer pairs in total). 



#### FM-IQA

- **Dataset**: http://research.baidu.com/Downloads 
- **Publication:**  https://arxiv.org/pdf/1505.05612v3.pdf (NeurIPS 2015)
- **Coverage**: English, Chinese
- **Abstract**: A question-answering dataset containing over 150,000 images and 310,000 freestyle Chinese question-answer pairs and their English translations.



#### MLQA

- **Dataset**: https://github.com/facebookresearch/mlqa
- **Publication:**  https://arxiv.org/pdf/1910.07475v3.pdf (ACL 2020)
- **Coverage**: English, Arabic, German, Spanish, Hindi, Vietnamese, Chinese
- **Abstract**: MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance. MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic, German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between 4 different languages on average.



