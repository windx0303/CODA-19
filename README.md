# CODA-19: **CO**VI**D**-19 Research **A**spect Dataset

#### This is the Github repo for [CODA-19: Reliably Annotating Research Aspects on 10,000+ CORD-19 Abstracts Using a Non-Expert Crowd](https://arxiv.org/abs/2005.02367).

CODA-19 is a human-annotated dataset that denotes the **Background, Purpose, Method, Finding/Contribution, and Other** for **10,966 English abstracts** in the [COVID-19 Open Research Dataset](https://www.semanticscholar.org/cord19).

CODA-19 was created by 248 crowd workers from [Amazon Mechanical Turk](https://www.mturk.com/) collectively within ten days.
Each abstract was annotated by nine different workers, and the final labels were obtained by majority voting.

CODA-19's labels have an accuracy of 82% and an inter-annotator agreement (Cohen's kappa) of 0.74 when compared against expert labels on 129 abstracts.

The following is an actual abstract (you can see the paper [here](https://www.biorxiv.org/content/10.1101/509141v1.full)) annotated by crowd workers in CODA-19. 

<p align="center">
  <img src="https://crowd.ist.psu.edu/CODA19/img/example_v2_color_blind_safe.png" width="50%">
</p>

## Motivation

This work was developed upon the long history of research on understanding scientific papers at scale. 
In short, the rapid acceleration in new coronavirus literature makes it hard to keep up with.
So we highlighted the papers with its **Background, Purpose, Method, Finding/Contribution, and Other**.
People can use these data to build an automated annotator to label the remaining papers in [CORD-19](https://pages.semanticscholar.org/coronavirus-research) and, more importantly, future papers.
This type of annotation can also be useful for various BioNLP tasks.

## Data Selection and Preprocessing

#### Tokenization, Sentence Segmentation, and Text Segmentation
We used [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) to tokenize and segment sentences for all the abstracts in CORD-19.
We further used comma (,), semicolon (;), and period (.) to split each sentence into shorter fragments, where a fragment has no less than six tokens (including punctuation marks) and has no orphan parentheses.
As of April 15, 2020, 29,306 article in CORD-19 had a non-empty abstract.

#### Paper Filtering
We filtered out the 538 (1.84\%) abstracts with only one sentence and the 145 (0.49\%) abstracts that had more than 1,200 tokens.
We randomly selected 11,000 abstracts from the remaining data for annotation.

#### Language Identification
During the annotation process, workers informed us that a few articles were not in English. 
We identified them automatically using [langdetect](https://github.com/Mimino666/langdetect) and excluded them.
The released version of CODA-19 has totally 10,966 abstracts.

## Folder Structure
```
├── human_label                                     # human labels folder
│   ├── test                                        # test set, containing 1,000 abstracts
│   │   └── expert                                  # expert labels folder
│   │       ├── biomedical_expert                   # expert labels from Bio expert
│   │       ├── computer_science_expert             # expert labels from CS expert
│   │       ├── biomedical_expert-eval.csv          # crowd labels evaluated against Bio expert's labels
│   │       └── computer_science_expert-eval.csv    # crowd labels evaluated against CS expert's labels
│   ├── dev                                         # dev set, containing 1,000 abstracts
│   ├── train                                       # training set, containing 8,965 abstracts
│   └── coda_metadata.csv                           # metadata for CODA-19
└── machine_label                                   # empty folder (desgined for future automatic labels)
```

## Data JSON Schema

```
{
  "paper_id": the paper ID in CORD-19,
  "metadata": {
    "title": the title of the paper,
    "coda_data_split": test/dev/train in CODA-19,
    "coda_paper_id": numeric id (starting from 1) in CODA-19,
    "coda_has_expert_labels": if this paper comes with expert labels in CODA-19,
    "subset": the subset (custom_license/biorxiv_medrxiv/comm_use_subset/noncomm_use_subset) in CORD-19
  },
  "abstract": [
    { 
      "original_text": the tokenized text of the paragraph 1,
      "sentences": [
        [ 
          {
            "segment_text": the tokenized text of the text segment 1 in sentence 1 in paragraph 1, 
            "crowd_label": the label derived (e.g., majority vote) from a set of crowd labels
          },
          {
            "segment_text": the tokenized text of the text segment 2 in sentence 1 in paragraph 1, 
            ...
          },
          ...
        ],
        [ 
          {
            "segment_text": the tokenized text of the text segment 1 in sentence 2 in paragraph 1, 
            ...
          },
          ...
        ],
        ...
      ]
    }
    { 
        "original_text": the tokenized text of the paragraph 2,
        "sentences": [
            ...
        ]
    },
    ...
  ],
  "abstract_stats": {
    "paragraph_num": the total number of paragraphs in this abstract,
    "sentence_num": the total number of sentences in this abstract,
    "segment_num": the total number of text segments in this abstract,
    "token_num": the total number of token in this abstract
  }
}
```

## Data Quality

We worked with a biomedical expert and a computer scientist to assess the label quality.
Both experts respectively annotated the same 129 abstracts randomly selected from CODA-19.
The inter-annotator agreement (Cohen's kappa) between two expert is 0.788.
Table 2 shows the aggregated crowd label's accuracy, along with the precision, recall, and F1-score of each class.
CODA-19's labels have an accuracy of 0.82 and a kappa of 0.74, when compared against two experts' labels.
It is noteworthy that when we compared labels between two experts, the accuracy (0.850) and kappa (0.788) were only slightly higher.

<p align="center">
  <img src="https://crowd.ist.psu.edu/CODA19/img/crowd_eval_table.png" width="90%">
</p>




## How much did it cost?
Annotating one abstract costs **$3.2** on average with our setup. This cost includes the payments for workers and the 20% fee charged by mturk.

Our current budget allowed us to annotate ~11,000 abstracts.
**If you are interested in funding this annotation effort, please contact Kenneth at txh710@psu.edu).**

## How to Cite?
```
@article{huang2020coda,
  title={CODA-19: Reliably Annotating Research Aspects on 10,000+ CORD-19 Abstracts Using a Non-Expert Crowd},
  author={Huang, Ting-Hao K. and Huang, Chieh-Yang and Ding, Chien-Kuang C. and Hsu, Yen-Chia and Giles, C Lee},
  journal={arXiv preprint arXiv:2005.02367},
  year={2020}
}
```

## Acknowledgements
This project is supported by the Huck Institutes of the Life Sciences' Coronavirus Research Seed Fund (CRSF) at Penn State University and the College of IST COVID-19 Seed Fund at Penn State University.
We thank the crowd workers for participating in this project and providing useful feedback. 
We thank VoiceBunny Inc. for granting a 20% discount for the voiceover for the worker tutorial video to support projects relevant to COVID-19.
We also thank Tiffany Knearem and Shih-Hong (Alan) Huang for reviewing our interfaces and the text used in our HITs.


## Misc

- [IST seed grants support tech projects related to COVID-19](https://news.psu.edu/story/619919/2020/05/13/research/ist-seed-grants-support-tech-projects-related-covid-19). May 13, 2020. PSU News.

- [Some discussion on the corner cases and FAQs.](https://twitter.com/windx0303/status/1258508964687253506)

- [Human and AI annotations aim to improve scholarly results in COVID-19 searches](https://news.psu.edu/story/616031/2020/04/17/research/human-and-ai-annotations-aim-improve-scholarly-results-covid-19). April 17th, 2020. Jordan Ford. PSU News.

- [Seed grants jump-start 47 interdisciplinary teams to conduct COVID-19 research
](https://news.psu.edu/story/615456/2020/04/14/research/seed-grants-jump-start-47-interdisciplinary-teams-conduct-covid-19). April 14, 2020. Sara LaJeunesse. PSU News.
