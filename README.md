# CODA-19: COVID-19 Open Research Abstracting Dataset
CODA-19 is a human-annotated large-scale scientific abstract dataset, in which human annotators manually labeled all the text segments in each abstract with one of the following *information types*: **Background, Purpose, Method, Finding/Contribution, and Other**. This annotation schema is adopted from [SOLVENT by Chan et al. (CSCW'18)](https://dl.acm.org/doi/10.1145/3274300), with minor changes.

We teamed up with 248 crowd workers on [Amazon Mechanical Turk](https://www.mturk.com/) to exhaustively label **10,966 abstracts**, containing 103,978 sentences, which were further divided into 168,286 text segments, within **10 days** (from April 19, 2020 to April 29, 2020, including the time for worker training and post-task survey).
These abstracts were randomly selected from the [COVID-19 Open Research Dataset (CORD-19)](https://www.semanticscholar.org/cord19).
The aggregated crowd labels resulted in **a label accuracy of 82% and an Cohen's kappa coefficient (κ) of 0.74**, comparing against biomedical expert labels annotated on 129 abstracts.

The following is an actual abstract (you can see the paper [here](https://www.nature.com/articles/s41422-020-0305-x)) annotated by crowd workers in CODA-19. 

![Example Annotation](https://crowd.ist.psu.edu/CODA19/img/example.JPG)

## Why create these annotations?

This work was developed upon the long history of research on understanding scientific papers at scale. 
In short, the rapid acceleration in new coronavirus literature makes it hard to keep up with.
So we highlighted the papers with its **Background, Purpose, Method, Finding/Contribution, and Other**.

People can use these data to build an automated annotator to label the remaining papers in [CORD-19](https://pages.semanticscholar.org/coronavirus-research) and, more importantly, future papers.
This type of annotation can also be useful for various BioNLP tasks.

## Annotation Schema

- Background
  - Why is this problem important?
  - What relevant works have been done before?
  - What is still missing in the previous works?
  - What are the high-level research questions?
  - How might this help other research(ers)?
- Purpose	
  - What specific thing(s) do the researchers want to do?
  - What specific knowledge do the researchers want to know?
  - What specific hypothesis do the researchers want to test?
- Method	
  - How did the researchers do it or find out?
  - What are the procedures and steps of the research?
- Finding/Contribution	
  - What did the researchers find out?
  - Did the proposed methods work?
  - Did the thing behave as the researchers expected?
- Other	
  - Text fragments that do NOT fit into any of the four categories above.
  - Text fragments that are NOT part of the article.
  - Text fragments that are NOT in English.
  - Text fragments that contains ONLY reference marks (e.g., "[1,2,3,4,5") or ONLY dates (e.g., "April 20, 2008").
  - Captions for figures and tables (e.g. "Figure 1: Experimental Result of ...", or "Table 1: The Typical Symptoms of ...")
  - Formatting errors.
  - I really don't know or I'm not sure.

## Data Selection & Preprocessing

#### Paper Filtering

#### Tokenization, Sentence Segmentation, and Text Segmentation

#### Language Identification

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

## How much did it cost?
Annotating one abstract costs **$3.2** on average with our setup. This cost includes the payments for workers and the 20% fee charged by mturk.

Our current budget allowed us to annotate ~11,000 abstracts.
**If you are interested in funding this annotation effort, please contact Kenneth at txh710@psu.edu).**

## How to Cite?

## Media Coverage

- [Human and AI annotations aim to improve scholarly results in COVID-19 searches](https://news.psu.edu/story/616031/2020/04/17/research/human-and-ai-annotations-aim-improve-scholarly-results-covid-19). April 17th, 2020. Jordan Ford. PSU News.

- [Seed grants jump-start 47 interdisciplinary teams to conduct COVID-19 research
](https://news.psu.edu/story/615456/2020/04/14/research/seed-grants-jump-start-47-interdisciplinary-teams-conduct-covid-19). April 14, 2020. Sara LaJeunesse. PSU News.


## Acknowledgements
This project is supported by Coronavirus Research Seed Fund (CRSF) and College of IST COVID-19 Seed Funding, both at the Penn State University.
We thank the crowd workers for participating in this project and providing useful feedback.
We thank Tiffany Knearem and Shih-Hong (Alan) Huang for reviewing our interfaces and the text used in our HITs.
We thank VoiceBunny.com for granting a 20% discount for the voiceover for the worker tutorial video to support projects relevant to COVID-19.
We also thank the staff members in the Finance Office in IST for acting quickly, allowing us to start the project rapidly.
