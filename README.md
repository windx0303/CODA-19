# CODA-19: COVID-19 Open Research Abstracting Dataset
CODA-19 is a human-annotated large-scale scientific abstract dataset, in which human annotators manually labeled all the text segments in each abstract with one of the following *information types*: **Background, Purpose, Method, Finding/Contribution, and Other**.

We teamed up with 200+ crowd workers on [Amazon Mechanical Turk](https://www.mturk.com/) to exhaustively label 10,966 abstracts, containing 103,978 sentences, which were further divided into 168,286 text segments, within 10 days.
These abstracts were randomly selected from the [COVID-19 Open Research Dataset (CORD-19)](https://www.semanticscholar.org/cord19).
The aggregated crowd label resulting an ~82% average label accuracy comparing against two sets of expert labels annotated on the same 129 abstracts, respectively.

## How did we do it?

## Why?

## JSON Schema

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
## How to Cite?

## Acknowledgements



