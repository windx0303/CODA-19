# CODA-19: COVID-19 Open Research Abstracting Dataset
CODA-19 is a human-annotated large-scale scientific abstract dataset, in which human annotators manually labeled all the text segments in each abstract with one of the following *information type*: **Background, Purpose, Method, Finding/Contribution, and Other**.
We teamed up with 200+ crowd workers on [Amazon Mechanical Turk](https://www.mturk.com/) to exhaustively label 10,966 abstracts, containing 103,978 sentences, which were further divided into 168,286 text segments, within 10 days.
These abstracts were extracted from the [COVID-19 Open Research Dataset (CORD-19)](https://www.semanticscholar.org/cord19).
The aggregated crowd label resulting an ~82% average accuracy comparing against two sets of expert labels annotated on the same 129 abstracts, respectively.

## How did we do it?

## Why?

## JSON Schema

```
{
  "paper_id": "9d9e41392d9817eeb79b994908433088e3aabff6",
  "metadata": {
    "title": "A second, non-canonical RNA-dependent RNA polymerase in SARS Coronavirus",
    "coda_data_split": "test",
    "coda_paper_id": 373,
    "coda_has_expert_labels": true,
    "subset": "custom_license"
  },
  "abstract": [
    {
      "original_text": "In ( þ ) RNA coronaviruses, replication and transcription of the giant B30 kb genome to produce genome-and subgenome-size RNAs of both polarities are mediated by a cognate membrane-bound enzymatic complex. Its RNAdependent RNA polymerase (RdRp) activity appears to be supplied by non-structural protein 12 (nsp12) that includes an RdRp domain conserved in all RNA viruses. Using SARS coronavirus, we now show that coronaviruses uniquely encode a second RdRp residing in nsp8. This protein strongly prefers the internal 5 0 -(G/U)CC-3 0 trinucleotides on RNA templates to initiate the synthesis of complementary oligonucleotides of o6 residues in a reaction whose fidelity is relatively low. Distant structural homology between the C-terminal domain of nsp8 and the catalytic palm subdomain of RdRps of RNA viruses suggests a common origin of the two coronavirus RdRps, which however may have evolved different sets of catalytic residues. A parallel between the nsp8 RdRp and cellular DNA-dependent RNA primases is drawn to propose that the nsp8 RdRp produces primers utilized by the primerdependent nsp12 RdRp.",
      "sentences": [
        [
          {
            "segment_text": "In ( þ ) RNA coronaviruses ,",
            "crowd_label": "background"
          },
          {
            "segment_text": "replication and transcription of the giant B30 kb genome to produce genome-and subgenome-size RNAs of both polarities are mediated by a cognate membrane-bound enzymatic complex .",
            "crowd_label": "background"
          }
        ],
        [
          {
            "segment_text": "Its RNAdependent RNA polymerase ( RdRp ) activity appears to be supplied by non-structural protein 12 ( nsp12 ) that includes an RdRp domain conserved in all RNA viruses .",
            "crowd_label": "background"
          }
        ],
        [
          {
            "segment_text": "Using SARS coronavirus , we now show that coronaviruses uniquely encode a second RdRp residing in nsp8 .",
            "crowd_label": "method"
          }
        ],
        [
          {
            "segment_text": "This protein strongly prefers the internal 5 0 - ( G/U ) CC-3 0 trinucleotides on RNA templates to initiate the synthesis of complementary oligonucleotides of o6 residues in a reaction whose fidelity is relatively low .",
            "crowd_label": "finding"
          }
        ],
        [
          {
            "segment_text": "Distant structural homology between the C-terminal domain of nsp8 and the catalytic palm subdomain of RdRps of RNA viruses suggests a common origin of the two coronavirus RdRps ,",
            "crowd_label": "finding"
          },
          {
            "segment_text": "which however may have evolved different sets of catalytic residues .",
            "crowd_label": "finding"
          }
        ],
        [
          {
            "segment_text": "A parallel between the nsp8 RdRp and cellular DNA-dependent RNA primases is drawn to propose that the nsp8 RdRp produces primers utilized by the primerdependent nsp12 RdRp .",
            "crowd_label": "finding"
          }
        ]
      ]
    }
  ],
  "abstract_stats": {
    "paragraph_num": "1",
    "sentence_num": "6",
    "segment_num": "8",
    "token_num": "186"
  }
}
```
## How to Cite?

## Acknowledgements



