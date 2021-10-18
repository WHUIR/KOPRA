# KOPRA

Open resource for paper [Joint Knowledge Pruning and Recurrent Graph Convolution forNews Recommendation](chrome-extension://bocbaocobfecmglnmeaeppambideimao/pdf/viewer.html?file=https%3A%2F%2Fyuh-yang.github.io%2Fresources%2Fkopra.pdf)

## Requirement

> Python: 3.8  
> Pytoch: 1.6  
> cuda: 10.1  
> DGL: dgl-cu101  

## Dataset

We use a real-world news dataset provided by Microsoft Research Asia(MIND), 2020, and a norwegian News dataset, 2006(Adressa).
Moreover, we also disclose the knowledge graphs of these two datasets to support further research.
The KGs are in Graph.zip.

The raw datasets have been open source and can be obtained from the links mentioned in the paper.
And the logic of KG filtering we use is Liu's work in [NewsGraphRec](https://github.com/danyang-liu/NewsGraphRec).
