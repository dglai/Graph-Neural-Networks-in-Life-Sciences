# KDD 2022 Hands-on Tutorial: Graph Neural Networks in Life Sciences: Opportunities and Solutions

## Abstract
Graphs (or networks) are ubiquitous representations in life sciences and medicine, from molecular interactions maps, signaling transduction pathways, to graphs of scientific knowledge , and patient-disease-intervention relationships derived from population studies and/or real-world evidences. Recent advance in graph machine learning (ML) approaches such as graph neural networks (GNNs) has transformed a diverse set of problems relying on biomedical networks that traditionally depend on descriptive topological data analyses. Small- and macro- molecules that were not modeled as graphs also saw a bloom in GNN-based algorithms improving the state-of-the-art performance for learning their properties. Comparing to graph ML applications from other domains, life sciences offer many unique problems and nuances ranging from graph construction to graph-level, and bi-graph-level supervision tasks.

The objective of this tutorial is twofold. First, it will provide a comprehensive overview of the types of biomedical graphs/networks, the underlying biological and medical problems, and the applications of graph ML algorithms for solving those problems. Second, it will showcase four concrete GNN solutions in life sciences with hands-on experience for the attendees. These hands-on sessions will cover: 1) training and fine-tuning GNN models for small-molecule property prediction on atomic graphs, 2) macro-molecule property and function prediction on residue graphs, 3) bi-graph based binding affinity prediction for protein-ligand pairs, and 4) organizing and generating new knowledge for drug discovery and repurposing with knowledge graphs. This tutorial will also instruct the attendees to develop in two extensions of the software library Deep Graph Library (DGL), including DGL-lifesci and DGL-KE, so that they could jumpstart their own graph ML journey to advance life science research and development.

## Outline

The tutorial introduces to data science researchers and practitioners graph neural network (GNN) based approaches applied to various problems in biomedical sciences and healthcare.  The tutorial first provides an overview of the various opportunities in leveraging GNNs for small molecules, macromolecules and biomedical knowledge graphs. The four hands-on activities will provide the participants a diverse set of biomedical problems and in particular how to deploy a GNN-based library for these applications leading to biological phenotype prediction, interaction prediction, affinity prediction and drug discovery.

The tutorial will be broken up into the following five sections:

*Section 1: Overview of Graph ML in biomedical science*. This section describes different types of graphs commonly used in biomedical sciences and how graph-based machine learning approaches like GNNs can be leveraged. In particular, we will cover single-entity biomedical networks including gene regulatory network and protein-protein interaction networks, as well as multi-entity networks such as knowledge graphs of proteins, genes, diseases, symptoms, and drugs. This section also introduces graph representations for small and large molecules such as organic compounds and proteins, which can be modeled as independent graphs of atoms and residues, respectively. [section format: slides] 


*Section 2: Making sense of small molecules with GNNs*. This section demonstrates how to develop end-to-end graph-based ML pipeline for molecular property prediction. The pipeline first covers how to construct features from atom graphs for small organic compounds. Then, it will cover two use cases using DGL-lifesci command-line interface: 1) training a GNN for molecular property prediction from scratch, and 2) fine-tuning a pre-trained GNN for molecular property prediction. [section format: hands-on with Jupyter Notebook]


*Section 3:* *Making sense of macro-molecules with GNNs*. This section demonstrates how to use GNNs to predict properties for macro-molecules including RNAs and proteins. We will cover two hands-on case studies: 1) Prediction of COVID-19 mRNA vaccine degradation with GCN, and 2) protein function prediction using an equivariant GNN on graphs of amino acid residues. [section format: hands-on with Jupyter Notebook]


*Section 4: Going beyond single graph, bi-graph based binding affinity prediction for protein-ligand pairs*. This section demonstrates a case study for making predictions between a pair of graphs. Protein-ligand binding affinity prediction is important for candidate drug screening during the early stage of drug discovery. We demonstrate how PotentialNet can be used for this task, as well as a novel molecular data anonymization procedure for protecting IP of molecular structures. [section format: hands-on with Jupyter Notebook]


*Section 5: Organizing and generating new knowledge for drug discovery and repurposing with knowledge graphs (KGs).* This section showcases another application of graphs in life sciences by employing large-scale KGs to organize the information from diverse medical sources and make prediction on these KGs. KG is a directed heterogeneous multigraph whose node and relation types have domain-specific semantics. We will review three approaches to construct such medical KGs 1) mining medical documents and publications 2) processing and stitching together different KGs coming from various medical databases 3) converting relational databases to KGs.  We will show examples detailing how to construct such KGs. The resulting KGs store information efficiently and can be used for KG completion, drug repurposing, and question answering among other tasks. We will review notebooks showcasing how to use the KGs and graph ML to make predictions in these KGs. We also will explain common objectives used for KG completion. [section format: hands-on with Jupyter Notebook]

## Instructions for the hands-on sessions:

This workshop requires a Jupter Notebook and related data. For the purposes of this tutorial, we will be using AWS to set up our environment.

Within the AWS environment, we will use

- SageMaker notebook instance: An Amazon SageMaker notebook instance is a machine learning (ML) compute instance running the Jupyter Notebook App. SageMaker manages creating the instance and related resources. Use Jupyter notebooks in your notebook instance to prepare and process data, write code to train models, deploy models to SageMaker hosting, and test or validate your models.
- Neptune database instance: Amazon Neptune is a fast, reliable, fully managed graph database service that makes it easy to build and run applications.
- Neptune ML: Amazon Neptune ML is a new capability of Neptune that uses Graph Neural Networks (GNNs), a machine learning technique purpose-built for graphs, to make easy, fast, and more accurate predictions using graph data. With Neptune ML, you can improve the accuracy of most predictions for graphs by over 50% when compared to making predictions using non-graph methods.
- IAM execution roles for SageMaker: An IAM role is an IAM identity that you can create in your AWS account that has specific permissions you can use with your SageMaker notebook.
- S3 Bucket: Amazon Simple Storage Service (Amazon S3) is an object storage service offering industry-leading scalability, data availability, security, and performance.
As part of this workshop, this AWS environment has already been set up for you via AWS Event Engine.

Head to https://dashboard.eventengine.run/login  and enter the event engine hash `f843-10d7089734-9a`. You will be asked to login with an email to receive the OTP to get an AWS Account. Follow the instructions on the website

1. Click on AWS Console
2. On the window popup, select `Open Console`. This will open an AWS Console. The AWS Management Console is a browser-based GUI for Amazon Web Services (AWS). Through the console, a customer can manage their cloud computing, cloud storage and other resources running on the Amazon Web Services infrastructure.
3. On the top right, ensure `N.Virginia (us-east-1)` is selected. If for any reason, you are logged into the different region, please switch to N.Virginia.
4. On the top of the Console, type in `SageMaker` in the search bar
5. On the left sidebar, head to `Notebook` > `Notebook instances`
6. You will see an instance set up. Select `Open Jupyter` on the right side of the page under Actions. This will open a Jupyer notebook interface hosted on Amazon SageMaker
7. Time to begin with your workshop!
