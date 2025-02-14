# MLOps ARISA - Lesson 1 - Motivating the Principles of MLOps
## Introduction - What is MLOps?
According to the Continous Delivery Foundation, "MLOps could be narrowly defined as “the ability to apply DevOps principles to Machine Learning applications“ however [...], this narrow definition misses the true value of MLOps to the customer. Instead, we define MLOps as “the extension of the DevOps methodology to include Machine Learning and Data Science assets as first class citizens within the DevOps ecology“." ([CDF - MLOps Roadmap 2024](https://github.com/cdfoundation/sig-mlops/blob/main/roadmap/2024/MLOpsRoadmap2024.md "CDF - MLOps Roadmap 2024"))
In order words, the field of MLOps seeks to apply DevOps principles and methods, which are quite a bit more mature, to the fields of Data Science (DS) and Machine Learning (ML), which are comparatively in their infancy.   
This definition, however, might be difficult to understand at a glance, without already being familiar with the details and goals of both DevOps and ML.  
Rather than trying to explain, from a purely theoretical perspective, what MLOps is supposed to be, in this course we will be starting from what is the typical endpoint (no pun intended) of a data science project, namely a proof-of-concept (PoC) trained model. These typically come in the form of one or more jupyter notebooks, with a mix of data preprocessing and training code, as well as an inference step where predictions are made on unseen data, usually to evaluate model performance.  
To use a more everyday analogy, one which founder Dr John Elder of the industry leading DS and ML Consultancy firm Elder Research [^1], is fond of, DS and ML is akin to building a car engine, where MLOps then is constructing the rest of the car. Depending on the maturity of the organization in question, data engineers might be brought in on occasion, to build a proverbial drive-way (data infrastructure) and so on.  

As mentioned, since MLOps is still a relatively young field, there are no strictly agreed upon standards and principles that everyone follows when starting a DS/ML project. There are, however, several attempts at codifying general MLOps principles or roadmaps, including by the Continuous Delivery Foundation (see CDF - MLOps Roadmap 2024 above), and Marvelous MLOps (see [Marvelous MLOps Substack][Ref2] run by Başak Tuğçe Eskili and Maria Vechtomova, mainly using databricks but containing a goldmine of information about general MLOps principles and practices). In this course we will be presenting and following the latter, first laying out the principles of MLOps and then motivating them through code examples.  



### Supplementary Materials
**[^1]. Data Science Connect Keynote, John Elder - The Twin Crises of Science and How to Defeat Them.**  
https://www.elderresearch.com/resource/videos/the-twin-crises-of-science-and-how-to-defeat-them/  
(Backup link in case the original disappears:
 https://e.pcloud.link/publink/show?code=XZdg3KZ1QlaMytwvrREylDtNgWCLFxt8d8V)  
Dr Elder covers the first crisis, "most experimental findings are false", where MLOps attempts to address the second one, "implementation is too rare" (only 20% of models make it into production) *Critical to point out that the biggest challenge is not always technological or scientific, but getting stakeholder buy-in i.e. going from notebook to production*.  
Dr Elder mentions three key points to successful *real life* implementation:  
 * Address change management from the very start.  
 * (Form a close) team with stakeholders; gain trust with transparency; always seek their thriving.
 * Deliver a tool that fits seamless into their environment (i.e. fix a pain point, do not create a new one).  

An as a final point: **Success is impossible without implementation*!*  
Elder Research has a youtube channel and mailing list (https://www.elderresearch.com/insights/).

**2. Materials 2**
  

### Notes
Need to mention that machine learning and data science, at least in the beginning stages, is sometimes more of an art than a science which can be considered in conflict with certain aspects of devops, like agile.


## References  

* [1]: <https://github.com/cdfoundation/sig-mlops/blob/main/roadmap/2024/MLOpsRoadmap2024.md> "CDF - MLOps Roadmap 2024"
* [Ref2]: https://marvelousmlops.substack.com/