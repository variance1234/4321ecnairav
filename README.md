#Supplementary Material Info

Due to the file size limit and for reviewers’ convenience, we only upload small documents with the submission as supplementary material. The Weights and the Code (which are too large to submit as supplementary material) are put in an anonymous GitHub repository: https://github.com/variance1234/4321ecnairav
Material list:

Name(s) |Description|Location
---|---|---
Survey Questions.pdf| The survey questions| Supplementary, GitHub
Training configuration.pdf| The training configuration for 6 networks| Supplementary, GitHub
Relevant-AI-Paper.csv| Relevant-Non-AI-Paper.csv| The list of relevant AI and Non-Ai papers in our survey| Supplementary, GitHub
analysis_result.csv, analysis_raw.csv| The analysis result for our experiences.| Supplementary, GitHub
Weights| Folder contains the weights of the most extreme models in our experiments| GitHub
Code| The source code| GitHub


## Paper list files info
**Relevant-AI-Papers.csv**
This file contains the list of AI papers that we found relevant to our study.
**Relevant-Non-AI-Papers.csv**
This file contains the list of Non-AI papers that we found relevant to our study.

+Conference: The conference
+Title: Paper title
+Relevant to our study? Does the work train deep learning networks?
+Do they do multiple identical runs? Do the work report multiple identical runs?

## Analysis files info
**analysis_result.csv**
This file contains the main analysis result of the experimental runs

+backend: core library
+backend_version: core library version
+cuda_version: cuda version
+cudnn_version: cudnn version
+network: network	
+random_seed	: if 1 ->  fixed-seed, if -1 -> random seed
+stopping_type: selection criterion	
+no_try: number of identical runs	
+max_accuracy_diff:	largest overall accuracy difference
+max_accuracy: overall accuracy of the most accurate run	
+min_accuracy: overall accuracy of the least accurate run	
+std_dev_accuracy: overall accuracy standard deviation	
+mean_accuracy: mean overall accuracy	
+max_diff_label: the class index with the largest accuracy gap for this experimental set	
+max_per_label_acc_diff: largest per-class accuracy difference	
+max_label_accuracy	: largest per-class accurate for the class 
+min_label_accuracy: lowest per-class accurate for the class	
+no_samples_max_diff: number of test samples for class (max_diff_label)	
+max_std_label: the class index with the largest per-class accuracy standard deviation for this experimental set		
+max_per_label_acc_std: the per-class accuracy standard deviations	
+no_samples_max_std: number of test samples for class (max_std_label)	
+max_convergent_diff: largest convergence time difference	
+max_convergent: convergence time of the slowest run (most time)	
+min_convergent: convergence time of the fastest run (least time)	
+std_dev_convergent	: standard deviation of convergence times
+mean_convergent: average convergence time	
+max_convergent_diff_epoch: largest number of epochs to convergence difference	
+max_convergent_epoch: largest number of epochs to convergence	
+min_convergent_epoch: smallest number of epochs to convergence	
+std_dev_convergent_epoch: standard deviation of the number of epochs to convergence
+mean_convergent_epoch: average number of epochs to convergence

**analysis_raw.csv**
This file contains the overall accuracy of all training runs

+backend: core library
+backend_version: core library version
+cuda_version: cuda version
+cudnn_version: cudnn version
+network: network	
+random_seed	: if 1 ->  fixed-seed, if -1 -> random seed
+stopping_type: selection criterion	
+try: run index
+accuracy: overall accuracy of the model
+convergent: time to convergence of this run
+convergent_epoch: number of epochs to convergence
