# ContentWise Impressions

This is the repository of our article "ContentWise Impressions: An industrial dataset with impressions included" on CIKM 2020.
The article is under review at the moment.

## How to download the dataset?
You can obtain the link to download the dataset by filling this [survey](https://forms.office.com/Pages/ResponsePage.aspx?id=K3EXCvNtXUKAjjCd8ope6_zxBj9DRzpKnC4jkclZQupUQ0szOVhTQ1FCT0tZSEw1T1g0RzVBRVhSSC4u).

Filling the survey is completely optional, and it won't block you from getting the link to download the dataset. 

After you receive the dataset link, download the zip file and decompress it on your local environment. 

You'll find a `README.md` file, that includes information about the dataset, authors,
license, and more. You'll also find the `data` folder. Inside this folder you'll find the dataset (`interactions`, 
`impressions-direct-link`, and `impressions-non-direct-link`) alongside the URM splits that we used in our experiments.
Moreover, if you wish to run the scripts inside the repository, you'll need the whole `data` folder.

## What about your results?

You can download the results of our experiments on this [link](https://polimi365-my.sharepoint.com/:u:/g/personal/10565493_polimi_it/Ec_qXpgaTHFMrDEiNzqsIfMB6faqm6_8JJoVNU5fFTcJpg?e=tSokYW). 
There you'll find two folders: `statistics` and `result_experiments`. The first folder contains the statistical features 
of the dataset alongside several plots, including others that didn't make it into the paper. The second folder contains 
all the fine-tuned trained recommender models.

_Note_: As we exported the models for several recommenders, the results folder takes approximately 2GB on disk.

In this repository we provide several tools to load and use the dataset. We strongly recommend you to go through the 
[installation](#installation) and [using the repo](#using-the-repo) sections to know which scripts we provide and 
how to run them.

## Installation
Note: that this repository requires Python 3.7

First we suggest you create an environment for this project using conda. We have tested the installation procedures
on Linux 64-bits (Ubuntu 18.04), macOS Catalina 10.15, and Windows 10. 

First, install *miniconda*, instructions on how to install miniconda are found in 
[Their docs](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

Second, clone this repository, checkout, and install the environment with the following:
```console
git clone https://github.com/ContentWise/contentwise-impressions.git
cd contentwise-impressions
conda env create -f environment.yml
conda activate contentwise-impressions
```

Now, depending on your platform, there are special installation procedures that you need to perform. If you:
 * Are running on linux, then see [Linux dependencies](#linux-install-more-dependencies).
 * Are running on macOS, then see [macOS dependencies](#macos-install-more-dependencies).
 * Are running on Windows, then see [Windows dependencies](#windows-install-more-dependencies).
 
After you have performed your environment specific steps, continue to [Compiling Cython](#compiling-cython-code). 
  
### [Linux] Install more dependencies
At this point, having installed all dependencies, you have to compile all Cython algorithms.

In order to compile you must first have installed: _gcc_ and _python3 dev_. Under Linux those can be installed with the 
following commands:
```console
sudo apt install gcc 
sudo apt-get install python3-dev
sudo apt-get install libopenblas-base libopenblas-dev
```

Now, continue to [Compiling Cython](#compiling-cython-code). 

### [macOS] Install more dependencies
You must download Xcode, and the command-line tools in order to have a C compiler installed on your system. More 
information about Xcode is found on [Apple docs](https://developer.apple.com/xcode/).

Now, continue to [Compiling Cython](#compiling-cython-code). 

### [Windows] Install more dependencies
If you are using Windows as operating system, the installation procedure is a bit more complex. You may refer 
to [THIS](https://github.com/cython/cython/wiki/InstallingOnWindows) guide.

Continue to [Compiling Cython](#compiling-cython-code).  

### Compiling Cython code

Now you can compile all Cython algorithms by running the following command. The script will compile within the current 
active environment. The code has been developed for Linux and Windows platforms. During the compilation you may see 
some warnings. These are expected 

```console
(contentwise-impressions): python run_compile_all_cython.py
```

### Place the data
Now that you have the environment set, download the dataset and the splits. Please place the `data` folder inside the 
repository folder.

After you've done this, you're ready to [use the repo](#using-the-repo).

## Using the repo.
We have provided several python scripts that uses the dataset in different ways. 

In the following sections we describe each script that we provide. 

### Generating URM splits
*Prerequisites*: You need to have the environment fully installed.

*NOTE*: On our tests, this process consumes up to `16GiB` of RAM. Please ensure to have these resources or use our
 splits.

In order to download the data and generate the URM splits that we used in our experiments, you must use the 
`run_generate_splits.py`. 

- If it's run without arguments, it will download the *interactions*, *interacted impressions*, and *non-interacted impressions*.
- If it's run with the `-i` or `--items` arguments, it will download the dataset and will generate three URM splits of the interactions *Train*, *Validation*, and *Test*. Using a proportion of 0.7, 0.1, and 0.2, respectively. Users are rows and *Items* are columns. 
- If it's run with the `-s` or `--series` arguments, it will download the dataset and will generate three URM splits of the interactions *Train*, *Validation*, and *Test*. Using a proportion of 0.7, 0.1, and 0.2, respectively. Users are rows and *Series* are columns.  
 
Examples:
```bash
(contentwise-impressions): python run_generate_splits.py -i -s
```
 
### Tuning Hyper-parameters of recommendation algorithms
*Prerequisites*: You need to have the environment fully installed, and the data splits, preferably.

*NOTE*: Depending on your environment and available resources, the process could get killed because of insufficient
memory. We used an _r4.4xlarge_ Linux Amazon EC2 Instance to run our experiments. It had 16vCPUs and 128 GiB of RAM.
However, we utilized this type of instance to run several experiments on parallel. By our own calculations, running
each recommender should take less than `20GiB` of RAM using eight cores if the evaluation is done in parallel. Execution
times for different recommenders vary.

In order to tune the hyper-parameters of several recommendation algorithms, you must use the 
`run_hyperparameter_tuning.py` script. You need to provide the `-t` or `--tune_recommenders` arguments in order to
 make it run. This is to ensure that you're willing to run the hyperparameter tuning. 
  
We ran the experiments using the following recommenders:
* Random: recommends a list of random items,
* TopPop: recommends the most popular items,
* ItemKNN: Item-based collaborative KNN,
* RP3beta: collaborative graph-based algorithm with re-ranking,
* PureSVD: SVD decomposition of the user-item matrix,
* Impressions MatrixFactorization BPR (BPRMF): machine learning based matrix factorization optimizing ranking with BPR,
  with the possibility to sample negative items at random, inside the impressions, or outside the impressions.

Examples:
```bash
(contentwise-impressions): python run_hyperparameter_tuning.py -t
```  

### Gathering results
*Prerequisites*: You need to have the environment fully installed, the data splits, and the `result_experiments` folder
 in the repository folder. 

The `run_results_gathering.py` script outputs a table with the hyperparameter tuning results. You need to provide the
`-s` or `--show_results` arguments in order to make it run. This is to ensure that you're willing to run the script.
 
Examples:
```bash
(contentwise-impressions): python run_results_gathering.py -s
```  

### Obtain statistics of the dataset
*Prerequisites*: You need to have the environment fully installed, and the dataset saved (not necessarily with the
 splits).

In order to generate the statistics of the dataset, we provide a jupyter notebook, `notebook_generate_statistics.ipynb`,
that lets you to generate several statistics of the dataset on the same place.

In order to run the code.
```bash
(contentwise-impressions): jupyter lab --no-browser
```  

Inside the notebook, just run the different sections to obtain different statistics. We provide documentation of what
 kind of statistics we calculate. All the statistics and plots are generated into the `statistics` folder. This
  notebooks generates all of the plots, numbers and figures that we used in the paper.
 
### Run the tests
*Prerequisites*: You need to have the environment fully installed, and the dataset saved (not necessarily with the
 splits). 

We provide consistency tests of the dataset. It will check several properties of the dataset that are reported in the
 paper.
 
We use `pytest` as test runner. To run the tests is just as easy as to run the following:
```bash
(contentwise-impressions): pytest test_dataset_consistency.py --verbose --color=yes
```  

This command doesn't write any report, instead it shows on the console the results of the tests in a PASS/FAIL fashion.

## If you run into issues or want to ask us something
Please, don't hesitate to let us know by opening an issue on the 
[Issue Tracker](https://github.com/ContentWise/contentwise-impressions/issues/new). We highly appreciate your feedback.

## Closing remarks
Thanks for using ContentWise Impressions, this repo and continue our work. We hope that it's useful for your purposes.

## Disclaimer
This is not an official ContentWise product.

## Contact information
For help or issues using ContentWise Impressions, please submit a GitHub issue.

For personal communication related to ContentWise Impressions, please contact:

- Fernando Benjamín Pérez Maurera ([fernando.perez@contentwise.com](mailto:fernando.perez@contentwise.com) or [fernandobenjamin.perez@polimi.it](mailto:fernandobenjamin.perez@polimi.it))
- Maurizio Ferrari Dacrema ([maurizio.ferrari@polimi.it](mailto:maurizio.ferrari@polimi.it)).
- Lorenzo Saule ([lorenzo.saule@gmail.com](mailto:lorenzo.saule@gmail.com)).
- Mario Scriminaci ([mario.scriminaci@contentwise.com](mailto:mario.scriminaci@contentwise.com)).
- Paolo Cremonesi ([paolo.cremonesi@polimi.it](mailto:paolo.cremonesi@polimi.it)).
