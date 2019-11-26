### Reproducibility of the main results

**Main dependencies**

- Python (3.6.7)
- Pytorch 1.0 (torch 1.0.1.post2, torchvision 0.2.2)
- numpy (1.16.2)
- scikit.learn (0.20.3)
- tqdm (4.31.1)
- torchsummary


**Results in main text**

We will discuss how to reproduce the main results in order of appearance in the main text.


The first steps are

1. clone the repository
  ```
  $ git clone https://github.com/ansuini/IntrinsicDimDeep.git
  ```  
2. change directory to IntrinsicDimDeep (this relative directory we will be referred as ROOT in what follows)

3. download and unzip in ROOT the data provided at this [link](https://figshare.com/s/8a039f58c7b84a215b6d)

Now you are ready to reproduce the results:

- **Figure 2: VGG-16-R on a custom dataset**

  You can plot the results of a typical run just by opening the notebook plots.ipynb in scripts/custom. 

  If you want to do the  training and analyse your results you can do as follows:

  - change directory into data/custom

  - generate the train/test split:
  
    ```
    $ python train_test_split.py
    ```  
  
  - change directory to scripts/custom 

  - launch finetuning:

    ```
    $ python finetune
    ```
  
    This will finetune a VGG-16 pre-trained on ImageNet. At the end it will also extract representations and save them in data/custom/results by default
  
- run the bash script
  
    ```
    $ ./run.sh
    ```
    
    This will analyse the extracted representations and generate the data (saving it by default) required for Figure 2 (there can be small fluctuations due to the number of epochs of training that you use and/or the random splitting between train and test).
    
  
    
    You can visualize your results opening the notebook plots.ipynb in scripts/custom,
  decommenting the line that specify to use your results.
    
    This pattern of usage will be maintained for all the experiments below.



- **Figure 3 (generality of the "hunchback" shape)**

  To show the results of a typical run

  - change directory to scripts/pretrained
  - open and run the notebook hunchback.ipynb

  To re-create the data from scratch:

  - ```
    $ ./last_hunchback.sh
    ```

- **Figure 4 (ID and generalization)**

  To show the results of a typical run

  - change directory to scripts/pretrained
  - open and run the last_hidden.ipynb

  To re-create the data from scratch:

  - ```
    $ ./last_hidden.sh
    ```



- **Figure 5 (Curvature)**

  To show the results of a typical run

  - change directory to scripts/pretrained
  - open and run the last_hidden_pca.ipynb

  To re-create the data from scratch:

  - ```
    $ ./last_hidden_pca.sh
    ```

- **Figure 6 (MNIST)**

  To show the results of a typical run

  - change directory to scripts/mnist
  - open and run the plot.ipybn

  To re-create the data the procedure is slightly longer than in the preceding cases:

  - train the small convolutional network on all the datasets (MNIST,MNIST* and MNIST+)

    ```
    $ ./train_all.sh
    ```

    You could do the three trainings separately:
  
    ```
    $ ./mnist.sh
    ```

    ```
    $ ./mnist_grad.sh
    ```

    ```
    $ ./mnist_shuffled.sh
    ```

  - analyze the results in the three cases:

    ```
    $ ./analyze_all.sh
    ```

    

    The epochs specified as parameters in analize_all.sh are by default the same used to train the different models.
  
    Anyway, for exploration purposes you can choose at which epoch to extract the data.

    ```
    $ python analyze.py --dataset yourdataset --epoch yourepoch
    ```

  -  The shuffled dataset (MNIST+) and the dataset perturbed with the luminosity gradient (MNIST*) are provided. Anyway, you can create a new shuffled dataset

    ```
    $ python create_shuffled_mnist.py --save 1
    ```
  
    and / or a new MNIST perturbed with a higher or lower stretching parameter lambda
    
    ```
    $ python create_mnist_with_gradient.py --save 1 --lambdavar yourlambda
    ```
