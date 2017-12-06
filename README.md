## Training a Deep Learning Language Model Using Keras and Tensorflow
A code pattern focusing on how to train a machine learning language model while using Keras and Tensorflow.

This code pattern will guide you through installing Keras and Tensorflow, downloading yelp data and training a language model using recurrent neural networks, or RNNs, to generate text.

Using deep learning to generate information is a hot area of research and experimentation. This approach has seemingly boundless amounts of areas it can be applied to, whether it be to solutions or areas of intrigue. One of these areas is text generation, which is what we look at in this pattern. Though training a RNN model to generate reviews tows the line between dangerous and productive, it's important to look at so that we can prevent any misuse of the same approach. Misuse would incude fake reviews or mimiking a certain person's style. Fake reviews specifically are a very real problem for companies such as Amazon and Yelp, who rely on reviews to vouch for their products and businesses that are featured. As of writing this, it is very easy for businesses to pay for fake, positive reviews, which ultimately end up elevating their sales and revenue. Unfortunately this leads users to places and products fradulently and can potentially lead to someone having a negative experience or worse. In order to combat these abuses and illegal activity, we must first understand how to generate fake reviews and build a dataset with them. Only then can we use our knowledge of generating text to prevent generated text. This approach follows the University of Chicago's paper [Automated Crowdturfing Attacks and Defenses in
Online Review Systems](https://arxiv.org/pdf/1708.08151.pdf).

The original yelp data used in this code pattern can be found [here](https://www.kaggle.com/c/yelp-recruiting/data) as well as in this repository under [data/](https://github.com/MadisonJMyers/Training-a-Deep-Learning-Language-Model-Using-Keras-and-Tensorflow/tree/master/data). 

![](doc/source/images/architecture.png)

## Flow

1. Locally, or on a server, download the yelp data.
2. Download and install Keras and Tensorflow.
3. Learn what you can do using Keras.
4. Train a language model to generate text.
5. Evaluate and learn how else this model can be applied.

## Included components

* [Jupyter Notebook](http://jupyter.org/): An open source web application that allows you to create and share documents that contain live code, equations, visualizations, and explanatory text.
* [Keras](https://keras.io/): The Python Deep Learning library.
* [Tensorflow](https://www.tensorflow.org/): An open-source software library for Machine Intelligence.

## Featured technologies

* [Cloud](https://www.ibm.com/developerworks/learn/cloud/): Accessing computer and information technology resources through the Internet.
* [Data Science](https://medium.com/ibm-data-science-experience/): Systems and scientific methods to analyze structured and unstructured data in order to extract knowledge and insights.
* [Python](https://www.python.org/): Python is a programming language that lets you work more quickly and integrate your systems more effectively.

# Watch the Video
TBD

# Steps

This pattern runs through the steps below. Check out the notebook for the code!

    1. Download yelp data.
    2. Download and install Keras and Tensorflow (and their dependencies).
    3. Start a jupyter notebook.
    4. Follow the notebook provided to run a language model to generate text.
    5. Learn where else this RNN model can be applied.

1. Download yelp data.

Go to the kaggle competition link here: https://www.kaggle.com/c/yelp-recruiting/data and download your data locally or on a server.

2. Download and install Keras and Tensorflow (and their dependencies).

First you must make sure you have the prerequisits:

    Have the python version you want and pip installed.
    Install numpy.
    Install Tensorflow. Go here to follow the directions if you are not on a MacOS: https://www.tensorflow.org/install/
    For Mac users, ```pip install tensorflow-gpu``` will install the gpu version, which is what I used on my server.
    For Keras, you can go to this link: https://keras.io/#getting-started-30-seconds-to-keras. 
    To install you can either sudo pip install or pip install keras. On the other hand, you can use git:
    ```git clone https://github.com/fchollet/keras.git```
    ```cd keras
    sudo python setup.py install```

    
3. Start a jupyter notebook.

       From here you can choose to work in terminal while using python or download jupyter notebook to follow along:
       ```pip install jupyter notebook```
       The notebook containing the code for this pattern is in the repository: 
       Once that is installed you can enter ```jupyter notebook``` in your terminal and a notebook should pop up in your browser.
       If a notebook was not created, go to the url given in your terminal.
   
4. Follow the notebook provided to run a language model to generate text.
5. Learn where else this RNN model can be applied.


## Sample output

Use a final _Analyze the results_ or _Conclusion_ step above, plus sample output to wrap up the code pattern for a developer running it and also for people that are only going to read the README. Sample output format will vary depending on the code pattern technology used.


# Links

* [Create Data Science Experience Notebooks](https://datascience.ibm.com/docs/content/analyze-data/creating-notebooks.html)
* [Jupyter Notebook](http://jupyter.org/): An open source web application that allows you to create and share documents that contain live code, equations, visualizations, and explanatory text.
* [Keras](https://keras.io/): The Python Deep Learning library.
* [Tensorflow](https://www.tensorflow.org/): An open-source software library for Machine Intelligence.


# Learn more

* **Data Analytics Code Patterns**: Enjoyed this Code Pattern? Check out our other [Data Analytics Code Patterns](https://developer.ibm.com/code/technologies/data-science/)
* **AI and Data Code Pattern Playlist**: Bookmark our [playlist](https://www.youtube.com/playlist?list=PLzUbsvIyrNfknNewObx5N7uGZ5FKH0Fde) with all of our Code Pattern videos
* **Data Science Experience**: Master the art of data science with IBM's [Data Science Experience](https://datascience.ibm.com/)
* **Spark on IBM Cloud**: Need a Spark cluster? Create up to 30 Spark executors on IBM Cloud with our [Spark service](https://console.bluemix.net/catalog/services/apache-spark)

# License
[Apache 2.0](LICENSE)
