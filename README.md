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


# Prerequisites

    1. Have a server you can use.
    2. Install libraries, keras and tensorflow.
    
1. Have a server you can use.

2. Install libraries, keras and tensorflow.

    Have the python version you want and pip installed.
    Install numpy.
    Install Tensorflow. Go here to follow the directions if you are not on a MacOS: https://www.tensorflow.org/install/
    For Mac users, ```pip install tensorflow-gpu``` will install the gpu version, which is what I used on my server.
    For Keras, you can go to this link: https://keras.io/#getting-started-30-seconds-to-keras. 
    To install you can either sudo pip install or pip install keras. On the other hand, you can use git:
    ``` git clone https://github.com/fchollet/keras.git ```
    ``` cd keras ```
    ``` sudo python setup.py install ```

# Steps

This pattern runs through the steps below. Check out the notebook for the code!

    1. Start a jupyter notebook.
    2. Run the notebook.
    3. Train a model.
    4. Analyze the result.

1. Start a jupyter notebook.

* From here you can choose to work in terminal while using python or download jupyter notebook to follow along:
       ```pip install jupyter notebook```
* Once that is installed you can enter ```jupyter notebook``` in your terminal and a notebook should pop up in your browser.
* If a notebook was did not automatically pop up, go to the url given in your terminal.
* The notebook containing the code for this pattern is in this [repository](https://github.com/MadisonJMyers/Training-a-Deep-Learning-Language-Model-Using-Keras-and-Tensorflow/tree/master/notebooks). 
* Download the notebooks, data and files into the folder you started jupyter notebook in. Once you do that it should appear in the browser. Click on it to open it up.

![](doc/source/images/Screen%20Shot%202017-12-11%20at%202.10.50%20PM.png)

2. Run the notebook.
 
When a notebook is executed, what is actually happening is that each code cell in
the notebook is executed, in order, from top to bottom.

Each code cell is selectable and is preceded by a tag in the left margin. The tag
format is `In [x]:`. Depending on the state of the notebook, the `x` can be:

* A blank, this indicates that the cell has never been executed.
* A number, this number represents the relative order this code step was executed.
* A `*`, this indicates that the cell is currently executing.

There are several ways to execute the code cells in your notebook:

* One cell at a time.
  * Select the cell, and then press the `Play` button in the toolbar.
* Batch mode, in sequential order.
  * From the `Cell` menu bar, there are several options available. For example, you
    can `Run All` cells in your notebook, or you can `Run All Below`, that will
    start executing from the first cell under the currently selected cell, and then
    continue executing all cells that follow.
* At a scheduled time.
  * Press the `Schedule` button located in the top right section of your notebook
    panel. Here you can schedule your notebook to be executed once at some future
    time, or repeatedly at your specified interval.
    
![](doc/source/images/Screen%20Shot%202017-12-11%20at%202.11.11%20PM.png)    
    
3. Train a model.

_See the notebook for further explanation on what is happening here._
* For this Code Pattern you'll need to go into the file where you downloaded everything.
* Make sure you have everything in the same folder.
* Now type ``` python transfer_learn.py ``` and push enter.
* The model should be running and generating text based on the yelp data it was given.

4. Analyze the result.

As you can see in the image below, you should expect to see text being generated with different diversities and then saved back into the weights. By this output you can see what different outputs are based on different diversities of text (more diverse vs less or more repetitive).

![](doc/source/images/Screen%20Shot%202017-12-07%20at%2011.16.22%20AM.png)

Congrats! Now you've learned how to generate text based on the data you've given it. Look out for the next Code Pattern which takes this generated data and learns how to detect the fake text vs the real, original text.

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
