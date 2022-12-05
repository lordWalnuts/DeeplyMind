---
title: "Classifying Peaky Blinders Characters using fast.ai"
author: "Koushik Balakrishnan"
summary: "Using fast.ai and resnet50 to classify images"
series: ["fast.ai course"]
# weight: 1
# aliases: ["/fastai-installation"]
tags: ["fast.ai","image classification","fast.ai course"]
date: 2022-12-02
comments: true  
showWordCount: true
showBreadCrumbs: false
---

# Intro 

We'll try to build an image classification model that identifies each member of the Peaky Blinders family. 

This is based on Lesson 1 of the ["Practical Deep Learning for Coders 2022"](https://course.fast.ai/) course by [fast.ai](https://www.fast.ai/)

(Also part of my Fast.ai Course series where I document my journey going through the Fast.ai course)

I view the process of training this model as three pieces:
1. Collecting the Data
2. Loading the Dataset into `Dataloaders`
3. Training the model

Before diving in, we need to set up the machine by installing necessary libraries 

The completed Google Colab notebook is linked in the Resources section

# Setting up Google Colab

I use Google Colab because I find it intuitive and can connect it with Google Drive to save the notebooks and data. 

You can find detailed information on how to set up Jupyter notebooks on different platforms [here](https://course.fast.ai/Lessons/lesson1.html)

Open a new Notebook and in the first code block, enter the following code

```python
!pip install -Uqq fastai
!pip install duckduckgo_search
```


- We prefix **!** When using bash commands. It's mostly used to install libraries and copy/move files around in our use case
- We install **fastai** and **duckduckgo_search** libraries where the latter is used to download images as we shall see in the next section

*Hint*: Don't forget to change the runtime of your notebook to GPU. You do that by *Runtime > change runtime type > GPU*

# Collecting the Data

### Downloading Data
- First import the following modules
```python
from duckduckgo_search import ddg_images
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *
```
- Then we define the **search_images** function
```python
def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term,max_results=max_images)).itemgot('image')
```
L is just a fast.ai component which returns a List

- Next we define our search terms and download the images. I use two Lists, the first list downloads the images the actors in Peaky Blinders set and the other downloads general images of actors
- This would be ideal since we would get images from different perspective and time period. This would force the model to learn from unique facial features 
```python
search_terms_1 = ['Thomas Shelby','Arthur Shelby','Ada Shelby','Polly Gray','Michael Gray','John Shelby']
search_terms_2 = ['Cillian Murphy','Paul Anderson','Sophie Rundle','Helen McCrory','Joe Cole','Finn Cole']

path = Path('peaky_blinders_family')
from time import sleep

for index in range(0,len(search_terms_1)):
  dest = (path/search_terms_1[index])
  dest.mkdir(exist_ok=True,parents=True)
  download_images(dest,urls=search_images(f'{search_terms_1[index]} peaky blinders photo',max_images=30))
  sleep(10)
  download_images(dest,urls=search_images(f'{search_terms_2[index]} actor photo',max_images=30))
  resize_images(path/search_terms_1[index],max_size=400,dest=path/search_terms_1[index])
```
	- sleep(10) is used to not overload the server
	- we resize the images to 400 since it's the most suited size for GPU training. 

### Cleaning the Data
- There's a high chance that some the photos are broken or corrupted. We can use **verify_images** function to easily check for them and remove them using **unlink** method
```python
path = Path('peaky_blinders_family')
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed) # to check number of failed items
```
Now we have our Dataset downloaded and cleaned

### Moving the dataset to Google drive (Optional)

If you plan on using this dataset again in the future, you should consider moving it to Google Drive. You could easily download it from there and also use it in other Colab Notebooks
```python
from google.colab import drive
drive.mount('/content/drive')
```
	You will be prompted to sign in with your google account.
	
Create a folder called dataset in your drive and use the following command to copy it to google drive
```python
!cp peaky_blinders_family/ drive/MyDrive/dataset/
```
# Loading the Dataset into `Dataloaders`

The following is from the [Kaggle Notebook](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data) for lesson 1:

*"To train a model, we'll need DataLoaders, which is an object that contains a training set (the images used to create a model) and a validation set (the images used to check the accuracy of a model -- not used during training)."*


- First we create a `Datablock` with various parameters and then pass it to `Dataloaders` object
```python
dls = DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2,seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192,method='squish')]
)
```
Here what each of the `DataBlock` parameters means:

    blocks=(ImageBlock, CategoryBlock),

Represents the Input and output types. In this case our input is an image and output is category(Tommy or Arthur etc.)

    get_items=get_image_files, 

gets the image files

    splitter=RandomSplitter(valid_pct=0.2, seed=42),

Splits the dataset into Training and Validation set. Training set are used to train the model and Validation set is used to test the accuracy of the model. In this case 0.2% of the dataset are Validation set

    get_y=parent_label,

Sets the output value. In this case we just use the folder name of the parent folder(Cillian Murphy etc.).

    item_tfms=[Resize(192, method='squish')]

Resizes the images by squishing it.

- We then call `Dataloaders` on the `Datablock` dls
```python
dls=dls.dataloaders(path,bs=42)
```
`bs` defines how many samples per batch to load.
	
*Hint*: If any of the parameters are confusing, you can always read about them in the [fast.ai documentation](docs.fast.ai). For example, [here](https://docs.fast.ai/data.load.html) is the page for `dataloaders` class 

# Training the Model
We can use one of the many pretrained models available to fine tune our model. For our task we shall use the [`resnet50`](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) model.

Fine-Tuning saves a lot of time by initializing the downloaded model with the set of weights that works the best. 

```python
learn = vision_learner(dls,resnet50,metrics=error_rate)
learn.fine_tune(5)
```
I used 5 epochs as it gives me the least possible error rate through trial and error. I also tried other models such as `resnet18` but the one we chose performs exponentially better with lower error rates.  

# Testing our Model
Let's upload some pictures to the notebook and test our model.

We load a test image using `PILImage.Create()` and use predict method on with it
```python
im = Path('drive/MyDrive/peaky_test/art1.jpg')
member,_,probs = learn.predict(PILImage.create(im))
print(f"This is {member}")
print(f"Probability it's {member}: {probs[0]:.4f}")
```

We get correct results for every test images with probability of 0.9 and above.

# Conclusion

It's incredible how easy it is to train an image classification model in under 30mins and with little data. It would seem that massive amounts of data are required to train something like this, but we've done it with 60 Images. 

# Resources and Links

- [fast.ai Course](https://course.fast.ai/)
- [fast.ai book lesson 1](https://github.com/fastai/fastbook/blob/master/01_intro.ipynb)
- [fast.ai forums for lesson 1](https://forums.fast.ai/t/lesson-1-official-topic/95287)
- [Colab Notebook for this lesson](https://colab.research.google.com/drive/1WPQS2DZIqTQPAN9x0Q3ocE8d8kk5MTYm?usp=sharing)

