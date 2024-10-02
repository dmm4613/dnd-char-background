# Dungeons and Dragons Character Backstory Machine Learning Model

<a id="notebook-one-header"> </a>

## Goals
My hopes for this exploration and technical modeling project is to create a product that takes in user output (Character Name, Character Species, and Character Class), and returns to the user a generated backstory. 

![character_pixelated_reroll_app](https://github.com/user-attachments/assets/aa6d8b87-4463-4c9e-a69d-b36c7e1339e9)


### How to accomplish this?
I am utilizing both the database created by [@janelleshane](https://github.com/janelleshane/DnD_bios) and ChatGPT4 to generate brief backstories. I also got a few backstories from some friends. I combined all of these into one dataset and worked on cleaning and preprocessing the data. My focus was to clean up the backstories and remove any stopwords and lemmentize words. This was to prepare the data for a seq2seq model I would train. 

### User interaction and the product
Users would provide the input. The product takes the input and predicts which type of backstory was relavant to the user. A random backstory would be pull from all related backstory clusters and then fed to the seq2seq model. The user receives back a generated paragraph of text. 

### First Steps
Read in the data and begin the cleaning process. I removed any rows that did not contain backstories, as they were useless to me. 

### Predictive Models
Next I used the character species and class to create clusters for the backstories. This would be used to help feed a backstory to the seq2seq model.

### Generate Text
The trickiest and hardest part of this project is returning to the user a comprehendable paragraph of text to be used for a backstory.

___

## Desired ouput vs. MVP
Due to the difficulty of this task and the time constaint imposed by the project, I did have to alter my expectations for the product released. It doesn't seem feasible, withouth using a very high powered, well trained NLP model, like GPT4, to produce the results I was looking for. Therefore, the new product will provide the user with some ideas or a word bank to use in crafting a backstory for their character. 

(update)
Created product that did return a random string of words. The words themselves don't seem to fit well enough together to work as a descriptive word bank. 

____

## Focal Shift
As the project continued it became clear there were a few roadblockers for this product to be successful. After working through an [Experimental Notebook](notebook-two-experimental.ipynb#notebook-two-header) to explore the original dataset noted above, and cleaning the data more, it is clear that lack of data is a huge roadblock. The product is completing the task at hand, but to continuously train the model would lead to considerable overfitting. Given this fact, this project has shifted from product oriented to research oriented. 

____

## Research Findings

### Data Blocker
The main finding of my work on this project was I was heavily blocked by the access to data. My initial dataset was small, so I tried to generate more backstories with GPT4, but that was not done perfectly and created a large amount of rehashed backstories. I overfit my own data. It felt very much like someone created the best madlibs prompt and a 1000 people filled it in. Or like Bart Simpson learning a lesson.
![bart_blackboard](https://github.com/user-attachments/assets/6adfc23e-efca-4756-a3b5-2623151c9799)

In my experimental notebook two, I worked on the original dataset, without GPT4 help, and really focused on the main D&D classes and species from the new 2024 Player's Handbook. This didn't make any improvements, which led me to understand this blocker more clearly. 

To work through this blocker, of course more time would be amazing, but also coming up with a better way of gathering this material. If I could find an active reddit group or other chat channel that would be willing to fill out surveys to get more backstories that would help tremedously, but again that takes time. The other idea is to come up with a better way of utilizing GPT4. I believe my issue was I asked for unique backstories in a batch of 1000. Maybe a smaller batch over more iterations would result in more variety. 

### seq2seq Text Generation
My Neural Network worked. It did generate text, but the results were always outstanding, or even good. 

https://github.com/user-attachments/assets/853bcb45-178f-4585-a2cc-ff40865f9975

I didn't train the network extensively, not due to time, but lack of a good connector, which is my second biggest blocker.

### What a Cluster ...
Due to my data size, my cluster creation and predictive model to determine which cluster to work with just never really made a big impact. At times I was predicting at 78%, but that was inflated. At my lows, I was predicting at 20%, which is worse than a guess I would say. This was definitely something I struggled with for a couple days. Working through different preprocessing steps, reworking those preporcessing steps, looking for better hyperparameters, anything to squeeze out a bit more accuracy, but in the end I do believe it is linked back to my dataset and its lack of substance. 

