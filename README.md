# PersonalPortfolio
Included are files of my work on various projects with a short description of what they do. I've also linked a Google Drive folder of a dataset I created.

[dataset.py](https://github.com/allybush/PersonalPortfolio/blob/993745367c546006452ca21ec0972458ab7c17dd/dataset.py) used Spotify’s web API to create our own custom dataset—we found that datasets on Kaggle were too small or not quite what we were looking for. It iterated through a list of Spotify playlists for each music genre and turned the first thirty seconds of each song into a mel-spectrogram (visualization of a sound’s pitch, frequency, and strength) that we could use with our CNN. It sorted the resulting mel-spectrogram for each song into folders by genre.

[This](https://drive.google.com/drive/folders/1LB6511kMThrGdbB9CrNYJ34WQg2j1K-F?usp=share_link) Google Drive folder is the actual data I generated with this program, sorted into genre categories with approximately the same number of images per category. This prevented the model from training unevenly. We could’ve made the data set bigger—and consequently, the model more accurate—by scouring more Spotify playlists, but for the sake of the project and the power of my computer, we limited each category to 300-400 images. 

[functions.js](https://github.com/allybush/PersonalPortfolio/blob/993745367c546006452ca21ec0972458ab7c17dd/functions.js) is the AJAX/Javascript for our webpage. Not only does it do simple things, like button clicking and color changing, it also passes user information to the model/backend, then passes the result of the CNN back to the user.

[runmodel.py](https://github.com/allybush/PersonalPortfolio/blob/993745367c546006452ca21ec0972458ab7c17dd/runmodel.py) is the code that actually ran the model on the first 30 seconds of a user’s song taken from Spotify’s API. It ensures the song length is uniform, reshapes the data a bit, converts the song into a mel-spectrogram, then runs the stored model on the mel-spectrogram and passes the output to the webpage.

[trainmodel.py](https://github.com/allybush/PersonalPortfolio/blob/993745367c546006452ca21ec0972458ab7c17dd/trainmodel.py) is the segment of code that created a convolutional neural network using the Keras Python library  and trained it on the data above. We tinkered with the parameters of the neural network to increase accuracy and prevent overtraining.
