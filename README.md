# FakeAI

Welcome to FakeAI. We have created this as a capstone project for our
2023-24 CogWorks class. Please download the Chrome extension linked
below and give it a try. Right now, we are hitting ~82% accuracy on detecting
wheather articles are true or fake. We analyze based off of article length, words used, 
and word density (among other factors). It is important to note that this model
does not fact check information.

### Download and use the Chrome Extension
Please download FakeAI on the [Chrome Web Store](https://chromewebstore.google.com/detail/fakeai/ppgglflfncmmpmcoecnmfoecfdookflm)
<br>Find source code available on [GitHub](https://github.com/CoderElectronics/fakeai-chrome)
<br>Once the extension is installed in Chrome, highlight any text and right click. Then select "Sent to FakeAI". This will load a sidebar with statistics detailing the predicted validity of the text.

### Install dependencies and setup (not needed for extension)
This is the sklearn based classification server for FakeAI.
Here is the basic process to install all dependencies, load data, and train a model for the classification server.

    pip3 install -r ./requirements.txt

    # Download all dataset files
    python3 download_datasets.py

    # Pre-process and split data for training
    python3 process_split_data.py

    # Pre-process and split data for training
    python3 process_split_data.py

    # Train vectorizer and models, save results.
    python3 process_split_data.py

Run the classifier server and use the test scripts to send requests.

    python3 classifier_server.py

    # GUI that makes it easy to send requests from the dataset for testing
    python3 test_query_dataset_view.py
