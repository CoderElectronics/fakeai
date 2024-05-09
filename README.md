# FakeAI

This is the sklearn based classification server for FakeAI.

### Install dependencies and setup
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