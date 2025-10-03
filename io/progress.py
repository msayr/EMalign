import datetime
from pymongo import MongoClient
import json

def get_mongo_client(config_file=None):
    """
    Returns a MongoClient object.
    If config_file is provided, it will be used to configure the client.
    Otherwise, it will use default parameters.
    """
    if config_file:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return MongoClient(**config)
    else:
        return MongoClient()

def get_mongo_db(client, project_name):
    """
    Returns a database object.
    """
    db_name = f"alignment_{project_name}"
    return client[db_name]

def log_progress(db, stack_name, step_name, global_slice_index, local_slice_index, metadata):
    """
    Logs progress to the database.
    """
    collection = db[stack_name]
    doc = {
        'step_name': step_name,
        'global_slice': global_slice_index,
        'local_slice': local_slice_index,
        'timestamp': datetime.datetime.now(datetime.UTC),
        **metadata
    }
    collection.insert_one(doc)

def check_progress(db, stack_name, step_name, global_slice_index):
    """
    Checks if a slice has already been processed.
    """
    collection = db[stack_name]
    return collection.count_documents({'step_name': step_name, 'global_slice': global_slice_index}) > 0

def wipe_progress(db, stack_name):
    """
    Wipes the progress for a given stack.
    """
    db.drop_collection(stack_name)