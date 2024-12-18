import sys 

sys.path.append('../data_structure')

from tree import Tree 

class Commit:
    def __init__(self, branch, files, commit_message, timestamp, parent_commit_ids ):
        self.branch = branch 
        self.files = files 
        self.author = author
        self.commit_message = commit_message
        self.timestamp = timestamp
        self.parent_commit_ids = parent_commit_ids
        self.commit_id = Commit.generate_commit_id(author, commit_message, timestamp, parent_commit_ids)
    
    @staticmethod
    def generate_commit_id(*args):
        data = ''.join(map(str, args))
        sha1 = hashlib.sha1()
        sha1.update(data.encode('utf-8'))
        return sha1.hexdigest()

class DirectoryTree:
    def __init__(self, root_dir, timestamp):
        self.timestamp = timestamp

        self.tree = DirectoryTree.make_tree(root_dir) 

    @staticmethod
    def make_tree(root_dir):
        return Tree(root_dir)
    