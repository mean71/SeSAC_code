import os 
import sys 
sys.path.append('../ADT')
sys.path.append('../data_structure')

import hashlib 
import util 
import difflib 

class GitRepository:
    """Class for git repository management. 

    In this project, you should implement various git repository managing functions. For managing the git repository status, you should know how the git works, and how to implement them. 

    The git management system is basically a Graph data structure management, where its nodes are snapshots of the files of interest. Its status and each nodes are stored in the .mygit directory. Branches are linked-list data structure composed of nodes. 

    Each nodes - here we call commits in the context of git - are composed of Tree data structure, which its nodes are files or directories. 

    Therefore, knowing how git works means 

    - To know how the nodes are composed and manipulated (will be implemented in the commit.py/Commit class) 
    - To know how the files are managed (will be implementedin the file.py/File class)
    - How to manage Branch (essentially linked-list of Commit nodes, will be implemented in the branch.py/Branch class)
    - How to manage whole git structure by managing commits, branches, etc,.. (will be implemented in this GitPreository class)

    1. Git Repository Lifecycle

    1.1 Initialization 

    When the git repo is initialized in the directory root, everything related to git will be saved under root/.mygit folder. The structure of .mygit folder is illustrated in 2.1 .mygit Directory section. 

    1.2 File Modification 

    You will than create/modify files in the directory. Those files will be marked 'Untracked' if it the file is never 'added' to the git repository. Otherwise, it will be marked 'Modified'.

    Remember that having a file in the working directory(physically) does not mean that git is tracking it - the file shall be first added to the git repository in order for git to track its changes. About the lifecycle of File objects, see file.py/File class documentation.

    Usually, this part of git lifecycle is called 'programming'. :)  

    1.3 Add the File (Staging Phase) 

    Once you have modified the files, you will be than asked to stage the modifications you have made. GitRepository.add method is responsible for handling this operation. 

    When GitRepository.add method is called, it will call File.add method or File.stage method accordingly, based on the


    2. Important Directories and Files 
    
    2.1. .mygit Directory

    .mygit directory is the core of git repository management. 

    .mygit/ 
    ├── status
    ├── branch 
    ├── stage/
    └── commits/ 
        ├── 3bcac7/ (commit id for commit 1)
        ├── 5f91b1/ (commit id for commit 2)
        ├── ...
        ├── 1d29a7/ (commit id for commit n)
        |    ├── commit_data
        |    └──blobs/
        ├── ...
        └── 4f4f1c/ (commit id for commit N)
    
    Each file/directory contains the following information. 
    
    - /status: Contains current status of the repository. It contains 
        * current commit version, a string object that contains hash value of the commit version. 
        * current branch, a Branch instance. 
        * whole commit graph, a Graph instance 

        Three elements will be pickled in status, as below. 
    - /branch 

    - /stage/
    - /commits/: Contains commit directories, which contains snapshots of the whole directory. 

    2.2 {commit id} directory (Commit Directory)

    Each commit directory contains the file 'commit_data' and directory blobs/. 

    - /commit_data
    - /blobs/: This directory stores the File instance, or pickled FileDiff object. 

    3. Addtional Notes 

    - On pickling/unpickling object, using util.save and util.load is highly recommended. 
    - 
    """

    GIT_DIR = '.mygit'
    STAGE_DIR = os.path.join(GIT_DIR, 'stage')
    BRANCH_FILES = os.path.join(GIT_DIR, 'branch')

    def __init__(self, root):
        self.root = root 
        git_dir = os.path.join(root, GitRepository.GIT_DIR)
        stage_dir = os.path.join(root, GitRepository.STAGE_DIR)
        util.create_dir(git_dir)
        util.create_dir(stage_dir)
        
        self.git_dir = git_dir 
        self.stage_dir = stage_dir 

        self.commit_history = Graph()
        self.head = None
        self.branch = 'origin'
    
    # --------------------------------------------
    # Work on current change 
    # --------------------------------------------
    
    def add(self, args):
        """Adds files to the current version. 

        Args:
        """
        pass

    def rm(self, args):
        pass 

    # --------------------------------------------
    # Status Management  
    # --------------------------------------------

    def status(self):
        pass 
    
    def log(self, args):
        pass 
    
    
    # --------------------------------------------
    # Git Branch Management  
    # --------------------------------------------

    def branch(self, args):
        pass 

    def checkout(self, args):
        pass 

    def merge(self, args):
        pass 

    def commit(self, args):
        pass 

    def reset(self, args):
        pass 
