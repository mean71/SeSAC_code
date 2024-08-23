import sys 
sys.path.append('../ADT')
sys.path.append('../data_structure')

class Branch:
    def __init__(self, branch_name, commits):
        self.branch_name = branch_name 
        self.commits = commits 