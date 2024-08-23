import os 
import sys 
sys.path.append('../ADT')
sys.path.append('../data_structure')

import util 
import difflib 

class File:
    UNTRACKED = 'Untracked' 
    UNMODIFIED = 'Unmodified'
    MODIFIED = 'Modified'
    STAGED = 'Staged'
    NEWLINE = '\n'

    def __init__(self, path, git_root):
        self.path = path 
        self.status = File.UNTRACKED
        self.verison = {}
        
    def add(self):
        self.status = File.STAGED

    def stage(self, current_version):
        self.status = File.STAGED

    def commit(self):
        self.status = File.UNMODIFIED

    def remove(self):
        self.status = File.UNTRACKED

    # DO NOT CHANGE THIS FUNCTION 
    def is_binary(self):
        try:
            with open(file_path, 'rb') as file:
                chunk = file.read(num_bytes)
                
                # if none in chunk, usually binary 
                if b'\x00' in chunk:  
                    return True
                
                text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(32, 127)))
                non_text_ratio = sum(byte not in text_chars for byte in chunk) / len(chunk)
                return non_text_ratio > 0.3  
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return False

    # --------------------------------------------
    # Work on current change 
    # --------------------------------------------
    
    def __sub__(self, other):
        if not isinstance(other, File):
            raise TypeError(f'{other} should be of type File, not {type(other)}')
        
        diff_generator = difflib.ndiff(self.content, other.content)
        diffs = []
        for i, line in enumerate(diff_generator):
            if line.startswith('+ '):
                diffs.append(('+', i, line[2:]))
            elif line.startswith('- '):
                diffs.append(('-', i, line[2:]))
        return FileDelta(diffs)

    def __add__(self, other):
        if not isinstance(other, FileDelta):
            raise TypeError(f'{other} should have type FileData, not {type(other)}')
        new_content = self.content[:]
        for diff in other.diffs:
            if diff[0] == '+':
                new_content.insert(diff[1], diff[2])
            elif diff[0] == '-':
                new_content.pop(diff[1])
        return new_content    

class FileDelta:
    def __init__(self, diffs):
        self.diffs = diffs 

    def __str__(self):
        result = []
        for diff in self.diffs:
            action = '+' if diff[0] == '+' else '-'
            result.append(f"{action} line {diff[1]}: {diff[2]}")
        return '\n'.join(result)

    def __add__(self, file):
        """Apply this FileDelta to a File object."""
        if not isinstance(file, File):
            raise TypeError("Can only add FileDelta to File.")
        return file + self

    def __sub__(self, other):
        """Invert this FileDelta to return to the original File."""
        if not isinstance(other, FileDelta):
            raise TypeError("Can only subtract FileDelta from FileDelta.")
        inverted_diffs = []
        for diff in other.diffs:
            if diff[0] == '+':
                inverted_diffs.append(('-', diff[1], diff[2]))
            elif diff[0] == '-':
                inverted_diffs.append(('+', diff[1], diff[2]))
        return FileDelta(inverted_diffs)

    
