import os
import pyperclip

def ospathup(folder,n=1):
    path=folder
    for i in range(n):
        path=os.path.split(path)[0]
    path=os.path.join(path,'')
    return path

def osfile(folder):
    file=os.path.split(folder)[1]
    return file

def ossplit(folder):
    dir,file=os.path.split(folder)
    return dir, file

def osfolderup(folder):
    return os.path.split(os.path.split(folder)[0])[1]
    
def copy_to_clipboard(string):
    pyperclip.copy(string)
    pyperclip.paste()
