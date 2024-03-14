import tkinter as tk
import numpy as np
from tkinter import filedialog
import sys
import os
import fnmatch


def askrange(label=["min", "max"], title="Select range"):
    class askrangec(tk.Frame):
        def __init__(self, master=None, title=None, label=["min", "max"]):
            tk.Frame.__init__(self, master)
            self.l0 = label[0]
            self.l1 = label[1]
            self.master = master
            self.title = title
            # self.master.geometry('200x200')
            # self.pack()
            self.make_widgets()
            self.mainloop()

        def make_widgets(self):
            winf = self.winfo_toplevel()
            self.top = winf
            self.top.title(self.title)
            tk.Grid.grid_columnconfigure(self.top, 0, minsize=100, weight=1)
            tk.Grid.grid_columnconfigure(self.top, 1, minsize=100, weight=2)
            tk.Grid.grid_rowconfigure(self.top, 0, minsize=2, weight=1)
            tk.Grid.grid_rowconfigure(self.top, 1, minsize=2, weight=1)

            self.framemin = tk.Frame(self.top, padx=10, pady=2)
            self.framemin.grid(columnspan=2, row=0, sticky=tk.W + tk.E + tk.S + tk.N)
            tk.Grid.grid_columnconfigure(self.framemin, 0, minsize=100, weight=1)
            tk.Grid.grid_columnconfigure(self.framemin, 1, minsize=100, weight=2)
            tk.Grid.grid_rowconfigure(self.framemin, 0, minsize=2, weight=1)
            tk.Grid.grid_rowconfigure(self.framemin, 1, minsize=2, weight=1)

            self.minlabel = tk.Label(
                self.framemin, text=self.l0, font=("Helvetica", 23)
            )
            self.minlabel.grid(column=0, row=0, sticky=tk.W + tk.E + tk.S + tk.N)

            self.selectionmin = tk.Entry(self.framemin, relief=tk.RIDGE)
            self.selectionmin.grid(column=1, row=0, sticky=tk.W + tk.E + tk.S + tk.N)

            self.framemax = tk.Frame(self.top, padx=10, pady=2)
            self.framemax.grid(columnspan=2, row=1, sticky=tk.W + tk.E + tk.S + tk.N)
            tk.Grid.grid_columnconfigure(self.framemax, 0, minsize=100, weight=1)
            tk.Grid.grid_columnconfigure(self.framemax, 1, minsize=100, weight=2)
            tk.Grid.grid_rowconfigure(self.framemax, 0, minsize=2, weight=1)
            tk.Grid.grid_rowconfigure(self.framemax, 1, minsize=2, weight=1)

            self.maxlabel = tk.Label(
                self.framemax, text=self.l1, font=("Helvetica", 23)
            )
            self.maxlabel.grid(column=0, row=1, sticky=tk.W + tk.E + tk.S + tk.N)

            self.selectionmax = tk.Entry(self.framemax, relief=tk.RIDGE)
            self.selectionmax.grid(column=1, row=1, sticky=tk.W + tk.E + tk.S + tk.N)

            self.framecom = tk.Frame(self.top, padx=10, pady=2)
            self.framecom.grid(columnspan=2, row=2, sticky=tk.W + tk.E + tk.S + tk.N)
            tk.Grid.grid_columnconfigure(self.framecom, 0, minsize=100, weight=1)
            tk.Grid.grid_columnconfigure(self.framecom, 1, minsize=100, weight=2)
            tk.Grid.grid_rowconfigure(self.framecom, 0, minsize=2, weight=1)
            tk.Grid.grid_rowconfigure(self.framecom, 1, minsize=2, weight=1)

            self.okbutton = tk.Button(
                self.framecom,
                text="Ok",
                font=("Helvetica", 23),
                command=self.master.quit,
                padx=15,
                pady=5,
            )
            self.okbutton.grid(column=0, row=2)  # ,sticky=tk.W+tk.E+tk.S+tk.N)

            self.cancelbutton = tk.Button(
                self.framecom,
                text="Cancel",
                font=("Helvetica", 23),
                command=sys.exit,
                padx=15,
                pady=5,
            )
            self.cancelbutton.grid(column=1, row=2)  # ,sticky=tk.W+tk.E+tk.S+tk.N)

        def range(self):
            try:
                xrange = [
                    np.float16(self.selectionmin.get()),
                    np.float16(self.selectionmax.get()),
                ]
            except ValueError:
                try:
                    xrange = [np.float16(self.selectionmin.get()), ""]
                except ValueError:
                    try:
                        xrange = ["", np.float16(self.selectionmax.get())]
                    except ValueError:
                        xrange = ["", ""]
            self.top.withdraw()
            self.top.update()
            self.top.destroy()
            return xrange

    root = tk.Tk()
    abc = askrangec(root, title=title, label=label)
    val = abc.range()
    return val


def askfolderexists_exit(folder):
    folder = os.path.expanduser(folder)
    if not os.path.exists(folder):
        print("Folder ", folder, " does not exist. Aborting...")
        sys.exit()


def askexists_skip(folder_or_file, verbose=False):
    folder_or_file = os.path.expanduser(folder_or_file)
    if os.path.exists(folder_or_file):
        if verbose == True:
            print("Folder or file", folder_or_file, "does exist. Continuing...")
        return True
    else:
        if verbose == True:
            print("Folder or file", folder_or_file, "does not exist. Continuing...")
        return False


def askfolderexists_create(folder):
    folder = os.path.expanduser(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)


def check_file_or_folder(file):
    file = os.path.expanduser(file)
    isfile = os.path.isfile(file)
    if isfile == True:
        return "file"
    isdir = os.path.isdir(file)
    if isdir == True:
        return "folder"
    else:
        return ""


def askfolder_path(initialdir="/Users/thales/", title="Choose directory"):
    root = tk.Tk()
    root.update()
    root.withdraw()
    root.update()
    folder = filedialog.askdirectory(initialdir=initialdir, title=title)
    folder = os.path.join(folder, "")
    root.withdraw()
    root.update()
    root.destroy()
    return folder


# return folder


def askfile_path(initialdir="/Users/thales/", title="Choose a file"):
    root = tk.Tk()
    root.update()
    root.withdraw()
    root.update()
    folder = filedialog.askopenfilename(initialdir=initialdir, title=title)
    # folder=os.path.join(folder,'')
    root.withdraw()
    root.update()
    root.destroy()
    return folder


def ask_path(initialdir="/Users/thales/", title="Choose a file or directory"):
    class Select_file_or_folder(tk.Frame):
        def __init__(self, master=None, initialdir=None, title=None):
            tk.Frame.__init__(self, master)
            self.master = master
            self.title = title
            self.initialdir = initialdir
            self.master.geometry("1000x1000")
            self.pack()
            self.make_widgets()

        def make_widgets(self):
            # don't assume that self.parent is a root window.
            # instead, call `winfo_toplevel to get the root window
            winf = self.winfo_toplevel()
            self.top = winf
            winf.title(self.title)

            self.botframe = tk.Frame(winf)
            self.botframe.pack(side=tk.BOTTOM, fill=tk.X)

            self.selection = tk.Entry(winf)
            self.selection.pack(side=tk.BOTTOM, fill=tk.X)
            self.selection.bind("<Return>", self.ok_event)

            self.filter = tk.Entry(winf)
            self.filter.pack(side=tk.TOP, fill=tk.X)
            self.filter.bind("<Return>", self.filter_command)

            self.midframe = tk.Frame(winf)
            self.midframe.pack(expand=tk.YES, fill=tk.BOTH)

            self.filesbar = tk.Scrollbar(self.midframe)
            self.filesbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.files = tk.Listbox(
                self.midframe, exportselection=0, yscrollcommand=(self.filesbar, "set")
            )
            self.files.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH)
            btags = self.files.bindtags()
            self.files.bindtags(btags[1:] + btags[:1])
            self.files.bind("<ButtonRelease-1>", self.files_select_event)
            self.files.bind("<Double-ButtonRelease-1>", self.files_double_event)
            self.filesbar.config(command=(self.files, "yview"))

            self.dirsbar = tk.Scrollbar(self.midframe)
            self.dirsbar.pack(side=tk.LEFT, fill=tk.Y)
            self.dirs = tk.Listbox(
                self.midframe, exportselection=0, yscrollcommand=(self.dirsbar, "set")
            )
            self.dirs.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)
            self.dirsbar.config(command=(self.dirs, "yview"))
            btags = self.dirs.bindtags()
            self.dirs.bindtags(btags[1:] + btags[:1])
            self.dirs.bind("<ButtonRelease-1>", self.dirs_select_event)
            self.dirs.bind("<Double-ButtonRelease-1>", self.dirs_double_event)

            self.ok_button = tk.Button(
                self.botframe, text="OK", command=self.ok_command
            )
            self.ok_button.pack(side=tk.LEFT)
            self.filter_button = tk.Button(
                self.botframe, text="Filter", command=self.filter_command
            )
            self.filter_button.pack(side=tk.LEFT, expand=tk.YES)
            self.cancel_button = tk.Button(
                self.botframe, text="Cancel", command=self.cancel_command
            )
            self.cancel_button.pack(side=tk.RIGHT)

            winf.protocol("WM_DELETE_WINDOW", self.cancel_command)

            winf.bind("<Alt-w>", self.cancel_command)
            winf.bind("<Alt-W>", self.cancel_command)

        def select(self, pattern="*", default="", key=None):
            dir_or_file = self.initialdir
            if key and key in dialogstates:
                self.directory, pattern = dialogstates[key]
            else:
                dir_or_file = os.path.expanduser(dir_or_file)
                if os.path.isdir(dir_or_file):
                    self.directory = dir_or_file
                else:
                    self.directory, default = os.path.split(dir_or_file)
            self.set_filter(self.directory, pattern)
            self.set_selection(default)
            self.filter_command()
            self.selection.focus_set()
            self.top.wait_visibility()  # window needs to be visible for the grab
            self.top.grab_set()
            self.how = None
            self.master.mainloop()  # Exited by self.quit(how)
            if key:
                directory, pattern = self.get_filter()
                if self.how:
                    directory = os.path.dirname(self.how)
                dialogstates[key] = directory, pattern
            self.top.withdraw()
            self.top.update()
            self.top.destroy()
            return self.how

        def quit(self, how=None):
            self.how = how
            self.master.quit()  # Exit mainloop()

        def dirs_double_event(self, event):
            self.filter_command()

        def dirs_select_event(self, event):
            dir, pat = self.get_filter()
            subdir = self.dirs.get("active")
            dir = os.path.normpath(os.path.join(self.directory, subdir))
            self.set_filter(dir, pat)
            self.set_selection(os.path.join(dir, ""))
            self.filter_command()

        def files_double_event(self, event):
            self.ok_command()

        def files_select_event(self, event):
            file = self.files.get("active")
            self.set_selection(file)

        def ok_event(self, event):
            self.ok_command()

        def ok_command(self):
            self.quit(self.get_selection())

        def filter_command(self, event=None):
            dir, pat = self.get_filter()
            try:
                names = os.listdir(dir)
            except OSError:
                self.master.bell()
                return
            self.directory = dir
            self.set_filter(dir, pat)
            names.sort()
            subdirs = [os.pardir]
            matchingfiles = []
            for name in names:
                fullname = os.path.join(dir, name)
                if os.path.isdir(fullname):
                    subdirs.append(name)
                elif fnmatch.fnmatch(name, pat):
                    matchingfiles.append(name)
            self.dirs.delete(0, tk.END)
            for name in subdirs:
                self.dirs.insert(tk.END, name)
            self.files.delete(0, tk.END)
            for name in matchingfiles:
                self.files.insert(tk.END, name)
            head, tail = os.path.split(self.get_selection())
            if tail == os.curdir:
                tail = ""
            # self.set_selection(tail)

        def get_filter(self):
            filter = self.filter.get()
            filter = os.path.expanduser(filter)
            if filter[-1:] == os.sep or os.path.isdir(filter):
                filter = os.path.join(filter, "*")
            return os.path.split(filter)

        def get_selection(self):
            file = self.selection.get()
            file = os.path.expanduser(file)
            return file

        def cancel_command(self, event=None):
            self.quit()

        def set_filter(self, dir, pat):
            if not os.path.isabs(dir):
                try:
                    pwd = os.getcwd()
                except OSError:
                    pwd = None
                if pwd:
                    dir = os.path.join(pwd, dir)
                    dir = os.path.normpath(dir)
            self.filter.delete(0, tk.END)
            self.filter.insert(tk.END, os.path.join(dir or os.curdir, pat or "*"))

        def set_selection(self, file):
            self.selection.delete(0, tk.END)
            self.selection.insert(tk.END, os.path.join(self.directory, file))

    root = tk.Tk()
    abc = Select_file_or_folder(root, initialdir=initialdir, title=title)
    file = os.path.expanduser(abc.select())
    return file


def flush_input():
    try:
        import msvcrt

        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError or ModuleNotFoundError:
        import sys, termios

        termios.tcflush(sys.stdin, termios.TCIOFLUSH)


def ask_user_input(Message=""):
    # flush_input()
    ans = input(Message)
    #    print('')
    return ans


def number_to_str_or(k, frmt="{:0.3f}", string=""):
    try:
        string = frmt.format(k)
    except ValueError:
        string = str(string)
    return string
