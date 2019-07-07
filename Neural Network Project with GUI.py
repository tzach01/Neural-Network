# Dependencies include:
    # tkinter
    # numpy
    # pandas
    # scikit-learn
    # keras
    # TensorFlow v1 - currently only compatible with Python version 3.6 and lower

import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from os import path
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense

class Window:

    def __init__(self, master):

        # Configuring grid to expand at 1 to 1 pxl when enlarged
        master.rowconfigure(0, weight=1)
        master.rowconfigure(1, weight=1)
        master.rowconfigure(2, weight=1)
        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)

        # Creates header frame at top of window
        self.frame_header = ttk.Frame(master)
        self.frame_header.grid(
            row=0, column=0, columnspan=2, sticky='nsew')
        self.frame_header.config(height=50, width=400, relief=RIDGE)

        # Creates left middle frame
        self.frame_tleft = ttk.Frame(master)
        self.frame_tleft.grid(row=1, column=0, sticky='nsew')
        self.frame_tleft.config(height=200, width=200, relief=RIDGE)

        # Confirgures left middle frame to expand when enlarged
        self.frame_tleft.rowconfigure(0, weight=1)
        self.frame_tleft.rowconfigure(1, weight=1)
        self.frame_tleft.rowconfigure(2, weight=1)
        self.frame_tleft.rowconfigure(3, weight=1)
        self.frame_tleft.rowconfigure(4, weight=1)
        self.frame_tleft.columnconfigure(0, weight=1)
        self.frame_tleft.columnconfigure(1, weight=1)

        # Creates left lower frame
        self.frame_bleft = ttk.Frame(master)
        self.frame_bleft.grid(row=2, column=0, sticky='nsew')
        self.frame_bleft.config(height=200, width=200, relief=RIDGE)

        # Creates frame on right of window
        self.frame_right = ttk.Frame(master)
        self.frame_right.grid(row=1, column=1, rowspan=2, sticky='nsew')
        self.frame_right.config(height=400, width=200, relief=RIDGE)

        # Creates label, entry field, and browse button for training
        self._ftlabel = ttk.Label(self.frame_tleft, text='Training File')
        self._ftlabel.grid(row=1, column=0, padx=4, pady=4, sticky='w')
        self._ftentry = ttk.Entry(self.frame_tleft)
        self._ftentry.grid(row=1, column=1, padx=4, pady=4, sticky='ew')
        self._ftbrowse = ttk.Button(
            self.frame_tleft, text='Browse...')
        self._ftbrowse.grid(row=1, column=2, padx=4, pady=4)

        # Creates label, entry field, and browse button for analysis
        self._fplabel = ttk.Label(self.frame_tleft, text='Analysis File')
        self._fplabel.grid(row=2, column=0, padx=4, pady=4, sticky='w')
        self._fpentry = ttk.Entry(self.frame_tleft)
        self._fpentry.grid(row=2, column=1, padx=4, pady=4, sticky='ew')
        self._fpbrowse = ttk.Button(self.frame_tleft, text='Browse...')
        self._fpbrowse.grid(row=2, column=2, padx=4, pady=4)

        # Creates label, entry field, and browse button for choosing where to save resulting .csv file
        self._rslabel = ttk.Label(self.frame_tleft, text='Results Folder')
        self._rslabel.grid(row=3, column=0, padx=4, pady=4, sticky='w')
        self._rspentry = ttk.Entry(self.frame_tleft)
        self._rspentry.grid(row=3, column=1, padx=4, pady=4, sticky='ew')
        self._rsbrowse = ttk.Button(self.frame_tleft, text='Save As...')
        self._rsbrowse.grid(row=3, column=2, padx=4, pady=4)

        # Creates dropdown field for entering column name for target values
        self._clnlabel = ttk.Label(self.frame_tleft, text='Column Name')
        self._clnlabel.grid(row=4, column=0, padx=4, pady=4, sticky='w')
        self._clnentry = ttk.Combobox(self.frame_tleft)
        self._clnentry.grid(row=4, column=1, padx=4, pady=4, sticky='ew')

        # Creates button to initiate analysis of files
        self.analyze = ttk.Button(self.frame_bleft, text='Analyze')
        self.analyze.grid(row=7, column=0, padx=4, pady=10, sticky='e')
        
        # Creates button to reset and delete fields
        self.reset = ttk.Button(self.frame_bleft, text='Reset')
        self.reset.grid(row=7, column=1, padx=4, pady=10)
        self.close = ttk.Button(self.frame_bleft, text='Quit', command=master.quit)
        self.close.grid(row=7, column=2, padx=4, pady=10, sticky='w')
        
        # Creates label, entry field for choosing the number of hidden layers of the neural network
        self._Layerslabel = ttk.Label(self.frame_bleft, text='Hidden Layers')
        self._Layerslabel.grid(row=3, column=0, padx=4, pady=4, sticky='w')
        self._layentry = ttk.Entry(self.frame_bleft)
        self._layentry.grid(row=3, column=1, padx=4, pady=4, sticky='ew')
        self._layentry.insert(0,2)

        # Creates label, entry field for choosing the number of nodes for the first layer of the neural network
        self._inodelabel = ttk.Label(self.frame_bleft, text='Initial Layer Nodes')
        self._inodelabel.grid(row=4, column=0, padx=4, pady=4, sticky='w')
        self._inodeentry = ttk.Entry(self.frame_bleft)
        self._inodeentry.grid(row=4, column=1, padx=4, pady=4, sticky='ew')
        self._inodeentry.insert(0,50)
        
        # Creates label, entry field for choosing the number of nodes for all of the hidden layers
        self._nodelabel = ttk.Label(self.frame_bleft, text='Hidden Layers Nodes')
        self._nodelabel.grid(row=5, column=0, padx=4, pady=4, sticky='w')
        self._nodeentry = ttk.Entry(self.frame_bleft)
        self._nodeentry.grid(row=5, column=1, padx=4, pady=4, sticky='ew')
        self._nodeentry.insert(0,50)

        # Creates label, entry field for choosing the number of Epochs the analysis will go through
        self._epochslabel = ttk.Label(self.frame_bleft, text='Epochs')
        self._epochslabel.grid(row=6, column=0, padx=4, pady=4, sticky='w')
        self._epocentry = ttk.Entry(self.frame_bleft)
        self._epocentry.grid(row=6, column=1, padx=4, pady=4, sticky='ew')
        self._epocentry.insert(0,100)
        statusupdate = StringVar()
        statusupdate.set = 'Enter Inputs'
        
        # Creates text label that provides the status of the step the analysis is in
        self.status = ttk.Label(self.frame_bleft, textvariable = statusupdate)
        self.status.grid(row=7, column=0, columnspan=4,padx=4, pady=4, sticky='sw')
       
        # The function that will be called update the text label for the status bar
        def statusbar(self, StatusText):
            statusupdate.set = StatusText
            
        # Function that asks user for training file and reads the file. Initiated by clicking browse button.
        def browsetrainfile():
            fname = fd.askopenfilename(title='Browse for Training File', filetypes=[('Comma Delimitted', '*.csv')])
            if fname != None:
                statusbar(self, 'Reading training file')
                self._ftentry.delete(0, END)
                # Inserts the path of the file into the entry box
                self._ftentry.insert(0, fname)
                # Reads the file provided
                ftrain = path.expanduser(fname)
                ft = pd.read_csv(ftrain)
                # Performs one hot encoding and drops first column created
                self.X_train = pd.get_dummies(ft, drop_first=True)
                # Creates list of column names from training file and populates dropdown
                features = list(self.X_train.columns)
                self._clnentry.delete(0, END)
                self._clnentry['values'] = features
                statusbar(self, '')
                
                
            else:
                self._clnentry.delete(0, END)
                self._clnentry['values'] = ['']

        # Assigns command to browse button to call function that asks for file
        self._ftbrowse.config(command=browsetrainfile)

        # Function that asks user for analysis file. Initiated by clicking browse button.
        def browsetestfile():
            fname = fd.askopenfilename(title='Browse for Testing File', filetypes=[('Comma Delimitted', '*.csv')])
            if fname != None:
                self._fpentry.delete(0, END)
                self._fpentry.insert(0, fname)

        # Assigns command to browse button to call function that asks for file
        self._fpbrowse.config(command=browsetestfile)

        # Function that asks user for file name for final output file. Initiated by clicking Save As button.
        def saveresultsfile():
            fname = fd.asksaveasfilename(title='Save results...', defaultextension =('.csv'), filetypes=[('Comma Delimitted', '*.csv')])
            if fname != None:
                self._rspentry.delete(0, END)
                self._rspentry.insert(0, fname)

        # Assigns command to browse button to call function that asks for filename
        self._rsbrowse.config(command=saveresultsfile)

        # Stores values from entry boxes and passes them to analysis functions. Initiated by clicking analysis button.
        def startanalysis():
            statusbar(self, 'Starting Analysis')
            ftrain = self._ftentry.get()
            ftest = self._fpentry.get()
            savefile = self._rspentry.get()
            UserTarget = self._clnentry.get()
            filetrain(self, ftrain, ftest, savefile, UserTarget)
            
            
        # Assigns command to analysis button to begin analysis
        self.analyze.config(command=startanalysis)

        # Creates function that clears entry boxes
        def reset():
            self._ftentry.delete(0, END)
            self._fpentry.delete(0, END)
            self._rspentry.delete(0, END)
            self._clnentry.delete(0, END)
            self._clnentry['values'] = ['']
            statusbar(self, '')

        
        # Assigns command to analysis button to begin delete entry boxes
        self.reset.config(command=reset)


        def filetrain(self, ftrain, ftest, savefile, UserTarget):
            self.status.config(text=('Processing training data'))
            self.UserTarget = UserTarget
            self.ftrain = ftrain
            self.savefile = savefile
            # Reads input file for training and stores it in variable
            ft = pd.read_csv(ftrain)
            # Runs pandas script to change num numerical values to dummy values (e.g., One Hot Encoding) and create a column for each categorical value
            # Drops first column of dummy values to avoid multicollinearity
            final = pd.get_dummies(ft, drop_first=True)
            # Adds NaN to blank cells
            final = final.fillna(0)
            self.TargetCol = final.columns.get_loc(UserTarget)
            # Stores data features (e.g, height and weight) from dataframe in array
            X_train = final.drop(UserTarget, axis='columns')
            # Stores feature names
            self.features = X_train.columns
            self.inum = len(self.features)
            X_train = X_train.values[:, :]
            # Fill blanks spots with NaN
            X_train = np.nan_to_num(X_train)
            # Stores target values (e.g., gender) from dataframe in array
            y_train = final.values[:, self.TargetCol]
            # Reshapes array to be a column of target values
            y_train = y_train.reshape(-1, 1)
            # Normalizes values to be between 0 and 1
            self.scalerX = MinMaxScaler(feature_range=(0, 1))
            self.scalery = MinMaxScaler(feature_range=(0, 1))
            #Applies normalization scale to training data
            self.X_train = self.scalerX.fit_transform(X_train)
            self.y_train = self.scalery.fit_transform(y_train)
            # Stores normalization constants to be used to denormalize results
            self.con1 = self.scalery.scale_[0]
            self.con2 = self.scalery.min_[0]
            statusbar(self, 'Creating Neural Network')
            # Initiates training functions and passes needed parameters
            train(self, self.inum, self.X_train, self.y_train, nodeinit = int(self._inodeentry.get()), 
                        nodehidd = int(self._nodeentry.get()), layernum = int(self._layentry.get()), epoch = int(self._epocentry.get()))
            
        
        # Creates functions responsible for creating deep neural network model
        def train(self, inum, X_train, y_train, nodeinit =50, nodehidd =100, layernum = 2, epoch =100):
            
            self.nnet = Sequential()
            statusbar(self, 'Creating neural network: initial layer')
            # Creates initial layer for the number of features provided in file
            self.nnet.add(Dense(nodeinit, input_dim=self.inum, activation='relu', bias=False))
            
            # Adds a user designated number of hidden layers with the same of user designated nodes
            while layernum > 0:
                self.nnet.add(Dense(nodehidd, activation='relu', bias=False))
                statusbar(self, 'Creating neural network: hidden layer {}'.format(layernum))
                layernum = layernum - 1
            
            # Creates output layers with one output variable
            self.nnet.add(Dense(1, activation='linear', bias=False))
            statusbar(self, 'Creating neural network: compiling neural network')
            # Compiles model to determine error for back propagation
            self.nnet.compile(loss="mean_squared_error", optimizer='adam')
            statusbar(self, 'Process training data through neural network')
            # Runs model to training data for a user defined number of epochs
            self.nnet.fit(self.X_train, self.y_train, epochs=epoch, shuffle='true', verbose=1)
            filepredict(self, self.ftrain, self.TargetCol, self.scalerX, self.scalery, self.features)

        def filepredict(self, ftrain, TargetCol, scalerX, scalery, features):
            statusbar(self, 'Process analysis data')
            # Reads input file for prediction and stores it in variable
            fp = pd.read_csv(ftrain)
            # Applies one hot encoding to analysis file
            fp = pd.get_dummies(fp, drop_first=True)
            # Replaces blanks values with 0
            fp = fp.fillna(0)
            # Creates array that stores actual values for target variables. This is only for analysis files where we know the real answer so we can compare the model predictions. Can be deleted.
            y_test = fp.values[:, TargetCol]
            y_test = y_test.reshape(-1, 1)
            # Scales target values the same way as the training file
            self.y_test = scalery.transform(y_test)
            # Identifies missing columns in test data
            missing_cols = set(features) - set(fp.columns)
            # Identifies extra columns in test data
            extra_cols = set(fp.columns) - set(features)
            # Removes extra columns
            for d in extra_cols:
                fp = fp.drop(d, axis='columns')
            # Add a missing column in test set with default value equal to 0 in the same location as the training file
            for c in missing_cols:
                loc = features.get_loc(c)
                if loc <= len(fp.columns):
                    fp.insert(loc, c, 0, allow_duplicates=False)
                else:
                    fp[c] = 0
            # Initiates loop if column names in test file are not in the same order as training file
            while all(fp.columns != features):
                # creates array out of column names in order
                colsort = np.array(features)
                # reshapes array
                colsort = colsort.reshape(1, -1)
                # reindexes prediciton file data in order of training data columns
                fp = fp.reindex(colsort, axis='columns')
            # Stores values in an array
            X_test = fp.values[:, :]
            self.fp = fp
            self.X_test = scalerX.transform(X_test)
            modelpredict(self)
            
        def modelpredict(self):
            statusbar(self, 'Predicting target values...')
            # Creates a model predicition for each line in analysis file
            list = self.nnet.predict(self.X_test)
            # Denormalizing values in predictions
            list = (list - self.con2) / self.con1
            self.guess = list
            output(self)
            
        def output(self):
            statusbar(self, 'De-normalizing values')
            # Denormalizing values in analysis file
            y_test = (self.y_test - self.con2) / self.con1
            statusbar(self, 'Creating output .csv file')
            # Converts list array into a pandas Dataframe
            y_test = pd.DataFrame(y_test)
            # Converts list of predictions into a dataframe
            data = pd.DataFrame(self.guess)
            # Names the resulting column
            data.columns = ['{} Guess'.format(self.UserTarget)]
            y_test.columns = ['{} Actual'.format(self.UserTarget)]
            # Merges the resulting predictions with the original input dataframe
            mergedres = pd.concat([self.fp, data, y_test], 1)
            # Outputs dataframe into a CSV file
            mergedres.to_csv(self.savefile, index=False)
            statusbar(self, 'Output .csv file saved')
        
def main():
    root = tk.Tk()
    root.title('Neural Network Calculator')
    Window(root)
    root.mainloop()


if __name__ == "__main__":
    main()
