# -----------------------------------------------------------
# This file contains the omni class, which is a lite version
# of the full aida.data.omniweb class of AIDA. This is used
# as an example and a guide for the developers of the AIDA
# project.
# -----------------------------------------------------------

import pandas as pd
import numpy as np

headers = ['year','day','hour','Bz','Np','V','Dst']
nulls = [None, None, None, 999.9, 999.9, 9999.,999]

class omnidata:
    """
    Class for the OMNIweb dataset.

    Class containing functions to read and process data from
    the OMNIweb database. In this example we limit the number of
    features to only four. We consider that the file has already
    been downloaded from the website.

    Notes
    -----
        This class is only a small example on which a more
        comprehensive class should be build, including data
        downloading

    Attributes
    ----------
    filename: str
        Is the name of the data file. This file is NOT an OMNIweb
        file. it does not contain the header rows and only contains
        four selected quantities in a specific column order.

    dt = 1:
        Time step, in hours between two history points used for
        the forecasting.
    nt = 5:
        Number of history points used for the forecasting.
    fcast = 1:
        Forecasting time in hours.
    """

    def __init__(self,
            filename,
            dt=1,
            nt=5,
            fcast=1):

        self.filename = filename
        self.dt = dt
        self.nt = nt
        self.fcast = fcast

    def len(self):
        """
        Count the number of lines in the file.
        """
        count = len(open(self.filename).readlines())
        return count

    def read(self, rowbeg, nrows, cols):
        """
        Read data from the file using Pandas.
        
        Read only nrows from the data file starting at rowbeg.
        Read only the columns cols.

        Parameters
        ----------
        rowbeg: int
            Starting row in the data file to read.
        nrows: int
            How many rows should be read in the data file.
        cols: :obj:`list` of :obj:`str`
            List of the columns to be read. This list should
            also be available from an independent function, so
            it is accessible before reading the full data file.
        """
        data = pd.read_table(self.filename,
                header=None,
                names=headers,
                delim_whitespace=True,
                skipinitialspace=True,
                skiprows=rowbeg,
                nrows=nrows)
        cnulls = [data.columns.get_loc(i) for i in cols]
        mask = data[cols].values!=[nulls[i] for i in cnulls]
        return data[cols], mask

    def load(self, xcols, ycols, rowbeg=0, nrows=None):
        """
        Function to read, load, and transform the data file into
        NumPy arrays that can be used in the machine learning model
        """
        T0  = self.dt*self.nt
        tau = T0+self.fcast
        
        xraw, xmask = self.read(rowbeg, nrows, xcols)
        yraw, ymask = self.read(rowbeg, nrows, ycols)
        m   = xraw.shape[0] - tau
        Y   = np.array(yraw.iloc[tau:])
        X   = np.array([xraw.iloc[i:i+T0:self.dt].values.flatten() for i in range(m)])
        
        Ymask = np.array(ymask[tau:])
        Xmask = np.array([xmask[i:i+T0:self.dt].flatten() for i in range(m)])
        mask  = np.array([Xmask.all(axis=1) & Ymask.all(axis=1)]).T
        
        m = Y.shape[0]
        Y = np.array([Y[i,:] for i in range(m) if mask[i]])
        X = np.array([X[i,:] for i in range(m) if mask[i]])
        
        return X,Y

    def normalize(self, rank, x, Max=None, Min=None, Mean=None):
        """
        Function to normalize any vector x
        """
        if Max is None: x_max = x.max(axis=0)
        else: x_max = Max
        if Min is None: x_min = x.min(axis=0)
        else: x_min = Min
        if Mean is None: x_mean = x.mean(axis=0)
        else: x_mean = Mean

        x_norm = (x - x_mean) / (x_max - x_min)
        return x_norm, x_max, x_min, x_mean

    def denorm(self, x_norm, xmax, xmin, xmean):
        """
        Function to de-normalize any vector x_norm
        """
        x = x_norm * (xmax - xmin) + xmean
        return x
