import numpy as np
cimport numpy as np

cdef extern from 'wrapped_code_2.h':
    void autofunc(double *y_5454703, double *X7_5454707, double *X8_5454708, double *X6_5454709, double *X1_5454710, double *X2_5454711, double *X0_5454712, double *X4_5454713, double *X10_5454714, double *X3_5454715, double *X5_5454716, int m_5454704)

def autofunc_c(np.ndarray[np.double_t, ndim=1] X7_5454707, np.ndarray[np.double_t, ndim=1] X8_5454708, np.ndarray[np.double_t, ndim=1] X6_5454709, np.ndarray[np.double_t, ndim=1] X1_5454710, np.ndarray[np.double_t, ndim=1] X2_5454711, np.ndarray[np.double_t, ndim=1] X0_5454712, np.ndarray[np.double_t, ndim=1] X4_5454713, np.ndarray[np.double_t, ndim=1] X10_5454714, np.ndarray[np.double_t, ndim=1] X3_5454715, np.ndarray[np.double_t, ndim=1] X5_5454716):

    cdef int m_5454704 = X5_5454716.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] y_5454703 = np.empty((m_5454704))
    autofunc(<double*> y_5454703.data, <double*> X7_5454707.data, <double*> X8_5454708.data, <double*> X6_5454709.data, <double*> X1_5454710.data, <double*> X2_5454711.data, <double*> X0_5454712.data, <double*> X4_5454713.data, <double*> X10_5454714.data, <double*> X3_5454715.data, <double*> X5_5454716.data, m_5454704)
    return y_5454703