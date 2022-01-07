#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdlib.h>


int houghaccum(PyObject *self, PyObject *args) {
    PyObject *arg1 = NULL;
    if (!PyArg_ParseTuple(args, "O", &arg1))
        return NULL;

    PyArrayObject *image = (PyArrayObject*) PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (image == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Could not parse as array");
        return NULL;
    }

    int dims = PyArray_NDIM(image);
    npy_intp *shape = PyArray_SHAPE(image);

    if (dims != 2) {
        PyErr_SetString(PyExc_ValueError, "Wrong array dimensions. Expected a 2-dimensinoal array.");
        Py_DECREF(image);
        return NULL;
    }

    int height = shape[0];
    int width = shape[1];

    int max_rho = height + width;
    int min_rho = -max_rho;
    int numrho = max_rho - min_rho + 1;
    int max_angle = 359, min_angle = 0;
    int numangle = max_angle - min_angle + 1;

    float tabSin[numangle];
    float tabCos[numangle];

    for (int i=0; i < numangle; i++) {
        tabCos[i] = cos(i * M_PI / 180);
        tabSin[i] = sin(i * M_PI / 180);
    }

    // stage 1. fill accumulator
    for( int i = 0; i < height; i++ ) {
        for( int j = 0; j < width; j++ ) {
            if( PyArray_GETPTR2(image, i, j) != 0 )
                for(int n = 0; n < numangle; n++ ) {
                    int r = cvRound( j * tabCos[n] + i * tabSin[n] );
                    r += (numrho - 1) / 2;
                    accum[(n+1) * (numrho+2) + r+1]++;
                }
        }
    }
}

static PyMethodDef methods[] = {
    {"houghaccum", houghaccum, METH_VARARGS, "Creates a Hough line accumulator based on an input image."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef hough_module = {
    PyModuleDef_HEAD_INIT,
    "hough",
    "Python interface for the OLED C library function",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_hough(void) {
    import_array();
    PyObject *module = PyModule_Create(&hough_module);
    return module;
}
