#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <stdio.h>


PyObject *houghaccum(PyObject *self, PyObject *args) {
    PyObject *arg1 = NULL;
    if (!PyArg_ParseTuple(args, "O", &arg1))
        return NULL;

    PyArrayObject *image = (PyArrayObject*) PyArray_FROM_OTF(arg1, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
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
    int max_angle = 179, min_angle = 0;
    int numangle = max_angle - min_angle + 1;

    double tabSin[numangle];
    double tabCos[numangle];

    double angle_step = M_PI / numangle;
    for (int i=0; i < numangle; i++) {
        tabCos[i] = cos((double) i * angle_step);
        tabSin[i] = sin((double) i * angle_step);
    }

    npy_intp accum_dims[] = {numangle, numrho};
    PyArrayObject *accum = (PyArrayObject*) PyArray_Zeros(2, accum_dims, PyArray_DescrFromType(NPY_INT64), 0);

    for( int i = 0; i < height; i++ ) {
        for( int j = 0; j < width; j++ ) {
            if( *(npy_uint8*) PyArray_GETPTR2(image, i, j) != 0 )
                for(int theta = 0; theta < numangle; theta++ ) {
                    int rho = round( j * tabCos[theta] - i * tabSin[theta] );
                    rho -= min_rho;
                    npy_int64 *bin = (npy_int64*)PyArray_GETPTR2(accum, theta, rho);
                    (*bin)++;
                }
        }
    }

	Py_DECREF(image);
	return (PyObject*) accum;
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
