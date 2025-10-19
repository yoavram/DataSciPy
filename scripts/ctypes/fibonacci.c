/*
    Filename: fibonacci.c
    To be used with fibonacci.py, as an imported library. Use Scons to compile,
    simply type 'scons' in the same directory as this file (see www.scons.org).
*/
#if defined(_MSC_VER)
    #define EXPORT __declspec(dllexport)
#else
    //  do nothing and hope for the best?
    #define EXPORT
#endif

EXPORT int fib(int);
EXPORT void fibseries(int *, int, int *);
EXPORT void fibmatrix(int *, int, int, int *);

/* Function prototypes */
int fib(int a);
void fibseries(int *a, int elements, int *series);
void fibmatrix(int *a, int rows, int columns, int *matrix);


int fib(int a)
{
    if (a <= 0) /*  Error -- wrong input will return -1. */
        return -1;
    else if (a==1)
        return 0;
    else if ((a==2)||(a==3))
        return 1;
    else
        return fib(a - 2) + fib(a - 1);
}

void fibseries(int *a, int elements, int *series)
{
    int i;
    for (i=0; i < elements; i++)
    {
    series[i] = fib(a[i]);
    }
}

void fibmatrix(int *a, int rows, int columns, int *matrix)
{
    int i, j;
    for (i=0; i<rows; i++)
        for (j=0; j<columns; j++)
        {
            matrix[i * columns + j] = fib(a[i * columns + j]);
        }
}
