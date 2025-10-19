import click

def factorial(n):
    '''Calculate the factorial of n using recursion.
    '''
    if n < 1:
        return 1
    else:
        return n * factorial(n-1)


@click.command()
@click.argument('n', type=int)
def main(n):
    f = factorial(n)
    print("n! =", f)

if __name__ == '__main__':
    main()