import  os

DATA_FOLDER = os.path.join('.', 'data')
CACHE_FOLDER = os.path.join('.', 'cache')

if __name__ == '__main__':
    import glob

    files = glob.glob(os.path.join(DATA_FOLDER, '**', '*.*'), recursive=True)
    print('Named explicitly:')
    for name in files:
        print(name)
    print(os.path.join(CACHE_FOLDER, '*.*'))
