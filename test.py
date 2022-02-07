success = True
try:
    import tensorflow
except ImportError:
    print("WARNING: TensorFlow not found")
    success = success and false
try:
    import tensorflow_datasets
except ImportError:
    print("WARNING: TensorFlow-datasets not found")
    success = success and false
try:
    import pandas
except ImportError:
    print("WARNING: Pandas not found")
    success = success and false
try:
    import numpy
except ImportError:
    print("WARNING: Numpy not found")
    success = success and false
try:
    import scipy
except ImportError:
    print("WARNING: Scipy not found")
    success = success and false

if success:
    print("All requirements satisfied")
