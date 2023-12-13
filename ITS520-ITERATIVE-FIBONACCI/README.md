#####################################################
## ARGPARSER: https://docs.python.org/3/howto/argparse.html

## python fibonacci-project.py --help
## py fibonacci-project.py --help

## python fibonacci-project.py --filename fibonacci_sequence/h5fib --iters 200 --epochs 400000 --learn_rate (1/10000) --thresh True
## py fibonacci-project.py --filename fibonacci_sequence/h5fib --iters 200 --epochs 400000 --learn_rate (1/10000) --thresh True
#####################################################
https://github.com/jzip219/ml2023/tree/main/ITS520-FIBONACCI-LINEAR-REGRESSION
#####################################################

we would need to create an Entry Widget for each argument expected by the parser,
then build our command line string from these entries
then launch the script with the command line arguments included
then, integrate ONNX Runtime into the existing code