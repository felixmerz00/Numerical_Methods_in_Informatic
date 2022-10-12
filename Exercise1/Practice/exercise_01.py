import frontend
import backend
from ctscanner import CTScanner

# Set up the scanner.
# The initial resolution is 20, you can either change the value here, or call the setResolution method later.
scanner = CTScanner(20)

# These are the methods you are supposed to implement in backend.py
A, b = backend.setUpLinearSystem(scanner)
x = backend.solveLinearSystem(A, b)

# Show the resulting image, as well as a reference solution
# No need to edit this
frontend.displayResults(x, scanner)

