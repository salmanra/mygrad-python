# mygrad-python
My copy of micrograd (ie a nn library from the basics).
# mygrad

**mygrad** is a neural net library from the pure basics of Python, and is functionally a copy of micrograd. 

# why mygrad
it's all for learning. 
the next steps are, in no particular order of dependency, but in absolute order of importance:

 1. Extend mygrad to make use of numpy's ndarrays to form a **Tensor capable of backprop**, a la tinygrad
 2. Write all the fancy new **optimizers**, SGD, SGD + momentum, Adam, AdamW, 2nd order optimization, and so on
 3. Make mygrad all again in C++
	 4. I lied about no dependencies. Here's the first: after remaking mygrad in C++, time profile the og Python against the newfangled C++. 
	 5. Here's the second: once you grasp the challenges and benefits of writing a neural net library in C++, learn enough of the Python/C API to first a) make a Python mygrad that is a wrapper around the C++ mygrad then second b) remake the basics of numpy. Even tinygrad seems to have remade array broadcasting for its Tensors, though that may just be calling numpy stuff.

What will you have gotten by the end of all this?

 1. Absolute, **presidential power** over the fundamental computational subroutines of deep learning architectures.
 2. Experience building a proper Python project, importable and usable in other projects.
 3. Experience building a proper C++ project, importable and usable in other projects.
 4. Sheer supreme jurisprudence over the Python/C API, allowing you to grasp how it is that C++ manages extreme speed gains over Python. 
 5. Some experience time profiling and optimizing various mathematical programs in C++ and Python. 
 6. yallah ya zolee henak nageddid hawa
