GoMNIST
=======

GoMNIST is a Go driver for reading Yann LeCun's MNIST dataset of handwritten digits

The MNIST dataset is included. It is a copy from [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/).

How to use
==========

Install the packages:

	% go get /petar/GoMNIST

Import them in your code as

	% import "/petar/GoMNIST"

Load the MNIST training and testing data sets into memory using

	train, test, err := Load("./data")
	if err != nil {
		…
	}

Sweep through the samples

	sweeper := train.Sweep()
	for {
		image, label, present := sweeper.Next()
		if !present {
			break
		}
		…
	}

You can also have random access to the sets, using the methods of Set.

Author
======

My homepage is [Petar Maymounkov](http://pdos.csail.mit.edu/~petar/).
Follow me on Twitter at [@maymounkov](http://twitter/maymounkov).
