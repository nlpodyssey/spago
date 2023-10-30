package main

import (
	"fmt"
	"log"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

func main() {
	// define the type of the elements in the tensors
	type T = float32

	// create a new node of type variable with a scalar
	a := mat.Scalar(T(2.0), mat.WithGrad(true)) // create another node of type variable with a scalar
	b := mat.Scalar(T(5.0), mat.WithGrad(true)) // create an addition operator (the calculation is actually performed here)
	c := ag.Add(a, b)

	// print the result
	fmt.Printf("c = %v (float%d)\n", c.Value(), c.Value().Item().BitSize())

	c.AccGrad(mat.Scalar(T(0.5)))

	if err := ag.Backward(c); err != nil {
		log.Fatalf("error during Backward(): %v", err)
	}

	fmt.Printf("ga = %v\n", a.Grad())
	fmt.Printf("gb = %v\n", b.Grad())
}
