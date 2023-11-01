package main

import (
	"fmt"

	//lint:ignore ST1001 allow dot import just to make the example more readable
	. "github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

func main() {
	w := mat.Scalar(3.)
	b := mat.Scalar(1.)
	x := mat.Scalar(2.)

	y := Sigmoid(Add(Mul(w, x), b)) // y = sigmoid(w*x + b)

	fmt.Printf("y = %0.3f\n", y.Value().Item())
}
