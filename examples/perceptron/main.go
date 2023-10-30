package main

import (
	"fmt"

	//lint:ignore ST1001 allow dot import just to make the example more readable
	. "github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

func main() {
	x := mat.Scalar(-0.8)
	w := mat.Scalar(0.4)
	b := mat.Scalar(-0.2)

	y := Sigmoid(Add(Mul(w, x), b))

	fmt.Printf("y = %0.3f\n", y.Value().Item())
}
