## Automatic Differentiation

The [ag package](https://github.com/nlpodyssey/spago/tree/main/pkg/ag) (a.k.a. auto-grad) is the centerpiece of the
spaGO machine learning framework.

Neural models optimized by back-propagation require gradients to be available during training. The set of expressions
characterizing the forward-step of such models must be defined within
the [ag.Graph](https://github.com/nlpodyssey/spago/blob/main/pkg/ag/graph.go) to take advantage of automatic
differentiation.

Let's see if spaGO can tell us what two plus five is. Then, let's go one step further now and ask spaGO to give us the
gradients on `a` and `b`, starting with arbitrary output gradients.

Write some code:

```go
package main

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/ag"
)

func main() {
	// create a new node of type variable with a scalar
	a := ag.NewVariable[float32](mat.NewScalar[float32](2.0), true)
	// create another node of type variable with a scalar
	b := ag.NewVariable[float32](mat.NewScalar[float32](5.0), true)
	// create an addition operator (the calculation is actually performed here)
	c := ag.Add(a, b)
	// print the result
	fmt.Printf("c = %v\n", c.Value())

	ag.Backward(c, mat.NewScalar[float32](0.5))
	fmt.Printf("ga = %v\n", a.Grad())
	fmt.Printf("gb = %v\n", b.Grad())
}
```

It should print:

```console
c = [7]
ga = [0.5]
gb = [0.5]
```

