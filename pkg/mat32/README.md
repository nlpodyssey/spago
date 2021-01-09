# Matrix

In spaGo, all mathematical operations are performed on 2D matrices (vectors and scalars as a subset of).
The [Matrix](https://github.com/nlpodyssey/spago/blob/master/pkg/mat32/matrix.go) interface defines _setter_ and _getter_ methods to access its elements plus a few methods to perform linear algebra operations with other matrices, such
as element-wise addition, subtraction, product, and matrix-matrix multiplication. Other convenient methods are the
usual _max()_, _min()_, _abs()_, and not much else.
[Dense](https://github.com/nlpodyssey/spago/blob/master/pkg/mat32/dense.go)
and [Sparse](https://github.com/nlpodyssey/spago/blob/master/pkg/mat32/sparse.go) are the two Matrix implementations
values that work with dense and sparse values, respectively.

### Note

The performance of linear algebra operations is a major bottleneck in a machine learning library. The higher the speed
on this basic module, the higher the throughput of high-level tasks. This is why after several attempts to write
efficient code in Go, I decided to rely on
Gonum [assembly code](https://github.com/nlpodyssey/spago/tree/master/pkg/mat32/internal/asm/f32), importing their _
internal_ package directly into spaGO.

By the way, some colleagues asked me why I didn't use [Gonum](https://github.com/gonum/gonum) library as a whole instead
of reimplementing it all from scratch. First, I wanted to get my hands dirty and understand what structures and math
operations are needed to build deep learning models from the ground up: that's what spaGO is all about! Second, Gonum
uses distinct types for matrices, vectors, symmetric and triangular matrices, transposed, and so on, often forcing you
to manage numerous "switch" in your code. It is just too sophisticated for what I think I need at the moment. Giving
power to the Matrix by adding many functions is a common approach. Nevertheless, in spaGO, I wanted to reduce the
responsibility of this part, keeping the implementation really to a minimum.
