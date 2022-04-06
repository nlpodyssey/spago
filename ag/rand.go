package ag

import (
	"fmt"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
	"time"
)

var globalGeneratorFloat32 = rand.NewLockedRand[float32](12345)
var globalGeneratorFloat64 = rand.NewLockedRand[float64](12345)

func globalGenerator[T mat.DType]() *rand.LockedRand[T] {
	switch any(T(0)).(type) {
	case float32:
		return (*rand.LockedRand[T])(globalGeneratorFloat32)
	case float64:
		return (*rand.LockedRand[T])(globalGeneratorFloat64)
	default:
		panic(fmt.Sprintf("ag: no random generator for type %T", T(0)))
	}
}

// Seed sets the seed for generating random numbers to the current time (converted to uint64).
func Seed[T mat.DType]() *rand.LockedRand[T] {
	r := globalGenerator[T]()
	r.Seed(uint64(time.Now().UnixNano()))
	return r
}

// ManualSeed sets the seed for generating random numbers.
func ManualSeed[T mat.DType](seed uint64) *rand.LockedRand[T] {
	r := globalGenerator[T]()
	r.Seed(seed)
	return r
}
