package graphviz

import (
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/stretchr/testify/assert"
)

func TestMarshal(t *testing.T) {
	buf, err := Marshal(ag.NewGraph[float32]())
	assert.Equal(t, buf, []byte{100, 105, 103, 114, 97, 112, 104, 32, 32, 123, 10, 9, 114, 97, 110, 107, 100, 105, 114, 61, 76, 82, 59, 10, 10, 125, 10})
	assert.Nil(t, err)
}
