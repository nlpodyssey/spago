// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package graphviz

import (
	"fmt"
	"github.com/awalterschulze/gographviz"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

type builder struct {
	g  *ag.Graph
	gv gographviz.Interface
}

func newBuilder(g *ag.Graph) *builder {
	return &builder{
		g:  g,
		gv: gographviz.NewEscape(),
	}
}

func (b *builder) build() (gographviz.Interface, error) {
	if err := b.gv.SetDir(true); err != nil {
		return nil, err
	}
	if err := b.gv.AddAttr("", "rankdir", "LR"); err != nil {
		return nil, err
	}

	for _, node := range b.g.Nodes() {
		switch nt := node.(type) {
		case *ag.Variable:
			if err := b.addVariable(nt); err != nil {
				return nil, err
			}
		case *ag.Operator:
			if err := b.addOperator(nt); err != nil {
				return nil, err
			}
		case *ag.Wrapper:
			if param, ok := nt.GradValue.(nn.Param); ok {
				if err := b.addParam(nt, param.Name()); err != nil {
					return nil, err
				}
				continue
			}
			if err := b.addWrapper(nt); err != nil {
				return nil, err
			}
		default:
			return nil, fmt.Errorf("unexpected node type %T", node)
		}
	}
	return b.gv, nil
}

func (b *builder) addVariable(v *ag.Variable) error {
	id := fmt.Sprintf("%d", v.ID())
	name := v.Name()
	if name == "" {
		name = "-"
	}
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%d</FONT><BR />
			variable <B>%s</B><BR />
			%s
		>`,
		v.ID(),
		name,
		matrixShapeString(v.Value()),
	)
	attrs := map[string]string{
		"label": label,
		"shape": "box",
	}
	return b.gv.AddNode("", id, attrs)
}

func (b *builder) addWrapper(v *ag.Wrapper) error {
	id := fmt.Sprintf("%d", v.ID())
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%d</FONT><BR />
			wrapper<BR />
			%s
		>`,
		v.ID(),
		matrixShapeString(v.Value()),
	)
	attrs := map[string]string{
		"label": label,
		"shape": "box",
	}
	return b.gv.AddNode("", id, attrs)
}

func (b *builder) addParam(v *ag.Wrapper, name string) error {
	name1 := name
	if name1 == "" {
		name1 = "-"
	}
	id := fmt.Sprintf("%d", v.ID())
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%d</FONT><BR />
			param <B>%s</B><BR />
			%s
		>`,
		v.ID(),
		name1,
		matrixShapeString(v.Value()),
	)
	attrs := map[string]string{
		"label": label,
		"shape": "box",
	}
	return b.gv.AddNode("", id, attrs)
}

func (b *builder) addOperator(op *ag.Operator) error {
	operatorID := fmt.Sprintf("%d", op.ID())
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%d</FONT><BR />
			<B>%s</B><BR />
			%s
		>`,
		op.ID(),
		op.Name(),
		matrixShapeString(op.Value()),
	)
	attrs := map[string]string{
		"label": label,
	}
	if err := b.gv.AddNode("", operatorID, attrs); err != nil {
		return err
	}

	for _, operand := range op.Operands() {
		operandID := fmt.Sprintf("%d", operand.ID())
		if err := b.gv.AddEdge(operandID, operatorID, true, nil); err != nil {
			return err
		}
	}
	return nil
}

func matrixShapeString(m mat.Matrix) string {
	if m == nil {
		return ""
	}
	if m.IsScalar() {
		return "scalar"
	}
	return fmt.Sprintf("%d × %d", m.Rows(), m.Columns())
}
