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

type graphvizGraph struct {
	g  *ag.Graph
	gg gographviz.Interface
}

// GraphvizGraph creates a gographviz graph representation of the Graph.
func GraphvizGraph(g *ag.Graph) (gographviz.Interface, error) {
	r := graphvizGraph{
		g:  g,
		gg: gographviz.NewEscape(),
	}

	if err := r.gg.SetDir(true); err != nil {
		return nil, err
	}
	if err := r.gg.AddAttr("", "rankdir", "LR"); err != nil {
		return nil, err
	}

	for _, node := range g.Nodes() {
		switch nt := node.(type) {
		case *ag.Variable:
			if err := r.addGVVariable(nt); err != nil {
				return nil, err
			}
		case *ag.Operator:
			if err := r.addGVOperator(nt); err != nil {
				return nil, err
			}
		case *ag.Wrapper:
			if param, ok := nt.GradValue.(nn.Param); ok {
				if err := r.addGVParam(nt, param.Name()); err != nil {
					return nil, err
				}
				continue
			}
			if err := r.addGVWrapper(nt); err != nil {
				return nil, err
			}
		default:
			return nil, fmt.Errorf("unexpected node type %T", node)
		}
	}
	return r.gg, nil
}

func (r *graphvizGraph) addGVVariable(v *ag.Variable) error {
	id := fmt.Sprintf("%d", v.ID())
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%d</FONT><BR />
			variable<BR />
			%s
		>`,
		v.ID(),
		gvMatrixShape(v.Value()),
	)
	attrs := map[string]string{
		"label": label,
		"shape": "box",
	}
	return r.gg.AddNode("", id, attrs)
}

func (r *graphvizGraph) addGVWrapper(v *ag.Wrapper) error {
	id := fmt.Sprintf("%d", v.ID())
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%d</FONT><BR />
			wrapper<BR />
			%s
		>`,
		v.ID(),
		gvMatrixShape(v.Value()),
	)
	attrs := map[string]string{
		"label": label,
		"shape": "box",
	}
	return r.gg.AddNode("", id, attrs)
}

func (r *graphvizGraph) addGVParam(v *ag.Wrapper, name string) error {
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
		gvMatrixShape(v.Value()),
	)
	attrs := map[string]string{
		"label": label,
		"shape": "box",
	}
	return r.gg.AddNode("", id, attrs)
}

func (r *graphvizGraph) addGVOperator(op *ag.Operator) error {
	operatorID := fmt.Sprintf("%d", op.ID())
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%d</FONT><BR />
			<B>%s</B><BR />
			%s
		>`,
		op.ID(),
		op.Name(),
		gvMatrixShape(op.Value()),
	)
	attrs := map[string]string{
		"label": label,
	}
	if err := r.gg.AddNode("", operatorID, attrs); err != nil {
		return err
	}

	for _, operand := range op.Operands() {
		operandID := fmt.Sprintf("%d", operand.ID())
		if err := r.gg.AddEdge(operandID, operatorID, true, nil); err != nil {
			return err
		}
	}
	return nil
}

func gvMatrixShape(m mat.Matrix) string {
	if m == nil {
		return ""
	}
	if m.IsScalar() {
		return "scalar"
	}
	return fmt.Sprintf("%d × %d", m.Rows(), m.Columns())
}
