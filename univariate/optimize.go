package univariate

import (
	"errors"
	"math"

	"github.com/btracey/opt/common"
)

// GradFree represents a gradient-free optimizer
type GradFreeOptimizer interface {
	Init(f Objective, initLoc, initObj float64) error
	Status() common.Status
	// The loc and obj are what are passed to convergence. Univariate provides
	// a check for where the best location is
	Iterate() (loc float64, obj float64, nFunEvals int, err error)
	// Result does any cleanup needed
	Result()
}

// Grad represents a gradient-based optimizer
type GradOptimizer interface {
	Init(f ObjGrader, initLoc, initObj, initGrad float64) error
	Status() common.Status
	// The loc and obj are what are passed to convergence. Univariate provides
	// a check for where the best location is
	Iterate() (loc float64, obj float64, grad float64, nFunEvals int, err error)
	// Result does any cleanup needed
	Result()
}

// TODO: Maybe this is superfluous with Helper. Maybe helper should contain the optimizer

// GradFreeWrapper is a convenience wrapper around a gradient-free algorithm that
// allows more fine-grained control over optimization progress. See OptimizeGradFree
// for example usage
type GradFreeWrapper struct {
	optimizer GradFreeOptimizer
	helper    *Helper
}

func NewGradFreeWrapper(optimizer GradFreeOptimizer) *GradFreeWrapper {
	return &GradFreeWrapper{
		optimizer: optimizer,
		helper:    NewHelper(),
	}
}

func (g *GradFreeWrapper) Init(settings *Settings, fun Objective, initLoc float64) error {

	initObj := settings.InitialObjective
	if math.IsNaN(initObj) {
		initObj = fun.Obj(initLoc)
	}

	g.helper.Init(settings, fun, initObj, math.Inf(1))
	return g.optimizer.Init(fun, initLoc, initObj)
}

func (g *GradFreeWrapper) Status() common.Status {
	return common.CheckStatus(g.helper, g.optimizer)
}

func (g *GradFreeWrapper) Iterate() (loc, obj float64, err error) {
	var nFunEvals int
	loc, obj, nFunEvals, err = g.optimizer.Iterate()
	if err != nil {
		return loc, obj, errors.New("error iterating optimizer: " + err.Error())
	}
	// Give a bogus value to gradient value
	g.helper.Iterate(loc, obj, math.Inf(1), nFunEvals)
	return loc, obj, nil
}

func (g *GradFreeWrapper) Result(status common.Status) *Result {
	g.optimizer.Result()
	return g.helper.Result(status)
}

// OptimizeGradFree optimizes a function that doesn't have (or use) the objective value
func OptimizeGradFree(f Objective, initLoc float64, settings *Settings, optimizer GradFreeOptimizer) (*Result, error) {
	if optimizer == nil {
		panic("no optimizer provided")
	}

	if settings == nil {
		settings = DefaultSettings()
	}

	wrapper := NewGradFreeWrapper(optimizer)

	err := wrapper.Init(settings, f, initLoc)
	if err != nil {
		return nil, errors.New("error initializing: " + err.Error())
	}

	var status common.Status
	for {
		// Check if it has converged
		status = wrapper.Status()
		if status != common.Continue {
			break
		}

		_, _, err := wrapper.Iterate()
		if err != nil {
			return nil, err
		}
	}
	return wrapper.Result(status), nil
}

// GradWrapper is a convenience wrapper around a gradient-based algorithm that
// allows more fine-grained control over optimization progress. See OptimizeGradFree
// for example usage
type GradWrapper struct {
	optimizer GradOptimizer
	helper    *Helper
}

func NewGradWrapper(optimizer GradOptimizer) *GradWrapper {
	return &GradWrapper{
		optimizer: optimizer,
		helper:    NewHelper(),
	}
}

func (g *GradWrapper) Init(settings *Settings, fun ObjGrader, initLoc float64) error {

	// TODO: Better error checking
	initObj := settings.InitialObjective
	initGrad := settings.InitialGradient
	if math.IsNaN(initObj) {
		initObj, initGrad = fun.ObjGrad(initLoc)
	}

	g.helper.Init(settings, fun, initObj, initGrad)
	return g.optimizer.Init(fun, initLoc, initObj, initGrad)
}

func (g *GradWrapper) Status() common.Status {
	return common.CheckStatus(g.helper, g.optimizer)
}

func (g *GradWrapper) Iterate() (loc, obj, grad float64, err error) {
	var nFunEvals int
	loc, obj, grad, nFunEvals, err = g.optimizer.Iterate()
	if err != nil {
		return loc, obj, grad, errors.New("error iterating optimizer: " + err.Error())
	}
	// Give a bogus value to gradient value
	g.helper.Iterate(loc, obj, grad, nFunEvals)
	return loc, obj, grad, nil
}

func (g *GradWrapper) Result(status common.Status) *Result {
	g.optimizer.Result()
	return g.helper.Result(status)
}

// OptimizeGradFree optimizes a function that doesn't have (or use) the objective value
func OptimizeGrad(f ObjGrader, initLoc float64, settings *Settings, optimizer GradOptimizer) (*Result, error) {

	if optimizer == nil {
		panic("no optimizer provided")
	}

	if settings == nil {
		settings = DefaultSettings()
	}

	wrapper := NewGradWrapper(optimizer)

	err := wrapper.Init(settings, f, initLoc)
	if err != nil {
		return nil, errors.New("error initializing: " + err.Error())
	}

	var status common.Status
	for {
		// Check if it has converged
		status = wrapper.Status()
		if status != common.Continue {
			break
		}

		_, _, _, err := wrapper.Iterate()
		if err != nil {
			return nil, err
		}
	}
	return wrapper.Result(status), nil
}
