package multivariate

import (
	"errors"
	"math"

	"github.com/btracey/opt/common"
)

type GradOptimizer interface {
	Init(f ObjGrader, initLoc []float64, initObj float64, initGrad []float64) error
	Status() common.Status
	// loc and grad put in place
	Iterate(loc, grad []float64) (obj float64, nFunEvals int, err error)
	Result()
}

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

func (g *GradWrapper) Init(settings *Settings, fun ObjGrader, initLoc []float64) error {

	initObj := settings.InitialObjective
	initGrad := settings.InitialGradient
	if math.IsNaN(initObj) {
		initGrad = make([]float64, len(initLoc))
		initObj = fun.ObjGrad(initLoc, initGrad)

	}

	g.helper.Init(settings, fun, initObj, initGrad)
	return g.optimizer.Init(fun, initLoc, initObj, initGrad)
}

func (g *GradWrapper) Status() common.Status {
	return common.CheckStatus(g.helper, g.optimizer)
}

func (g *GradWrapper) Iterate(loc, grad []float64) (obj float64, err error) {
	var nFunEvals int
	obj, nFunEvals, err = g.optimizer.Iterate(loc, grad)
	if err != nil {
		return obj, errors.New("error iterating optimizer: " + err.Error())
	}
	// Give a bogus value to gradient value
	g.helper.Iterate(loc, obj, grad, nFunEvals)
	return obj, nil
}

func (g *GradWrapper) Result(status common.Status) *Result {
	g.optimizer.Result()
	return g.helper.Result(status)
}

// OptimizeGradFree optimizes a function that doesn't have (or use) the objective value
func OptimizeGrad(f ObjGrader, initLoc []float64, settings *Settings, optimizer GradOptimizer) (*Result, error) {

	//fmt.Println("In optimize grad")
	if optimizer == nil {
		//	optimizer = NewLbfgs()
		optimizer = NewBfgs()
	}

	if settings == nil {
		settings = DefaultSettings()
	}

	if initLoc == nil {
		return nil, errors.New("nil init loc")
	}
	if f == nil {
		return nil, errors.New("objective function is nil")
	}

	//	fmt.Println("f = ", f)

	wrapper := NewGradWrapper(optimizer)

	err := wrapper.Init(settings, f, initLoc)
	if err != nil {
		return nil, errors.New("error initializing: " + err.Error())
	}
	loc := make([]float64, len(initLoc))
	grad := make([]float64, len(initLoc))

	var status common.Status
	for {
		// Check if it has converged
		status = wrapper.Status()
		if status != common.Continue {
			break
		}

		_, err := wrapper.Iterate(loc, grad)
		if err != nil {
			return nil, err
		}
	}
	return wrapper.Result(status), nil
}
