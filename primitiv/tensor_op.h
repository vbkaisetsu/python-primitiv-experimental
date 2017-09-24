#ifndef PYTHON_PRIMITIV_TENSOR_OP_H_
#define PYTHON_PRIMITIV_TENSOR_OP_H_

#include <primitiv/tensor.h>

primitiv::Tensor &tensor_inplace_multiply_const(primitiv::Tensor &tensor, float k);
primitiv::Tensor &tensor_inplace_add(primitiv::Tensor &tensor, const primitiv::Tensor &x);
primitiv::Tensor &tensor_inplace_subtract(primitiv::Tensor &tensor, const primitiv::Tensor &x);

#endif
