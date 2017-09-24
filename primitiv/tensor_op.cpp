#include "tensor_op.h"

#include <primitiv/tensor.h>

using namespace primitiv;

Tensor &tensor_inplace_multiply_const(Tensor &tensor, float k) {
    tensor *= k;
    return tensor;
}

Tensor &tensor_inplace_add(Tensor &tensor, const Tensor &x) {
    tensor += x;
    return tensor;
}

Tensor &tensor_inplace_subtract(Tensor &tensor, const Tensor &x) {
    tensor -= x;
    return tensor;
}

