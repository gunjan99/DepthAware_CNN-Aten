#include <TH/TH.h>

int depthconv_forward(at::Tensor *input, at::Tensor *offset,
                        at::Tensor *output);
int depthconv_backward(at::Tensor *grad_output, at::Tensor *grad_input,
                         at::Tensor *grad_offset);

int depthconv_forward(at::Tensor *input, at::Tensor *offset,
                        at::Tensor *output)
{
  //
  return 1;
}

int depthconv_backward(at::Tensor *grad_output, at::Tensor *grad_input,
                         at::Tensor *grad_offset)
{
  //
  return 1;
}