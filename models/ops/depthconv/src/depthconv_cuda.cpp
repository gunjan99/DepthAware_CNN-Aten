#include <THC/THC.h>
#include <pybind11/pybind11.h>
#include<torch/torch.h>


template <typename DType>
void depthconv_im2col(cudaStream_t stream, const DType *data_im,
                       const DType *data_depth, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w, DType *data_col);

template <typename DType>
void depthconv_col2im(cudaStream_t stream, const DType *data_col,
                       const DType *data_depth, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w, DType *grad_im);



int depthconv_forward_cuda(at::Tensor *input,
                             at::Tensor *input_depth,
                             at::Tensor *weight, at::Tensor * bias, at::Tensor *output,
                             at::Tensor *columns, at::Tensor *ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationH, int dilationW);

int depthconv_backward_input_cuda(
    at::Tensor *input, at::Tensor *input_depth, at::Tensor *gradOutput,
    at::Tensor *gradInput, at::Tensor *weight,
    at::Tensor *columns, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationH, int dilationW);

int depthconv_backward_parameters_cuda(
    at::Tensor *input, at::Tensor *input_depth, at::Tensor *gradOutput,
    at::Tensor *gradWeight, at::Tensor *gradBias,
    at::Tensor *columns, at::Tensor *ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationH, int dilationW,
    float scale);


extern THCState *state;

void shape_check(THCState *state, at::Tensor *input, at::Tensor *input_depth,
                 at::Tensor *gradOutput, at::Tensor *weight, at::Tensor *bias, int kH, int kW,
                 int dH, int dW, int padH, int padW, int dilationH,
                 int dilationW) {

  THArgCheck(weight->dim() == 4, 5,                                                                 // weight->nDimension => weight->dim()
             "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
             "but got: %s",
             weight->dim());                                                                        // weight->nDimension => weight->dim()

  THArgCheck((*weight).is_contiguous(), 5,                                   // THArgCheck(THCudaTensor_isContiguous(state, weight), 5,
            "weight tensor has to be contiguous");                        //           "weight tensor has to be contiguous");
                                                                    
                                                                      

  THArgCheck(kW > 0 && kH > 0, 9,
             "kernel size should be greater than zero, but got kH: %d kW: %d",
             kH, kW);

  THArgCheck((weight->size(2) == kH && weight->size(3) == kW), 9,                                   // weight->size[2] => weight->size(2) (x2 times)
             "kernel size should be consistent with weight, but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d", kH,
             kW, weight->size(2), weight->size(3));                                                 // weight->size[2/3] => weight->size(2/3)                

  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH,
             dW);

  THArgCheck(
      dilationW > 0 && dilationH > 0, 14,
      "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
      dilationH, dilationW);

  //////////// check bias //////////////////

  THArgCheck(!bias || (*bias).is_contiguous(), 5,                                         
             "bias tensor has to be contiguous");

  if (bias != NULL) {
//    THCUNN_check_dim_size(state, bias, 1, 0, weight->size[0]);
    THArgCheck(bias->dim()==1, 6,                                                                   // bias->nDimension => bias->dim()
             "Need bias of dimension %d but got %d", 1, bias->dim());                               // bias->nDimension => bias->dim()                    
    THArgCheck(bias->size(0)==weight->size(0), 6,                                                   // (bias/weight)->size[0] => (bias/weight)->size(0)    
             "Need bias of size %d but got %d", weight->size(0), bias->size(0));                    // (bias/weight)->size[0] => (bias/weight)->size(0)
  }
//////////////////////////////////////////

  int ndim = input->dim();                                                                          // input->nDimension => input->dim()
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THArgCheck(ndim == 3 || ndim == 4, 2,
             "3D or 4D input tensor expected but got: %s", ndim);

  long nInputPlane = weight->size(1);                                                              // weight->size[1] => weight->size(1)
  long inputHeight = input->size(dimh);                                                            // input->size[dimh] => input->size(dimh) 
  long inputWidth = input->size(dimw);                                                             // input->size[dimw] => input->size(dimw) 
  long nOutputPlane = weight->size(0);                                                             // weight->size[0] => weight->size(0)    
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  if (outputWidth < 1 || outputHeight < 1)
    THError(
        "Given input size: (%ld x %ld x %ld). "
        "Calculated output size: (%ld x %ld x %ld). Output size is too small",
        nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight,
        outputWidth);

  THArgCheck((inputHeight >= kH && inputWidth >= kW), 2,
             "input image is smaller than kernel");

/////////check depth map shape /////////

  int ndim_depth = input_depth->dim();                                                         // input_depth->nDimension => input_depth_dim()                          
  int dimf_depth = 0;
  int dimh_depth = 1;
  int dimw_depth = 2;

  if (ndim_depth == 4) {
    dimf_depth++;
    dimh_depth++;
    dimw_depth++;
  }

  THArgCheck(ndim_depth == 3 || ndim_depth == 4, 3,
             "3D input depth tensor expected but got: %s", ndim);

  long inputHeight_depth = input_depth->size(dimh_depth);                                   // input_depth->size[dimh_depth] => input_depth->size(dimh_depth)
  long inputWidth_depth = input_depth->size(dimw_depth);                                    // input_depth->size[dimw_depth] => input_depth->size(dimw_depth)

  THArgCheck(input_depth->size(1) == 1, 3,                                          // input_depth->size[1] => input_depth->size(1)
             "input depth should have only 1 channel",
             nInputPlane, input->size(1));                                          // input_depth->size[1] => input_depth->size(1)

  THArgCheck((inputHeight == inputHeight_depth && inputWidth == inputWidth_depth), 3,
             "input image and input depth should be the same size");
//////////////////////////////////////////

  if (gradOutput != NULL) {
    THArgCheck(gradOutput->size(dimf) == nOutputPlane, 4,                           // gradOutput->size[dimf] => gradOutput->size(dimf)
               "invalid number of gradOutput planes, expected: %d, but got: %d",
               nOutputPlane, gradOutput->size(dimf));                               // gradOutput->size[dimf] => gradOutput->size(dimf)

    THArgCheck((gradOutput->size(dimh) == outputHeight &&                           // gradOutput->size[dimh] => gradOutput->size(dimh)
                gradOutput->size(dimw) == outputWidth),                             // gradOutput->size[dimw] => gradOutput->size(dimw)
               4, "invalid size of gradOutput, expected height: %d width: %d , but got height: %d width: %d", outputHeight, outputWidth,
               gradOutput->size(dimh), gradOutput->size(dimw));                     // gradOutput->size[dimh/w] => gradOutput->size(dimh/w)
  }
}

int depthconv_forward_cuda(at::Tensor *input, at::Tensor *input_depth, at::Tensor *weight, at::Tensor *bias, at::Tensor *output,
                             at::Tensor *columns, at::Tensor *ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationH, int dilationW) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 7, input, input_depth, weight, output, columns, ones, bias));

  shape_check(state, input, input_depth, NULL, weight, bias, kH, kW, dH, dW, padH, padW,
              dilationH, dilationW);

  *input = input->contiguous();                                          //THCudaTensor_newContiguous(state, input);
  *input_depth = input_depth->contiguous();                              //THCudaTensor_newContiguous(state, input_depth);
  *weight = weight->contiguous();                                        //THCudaTensor_newContiguous(state, weight);

  int batch = 1;
  if (input->dim() == 3) {                                                        // input->nDimension => input->dim()
    // Force batch
    batch = 0;
    
    input->resize_({1, input->size(0), input->size(1), input->size(2)});
    input_depth->resize_({1, input->size(0), input->size(1), input->size(2)});
    

    //CHANGES END
  }

  long batchSize = input->size(0);                                                  // input->size[0] => input->size(0)
  long nInputPlane = input->size(1);                                                // input->size[1] => input->size(1)
  long inputHeight = input->size(2);                                                // input->size[2] => input->size(2)
  long inputWidth = input->size(3);                                                 // input->size[3] => input->size(3)

  long nOutputPlane = weight->size(0);                                              // weight->size[0] => weight->size(0)

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  *bias = bias ? bias->contiguous() : *bias;  //                                        THCudaTensor_newContiguous(state, bias) : bias;

  output->resize_({batchSize, nOutputPlane, outputHeight, outputWidth});
  columns->resize_({nInputPlane * kW * kW, outputHeight * outputWidth});


  if (ones->dim() != 2 ||                                                      // ones->nDimension => ones->dim()
      ones->size(0) * ones->size(1) < outputHeight * outputWidth) {             // ones->size[0/1] => ones->size(0/1)
    
    
    
    ones->resize_({outputHeight, outputWidth});

    // THCudaTensor_fill(state, ones, 1);
    ones->fill_(1);
  }


  at::Tensor input_n; //= at::new_empty();
  at::Tensor depth_n;// = new_empty();
  at::Tensor output_n;// = new_empty();

  // CHANGES END

  for (int elt = 0; elt < batchSize; elt++) {
    input_n = input->select(0, elt);
    depth_n = input_depth->select(0, elt);
    output_n = output->select(0, elt);

  
     long m_ = nOutputPlane;
     long n_ = outputHeight * outputWidth;
     long k_ = 1;

     if (bias) {


        THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
                        ones->data_ptr<float>(), k_,
                        bias->data_ptr<float>(), k_, 0.0f,
                        output_n.data_ptr<float>(), n_);

     } else {
      output_n.fill_(0);
     }

    depthconv_im2col(

        THCState_getCurrentStream(state), input_n.data_ptr<float>(), depth_n.data_ptr<float>(), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, columns->data_ptr<float>());

    long m = nOutputPlane;
    long n = columns->size(1);                                                 // columns->size[1] => columns->size(1)
    long k = nInputPlane * kH * kW;

    THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
                     columns->data_ptr<float>(), n,
                     weight->data_ptr<float>(), k, 1.0f,
                     output_n.data_ptr<float>(), n);
  }

  delete &input_n;
  delete &depth_n;
  delete &output_n;

  if (batch == 0) {

    output->resize_({nOutputPlane, outputHeight, outputWidth});
    input->resize_({nInputPlane, inputHeight, inputWidth});
  }


  delete input;
  delete input_depth;
  delete weight;  

  // if (bias) THCudaTensor_free(state, bias);
  if (bias) delete bias;

  return 1;
}

int depthconv_backward_input_cuda(
    at::Tensor *input, at::Tensor *input_depth, at::Tensor *gradOutput,
    at::Tensor *gradInput, at::Tensor *weight,
    at::Tensor *columns, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationH, int dilationW) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 6, input, input_depth, gradOutput, weight, columns, gradInput));

  shape_check(state, input, input_depth, gradOutput, weight, NULL, kH, kW, dH, dW, padH,
              padW, dilationH, dilationW);

  *input = input->contiguous();                                                        //THCudaTensor_newContiguous(state, input);
  *input_depth = input_depth->contiguous();                                            //THCudaTensor_newContiguous(state, input_depth);
  *gradOutput = gradOutput->contiguous();                                              //THCudaTensor_newContiguous(state, gradOutput);
  *weight = weight->contiguous();                                                       //THCudaTensor_newContiguous(state, weight);

  int batch = 1;
  if (input->dim() == 3) {                                                  // input->nDimension => input->dim()
    // Force batch
    batch = 0;


    input->resize_({1, input->size(0), input->size(1), input->size(2)});
    gradOutput->resize_({1, gradOutput->size(0), gradOutput->size(1), gradOutput->size(2)});

    // CHANGES END
  }

  long batchSize = input->size(0);                                          // input->size[0] => input->size(0)
  long nInputPlane = input->size(1);                                        // input->size[1] => input->size(1)
  long inputHeight = input->size(2);                                        // input->size[2] => input->size(2)
  long inputWidth = input->size(3);                                         // input->size[3] => input->size(3)

  long nOutputPlane = weight->size(0);                                      // weight->size[0] => weight->size(0)

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((input_depth->size(0) == batchSize), 3, "invalid batch size of input depth");      // input_depth->size[0] => input_depth->size(0)

  gradInput->resize_({batchSize, nInputPlane, inputHeight, inputWidth});
  columns->resize_({nInputPlane * kW * kH, outputHeight * outputWidth});

  at::Tensor gradInput_n; // = new_empty(state);
  at::Tensor input_depth_n; // = new_empty(state);
  at::Tensor gradOutput_n; // = new_empty(state);

  // CHANGES END

  for (int elt = 0; elt < batchSize; elt++) {

    gradInput_n = gradInput->select(0, elt);
    input_depth_n = input_depth->select(0, elt);
    gradOutput_n = gradOutput->select(0, elt);
    // CHANGES END


    long m = nInputPlane * kW * kH;
    long n = columns->size(1);                                  // columns->size[1] => columns->size(1)
    long k = nOutputPlane;

    THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
                     gradOutput_n.data_ptr<float>(), n,
                     weight->data_ptr<float>(), m, 0.0f,
                     columns->data_ptr<float>(), n);

    depthconv_col2im(
        THCState_getCurrentStream(state), columns->data_ptr<float>(),
        input_depth_n.data_ptr<float>(), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, gradInput_n.data_ptr<float>());
  }


  delete &gradInput_n;
  delete &input_depth_n;
  delete &gradOutput_n;

  if (batch == 0) {


    gradOutput->resize_({nOutputPlane, outputHeight, outputWidth});
    input->resize_({nInputPlane, inputHeight, inputWidth});
    input_depth->resize_({1, inputHeight, inputWidth});
    gradInput->resize_({nInputPlane, inputHeight, inputWidth});

  }


  delete input;
  delete input_depth;
  delete gradOutput;
  delete weight;

  return 1;
}

int depthconv_backward_parameters_cuda(
    at::Tensor *input, at::Tensor *input_depth, at::Tensor *gradOutput,
    at::Tensor *gradWeight, at::Tensor *gradBias,
    at::Tensor *columns, at::Tensor *ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationH, int dilationW,
    float scale) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 7, input, input_depth, gradOutput,
                                         gradWeight, gradBias, columns, ones));

  shape_check(state, input, input_depth, gradOutput, gradWeight, gradBias, kH, kW, dH, dW,
              padH, padW, dilationH, dilationW);

  *input = input->contiguous();                                                  //THCudaTensor_newContiguous(state, input);
  *input_depth = input_depth->contiguous();                                      //THCudaTensor_newContiguous(state, input_depth);
  *gradOutput = gradOutput->contiguous();                                        // THCudaTensor_newContiguous(state, gradOutput);

  int batch = 1;
  if (input->dim() == 3) {                          // input->nDimension  => input->dim()
    // Force batch
    batch = 0;

    input->resize_({1, input->size(0), input->size(1), input->size(2)});
    gradOutput->resize_({1, gradOutput->size(0), gradOutput->size(1), gradOutput->size(2)});


    // CHANGES END
  }

  long batchSize = input->size(0);                                              // input->size[0] => input->size(0)
  long nInputPlane = input->size(1);                                            // input->size[1] => input->size(1)    
  long inputHeight = input->size(2);                                            // input->size[2] => input->size(2)
  long inputWidth = input->size(3);                                             // input->size[3] => input->size(3)

  long nOutputPlane = gradWeight->size(0);                                      // gradWeight->size[0] => gradWeight->size(0)

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;


  // Define a buffer of ones, for bias accumulation
  if (ones->dim() != 2 ||                                                       // ones->nDimesion => ones->dim()
      ones->size(0) * ones->size(1) < outputHeight * outputWidth) {             // ones->size[0/1] => ones->size(0/1)
      // CHANGES START

    ones->resize_({outputHeight, outputWidth});
    
    ones->fill_(1);

  }

  columns->resize_({nInputPlane * kW * kH, outputHeight * outputWidth});

  at::Tensor input_n; // = new_empty();
  at::Tensor depth_n; // = new_empty();
  at::Tensor gradOutput_n; // = new_empty();

  for (int elt = 0; elt < batchSize; elt++) {

    input_n = input->select(0, elt);
    depth_n = input_depth->select(0, elt);
    gradOutput_n = gradOutput->select(0, elt);

    depthconv_im2col(
        THCState_getCurrentStream(state), input_n.data_ptr<float>(),
        depth_n.data_ptr<float>(), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, columns->data_ptr<float>());

    long m = nOutputPlane;
    long n = nInputPlane * kW * kH;
    long k = columns->size(1);                                              // long k = columns->size[1];

    THCudaBlas_Sgemm(state, 't', 'n', n, m, k, scale,
                     columns->data_ptr<float>(), k,
                     gradOutput_n.data_ptr<float>(), k, 1.0f,
                     gradWeight->data_ptr<float>(), n);
    long m_ = nOutputPlane;
    long k_ = outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    if (gradBias)

        THCudaBlas_Sgemv(
          state,
          't',
          k_, m_,
          scale,
          gradOutput_n.data_ptr<float>(), k_,
          ones->data_ptr<float>(), 1, 1.0f,
          gradBias->data_ptr<float>(), 1);

  }

  delete &input_n;
  delete &depth_n;
  delete &gradOutput_n;

  if (batch == 0) {


    gradOutput->resize_({nOutputPlane, outputHeight,
                          outputWidth});
    input->resize_({nInputPlane, inputHeight, inputWidth});

  }

  delete input;
  delete input_depth;
  delete gradOutput;
  return 1;
}

PYBIND11_MODULE(depthconv_cuda_cpp, m) {
  m.def("forward", &depthconv_forward_cuda, "depthconv forward");
  m.def("backward_input", &depthconv_backward_input_cuda, "depthconv backward input");
  m.def("backward_parameters", &depthconv_backward_parameters_cuda, "depthconv backward parameters");
}
