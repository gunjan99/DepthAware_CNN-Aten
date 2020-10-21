#include <THC/THC.h>
#include<torch/torch.h>
#include<iostream>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>


// template <typename T>

int depthavgpooling_forward_cuda(at::Tensor *input,
           at::Tensor *input_depth,
           at::Tensor *output,
           at::Tensor *depthweightcount,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) ;

int depthavgpooling_backward_input_cuda(
           at::Tensor *input,
           at::Tensor *input_depth,
           at::Tensor *depthweightcount,
           at::Tensor *gradOutput,
           at::Tensor *gradInput,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) ;

void AvePoolForward(cudaStream_t stream, const int count,
    const float* const input_data, const float* const input_depth_data,const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    float* const top_data, float* const depth_weight_count);

void AvePoolBackward(cudaStream_t stream, const int count, const float* const gradOutput,const float* const input_depth,const float* const depth_weight_count,
    const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    float* const bottom_diff);

extern THCState *state;  


void shape_check(THCState *state,
  at::Tensor *input, at::Tensor *input_depth,at::Tensor *depthweightcount, at::Tensor *gradOutput,
  int kH, int kW, int dH, int dW, int padH, int padW) {

  THArgCheck(kW > 0 && kH > 0, 5,                                                          
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 8,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);

  int ndim = input->dim();                                                              // nDimension -> dim() 
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THArgCheck(ndim == 3 || ndim == 4, 2,                                                        
             "3D or 4D input tensor expected but got: %d",
             ndim);

  long nInputPlane = input->size(dimh-1);                                             // size[dimh-1] -> size(dimh-1)
  long nInputRows = input->size(dimh);                                                // size[dimh] -> size(dimh)
  long nInputCols = input->size(dimw);                                                // size[dimw] -> size(dimw)      
  long nOutputRows, nOutputCols;
  long nOutputPlane = nInputPlane;


/////////check depth map shape /////////

  int ndim_depth = input_depth->dim();                                                // input_depth->nDimension => input_depth->dim()
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

  long inputHeight_depth = input_depth->size(dimh_depth);                             // size[dimh_depth] -> size(dimh_depth)
  long inputWidth_depth = input_depth->size(dimw_depth);                              // size[dimw_depth] -> size(dimw_depth)

  THArgCheck(input_depth->size(1) == 1, 3,                                            // size[1] -> size(1)
             "input depth should have only 1 channel",
             nInputPlane, input->size(1));                                            //size[1] -> size(1)

  THArgCheck((nInputRows == inputHeight_depth && nInputCols == inputWidth_depth), 3,
             "input image and input depth should be the same size, but got: weightcount(%d,%d), depth(%d,%d)",
             nInputRows, inputHeight_depth, nInputCols, inputWidth_depth);

  if (depthweightcount!=NULL){
      THArgCheck(depthweightcount->size(1) == 1, 3,                                   //size[1] -> size(1)
                 "input depth should have only 1 channel",
                 nInputPlane, input->size(1));                                        //size[1] -> size(1)

      THArgCheck((inputHeight_depth == depthweightcount->size(2) && inputWidth_depth == depthweightcount->size(3)), 3,              //depthweightcount->size[2] -> depthweightcount->size(2); depthweightcount->size[3] -> depthweightcount->size(3)
                 "input depth and input depthweightcount should be the same size, but got: weightcount(%d,%d), depth(%d,%d)",
                 depthweightcount->size(dimh_depth), depthweightcount->size(dimw_depth), inputHeight_depth, inputWidth_depth);       //size[dimh_depth] -> size(dimh_depth)
  }

    nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((nOutputRows - 1)*dH >= nInputRows + padH)
      --nOutputRows;
    if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
      --nOutputCols;
  }

  if (nOutputCols < 1 || nOutputRows < 1)
    THError("Given input size: (%dx%dx%d). "
            "Calculated output size: (%dx%dx%d). Output size is too small",
            nInputPlane,nInputRows,nInputCols,nInputPlane,nOutputRows,nOutputCols);

  if (gradOutput != NULL) {

    THArgCheck(gradOutput->size(dimf) == nOutputPlane, 4,                                         // size[dimf] -> size(dimf)
               "invalid number of gradOutput planes, expected: %d, but got: %d",
               nOutputPlane, gradOutput->size(dimf));                                             // size[dimf] -> size(dimf) 

    THArgCheck((gradOutput->size(dimh) == nOutputRows &&                                          // size[dimh] -> size(dimh)
                gradOutput->size(dimw) == nOutputCols),                                           // size[dimw] -> size(dimw)
               4, "invalid size of gradOutput, expected height: %d width: %d , but got height: %d width: %d", nOutputRows, nOutputCols,
               gradOutput->size(dimh), gradOutput->size(dimw));                                   // size[dimh] -> size(dimh); // size[dimw] -> size(dimw)
  }
}



int depthavgpooling_forward_cuda(at::Tensor* input,
           at::Tensor* input_depth,
           at::Tensor* output,
           at::Tensor *depthweightcount,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) {
          // THCudaTensor *inputx = (THCudaTensor *) input;
  THCAssertSameGPU(THCudaTensor_checkGPU(state, 4, input, input_depth, output, depthweightcount));
  shape_check(state, input, input_depth, NULL, NULL, kH, kW, dH, dW,
        padH, padW);

  // input = THCudaTensor_newContiguous(state, input);
  *input = input->contiguous();
  // input_depth = THCudaTensor_newContiguous(state, input_depth);
  *input_depth = input_depth->contiguous();

  int batch = 1;
  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;

  if (input->dim() == 3) {                                    // input->nDimension -> input->dim()
    nInputCols = input->size(2);                             //size[2] -> size(2)
    nInputRows = input->size(1);                             //size[2] -> size(2)
    nInputPlane = input->size(0);                            //size[0] ->size(0)
    batchSize = 1;
    batch = 0;
    input->resize_({1, input->size(0), input->size(1), input->size(2)});     /* input->size[0], input->size[1], input->size[2] -->  input->size(0), input->size(1), input->size(2)*/
    input_depth->resize_({1, input_depth->size(0), input_depth->size(1), input_depth->size(2)}); /*input_depth->size[0], input_depth->size[1], input_depth->size[2] --> input_depth->size(0), input_depth->size(1), input_depth->size(2)*/
  }
  else
  {
    nInputCols = input->size(3);  //input->size[3] ---> input->size(3)
    nInputRows = input->size(2);  //input->size[2] ---> input->size(2)
    nInputPlane = input->size(1);  //input->size[1] ---> input->size(1)
    batchSize = input->size(0);    //input->size[0] ---> input->size(0)
  }

  nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
  nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;


  if (padW || padH)
  {
    if ((nOutputRows - 1)*dH >= nInputRows + padH)
      --nOutputRows;
    if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
      --nOutputCols;
  }

  output->resize_({batchSize, nInputPlane, nOutputRows, nOutputCols});
  depthweightcount->resize_({batchSize, 1, nInputRows, nInputCols});
  at::Tensor input_n;
  at::Tensor depth_n;
  at::Tensor depthweightcount_n;
  at::Tensor output_n;
  for (int elt = 0; elt < batchSize; elt++) {
    input_n = input->select(0, elt);
    depth_n = input_depth->select(0, elt);
    depthweightcount_n = depthweightcount->select(0, elt);
    output_n = output->select(0, elt);
    int count = output_n.numel();

    AvePoolForward(THCState_getCurrentStream(state),                              // THCState_getCurrentStream(state) -> at::cuda::getCurrentCUDAStream(state)
        count, input_n.data_ptr<float>(), depth_n.data_ptr<float>(),
        nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
        kH, kW, dH, dW, padH, padW, output_n.data_ptr<float>(), depthweightcount_n.data_ptr<float>());

    THCudaCheck(cudaGetLastError());
  }

  delete &input_n;
  delete &depth_n;
  delete &depthweightcount_n;
  delete &output_n;

  if(batch == 0){
    output->resize_({nInputPlane, nOutputRows, nOutputCols});
    input->resize_({nInputPlane, nInputRows, nInputCols});
  }
  delete input;
  delete input_depth;
}


int depthavgpooling_backward_input_cuda(
           at::Tensor *input,
           at::Tensor *input_depth,
           at::Tensor *depthweightcount,
           at::Tensor *gradOutput,
           at::Tensor *gradInput,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 4, input, input_depth, gradOutput, gradInput, depthweightcount));
  shape_check(state, input, input_depth, depthweightcount, gradOutput, kH, kW, dH, dW,
        padH, padW);

  *input = input->contiguous();
  *input_depth = input_depth->contiguous();
  *gradOutput = gradOutput->contiguous();
  *depthweightcount = depthweightcount->contiguous();

  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;
  int dimCol = 2;
  int dimRow = 1;

  int batch = 1;
  if (input->dim() == 3) {         
    nInputPlane = input->size(0);   
    batchSize = 1;
    batch = 0;
    input->resize_({1, input->size(0), input->size(1),input->size(2)});
    gradOutput->resize_({1, gradOutput->size(0), gradOutput->size(1), gradOutput->size(2)});
  }
  else
  {
    dimCol = 3;
    dimRow = 2;
    nInputPlane = input->size(1); 
    batchSize = input->size(0);     
  }
  nInputCols = input->size(dimCol);  
  nInputRows = input->size(dimRow);  

  nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
  nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  if (padW || padH)
  {

    if ((nOutputRows - 1)*dH >= nInputRows + padH)
      --nOutputRows;
    if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
      --nOutputCols;
  }

  THArgCheck((input_depth->size(0) == batchSize), 3, "invalid batch size of input depth");  //input_depth->size[0] ---> input_depth->size(0)
  // THCudaTensor_resizeAs(state, gradInput, input);
  gradInput->resize_({input->size(0), input->size(1), input->size(2), input->size(3)});

  at::Tensor gradInput_n;
  at::Tensor depth_n;
  at::Tensor gradOutput_n;
  at::Tensor depthweightcount_n;

  for (int elt = 0; elt < batchSize; elt++) {
    gradInput_n = gradInput->select(0, elt);
    depth_n = input_depth->select(0, elt);
    gradOutput_n = gradOutput->select(0, elt);
    depthweightcount_n = depthweightcount->select(0, elt);

    int count = gradInput_n.numel();

    AvePoolBackward
      (THCState_getCurrentStream(state), count,                                      // // THCState_getCurrentStream(state) -> at::cuda::getCurrentCUDAStream(state)
        gradInput_n.data_ptr<float>(), depth_n.data_ptr<float>(), depthweightcount_n.data_ptr<float>(),
        nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
        kH, kW, dH, dW, padH, padW,
        gradInput_n.data_ptr<float>());
    THCudaCheck(cudaGetLastError());
  }
  delete &gradInput_n;
  delete &depth_n;
  delete &gradOutput_n;
  delete &depthweightcount_n;

  if (batch == 0) {

    gradOutput->resize_({nInputPlane, nOutputRows, nOutputCols});
    input->resize_({nInputPlane, nInputRows, nInputCols});
    input_depth->resize_({1, nInputRows, nInputCols});
    gradInput->resize_({nInputPlane, nInputRows,nInputCols});
  }

  // clean
  delete input;
  delete input_depth;
  delete depthweightcount;
  delete gradOutput;
}



PYBIND11_MODULE(depthavgpooling_cuda_cpp, m) {
  m.def("forward", &depthavgpooling_forward_cuda, "forward");
  m.def("backward", &depthavgpooling_backward_input_cuda, "depthAvgPooling backward");
}