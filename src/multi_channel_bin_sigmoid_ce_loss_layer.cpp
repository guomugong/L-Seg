#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/multi_channel_bin_sigmoid_ce_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void MultiChannelBinSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  const MCBSCELossParameter  mcbsce_loss_param = this->layer_param_.mcbsce_loss_param();
  num_label_ = mcbsce_loss_param.num_label();
	key_ = mcbsce_loss_param.key();
}

template <typename Dtype>
void MultiChannelBinSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->width()*bottom[0]->height(), bottom[1]->width()*bottom[1]->height()) <<
    "MULTICHANNEL_BIN_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same spatial dimension.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiChannelBinSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  
  //lt
  const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
  Dtype *temp_count_pos = new Dtype[num_label_];
  Dtype *temp_neg_count = new Dtype[num_label_];
  Dtype (*temp_count_neg)[5] = new Dtype[num_label_][5]; 
  Dtype (*temp_neg_loss)[5] = new Dtype[num_label_][5];
  Dtype *temp_pos_loss = new Dtype[num_label_];
  Dtype *temp_loss_neg = new Dtype[num_label_];
  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  Dtype count_pos = 0;
  Dtype count_neg = 0;
  
    //chu shi hua
  for(int i = 0; i < num_label_; i++){
	  temp_count_pos[i] = 0;
	  temp_neg_count[i] = 0;
	  temp_pos_loss[i] = 0;
	  temp_loss_neg[i] = 0;
	  for(int j = 0; j< 5;j++){
		  temp_count_neg[i][j] = 0;
		  temp_neg_loss[i][j] = 0;
	  }
  }
  int dim = bottom[0]->height()*bottom[0]->width();
    //jin xing tong ji
    for (int i = 0; i < num_label_; ++i) { /* loop over channels */
      for (int j = 0; j < dim; ++j) { /* loop over pixels */
	      int idx = i*dim+j;
		  Dtype temp = log(1 + exp(input_data[idx] - 2 * input_data[idx] * (input_data[idx] >= 0)));
      	if (target[j] == (i+1)) {
			temp_count_pos[i] = temp_count_pos[i] + key_;
			temp_pos_loss[i] -=input_data[idx] * (1 - (input_data[idx] >= 0)) - temp;
		}else{
			Dtype gailv = sigmoid_output_data[idx];
			if(gailv >=0 && gailv < 0.2){
				temp_count_neg[i][0]++;
				temp_neg_loss[i][0] -=input_data[idx] * (0 - (input_data[idx] >= 0)) - temp;
			}else if(gailv >=0.2 && gailv < 0.4){
				temp_count_neg[i][1]++;
				temp_neg_loss[i][1] -=input_data[idx] * (0 - (input_data[idx] >= 0)) - temp;
			}else if(gailv >=0.4 && gailv < 0.6){
				temp_count_neg[i][2]++;
				temp_neg_loss[i][2] -=input_data[idx] * (0 - (input_data[idx] >= 0)) - temp;
			}else if(gailv >=0.6 && gailv < 0.8){
				temp_count_neg[i][3]++;
				temp_neg_loss[i][3] -=input_data[idx] * (0 - (input_data[idx] >= 0)) - temp;
			}else if(gailv >=0.8 && gailv <= 1.0){
				temp_count_neg[i][4]++;
				temp_neg_loss[i][4] -=input_data[idx] * (0 - (input_data[idx] >= 0)) - temp;
			}
		}
	  }
	}
 // zai ci que ding  temp_neg_count  da xiao 
 for(int i = 0; i < num_label_;i++){
	 Dtype p_c = temp_count_pos[i];
	 if(p_c == 0){
		 for(int j = 0;j < 5; j++){
			 temp_neg_count[i] += temp_count_neg[i][j];
			 temp_loss_neg[i] +=temp_neg_loss[i][j];
		 }
	 }else if(p_c <= temp_count_neg[i][4]){
		 temp_neg_count[i] = temp_count_neg[i][4];
		 temp_loss_neg[i] =temp_neg_loss[i][4];
	 }else if(p_c <= temp_count_neg[i][4] + temp_count_neg[i][3]){
		 temp_neg_count[i] = temp_count_neg[i][4] + temp_count_neg[i][3];
		 temp_loss_neg[i] =temp_neg_loss[i][4] + temp_neg_loss[i][3];
	 }else if(p_c <= temp_count_neg[i][4] + temp_count_neg[i][3] + temp_count_neg[i][2]){
		 temp_neg_count[i] = temp_count_neg[i][4] + temp_count_neg[i][3] + temp_count_neg[i][2];
		 temp_loss_neg[i] =temp_neg_loss[i][4] + temp_neg_loss[i][3] + temp_neg_loss[i][2];
	 }else if(p_c <= temp_count_neg[i][4] + temp_count_neg[i][3] + temp_count_neg[i][2] + temp_count_neg[i][1]){
		 temp_neg_count[i] = temp_count_neg[i][4] + temp_count_neg[i][3] + temp_count_neg[i][2] + temp_count_neg[i][1];
		 temp_loss_neg[i] =temp_neg_loss[i][4] + temp_neg_loss[i][3] + temp_neg_loss[i][2] + temp_neg_loss[i][1];
	 }else{
		 temp_neg_count[i] = temp_count_neg[i][4] + temp_count_neg[i][3] + temp_count_neg[i][2] + temp_count_neg[i][1] + temp_count_neg[i][0];
		 temp_loss_neg[i] =temp_neg_loss[i][4] + temp_neg_loss[i][3] + temp_neg_loss[i][2] + temp_neg_loss[i][1] + temp_neg_loss[i][0];
	 } 
 }
 
 for(int i = 0; i < num_label_;i++){
	 Dtype temp = temp_count_pos[i]/key_;
	 count_pos += temp;
	 count_neg += (int)(temp_neg_count[i]/4);
 }
  for(int i = 0; i < num_label_;i++){
	 //Dtype temp = temp_count_pos[i]/key_;
	 loss_pos +=  temp_pos_loss[i] * count_neg/(count_pos+count_neg);
	 loss_neg +=  temp_loss_neg[i] * count_pos/(count_neg+count_pos);
 }
  //
  top[0]->mutable_cpu_data()[0] = (loss_pos + loss_neg);
}

template <typename Dtype>
void MultiChannelBinSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    //lt
	int dim = bottom[0]->height()*bottom[0]->width();
	Dtype *temp_count_pos = new Dtype[num_label_];
	Dtype *temp_neg_count = new Dtype[num_label_];
	Dtype (*temp_count_neg)[5] = new Dtype[num_label_][5]; 
	Dtype (*temp_neg_loss)[5] = new Dtype[num_label_][5];
	Dtype *temp_pos_loss = new Dtype[num_label_];
	Dtype *temp_loss_neg = new Dtype[num_label_];
	Dtype count_pos = 0;
	Dtype count_neg = 0;
	//chu shi hua
	for(int i = 0; i < num_label_; i++){
	  temp_count_pos[i] = 0;
	  temp_neg_count[i] = 0;
	  temp_pos_loss[i] = 0;
	  temp_loss_neg[i] = 0;
	  for(int j = 0; j< 5;j++){
		  temp_count_neg[i][j] = 0;
		  temp_neg_loss[i][j] = 0;
	  }
	}
	//jin xing tong ji
    for (int i = 0; i < num_label_; ++i) { /* loop over channels */
      for (int j = 0; j < dim; ++j) { /* loop over pixels */
	      int idx = i*dim+j;
      	if (target[j] == (i+1)) {
			temp_count_pos[i] = temp_count_pos[i] + key_;
		}else{
			Dtype gailv = sigmoid_output_data[idx];
			if(gailv >=0 && gailv < 0.2){
				temp_count_neg[i][0]++;
			}else if(gailv >=0.2 && gailv < 0.4){
				temp_count_neg[i][1]++;
			}else if(gailv >=0.4 && gailv < 0.6){
				temp_count_neg[i][2]++;
			}else if(gailv >=0.6 && gailv < 0.8){
				temp_count_neg[i][3]++;
			}else if(gailv >=0.8 && gailv <= 1.0){
				temp_count_neg[i][4]++;
			}
		}
	  }
	}
 // zai ci que ding  temp_neg_count  da xiao 
 for(int i = 0; i < num_label_;i++){
	 Dtype p_c = temp_count_pos[i];
	 if(p_c == 0){
		 for(int j = 0;j < 5; j++){
			 temp_neg_count[i] += temp_count_neg[i][j];
		 }
	 }else if(p_c <= temp_count_neg[i][4]){
		 temp_neg_count[i] = temp_count_neg[i][4];
	 }else if(p_c <= temp_count_neg[i][4] + temp_count_neg[i][3]){
		 temp_neg_count[i] = temp_count_neg[i][4] + temp_count_neg[i][3];
	 }else if(p_c <= temp_count_neg[i][4] + temp_count_neg[i][3] + temp_count_neg[i][2]){
		 temp_neg_count[i] = temp_count_neg[i][4] + temp_count_neg[i][3] + temp_count_neg[i][2];
	 }else if(p_c <= temp_count_neg[i][4] + temp_count_neg[i][3] + temp_count_neg[i][2] + temp_count_neg[i][1]){
		 temp_neg_count[i] = temp_count_neg[i][4] + temp_count_neg[i][3] + temp_count_neg[i][2] + temp_count_neg[i][1];
	 }else{
		 temp_neg_count[i] = temp_count_neg[i][4] + temp_count_neg[i][3] + temp_count_neg[i][2] + temp_count_neg[i][1] + temp_count_neg[i][0];
	 } 
 }
	
for(int i = 0; i < num_label_;i++){
	 Dtype temp = temp_count_pos[i]/key_;
	 count_pos += temp;
	 count_neg += (int)(temp_neg_count[i]/4);
 }
	//
	  /* calculate gradient */
    for (int i = 0; i < num_label_; ++i) { /* loop over channels */
	  Dtype p_c = temp_count_pos[i];
	  for (int j = 0; j < dim; ++j) { /* loop over pixels */
	      int idx = i*dim+j;
      	if (target[j] == (i+1)) {
	  	    bottom_diff[idx] = sigmoid_output_data[idx] - 1;
			bottom_diff[idx] *= count_neg/(count_neg + count_pos ); /* weight_pos_ was calculated in forward phase */
      	} else {
			Dtype gailv = sigmoid_output_data[idx];
			if(p_c == 0){
				bottom_diff[idx] = sigmoid_output_data[idx] - 0;
				bottom_diff[idx] *= count_pos/(count_pos+count_neg);
			}else if(p_c <= temp_count_neg[i][4]){
				if(gailv>= 0.8 && gailv<= 1.0){
					bottom_diff[idx] = sigmoid_output_data[idx] - 0;
					bottom_diff[idx] *= count_pos/(count_pos+count_neg);
				}else{
					bottom_diff[idx] = 0;
				}
			}else if(p_c <= temp_count_neg[i][4] + temp_count_neg[i][3]){
				if(gailv >= 0.6 && gailv <=1.0){
					bottom_diff[idx] = sigmoid_output_data[idx] - 0;
					bottom_diff[idx] *= count_pos/(count_pos+count_neg);
				}else{
					bottom_diff[idx] = 0;
				}
			}else if(p_c <= temp_count_neg[i][4] + temp_count_neg[i][3] + temp_count_neg[i][2]){
				if(gailv >= 0.4 && gailv <=1.0){
					bottom_diff[idx] = sigmoid_output_data[idx] - 0;
					bottom_diff[idx] *= count_pos/(count_pos+count_neg);
				}else{
					bottom_diff[idx] = 0;
				}
			}else if(p_c <= temp_count_neg[i][4] + temp_count_neg[i][3] + temp_count_neg[i][2]){
				if(gailv >= 0.2 && gailv <= 1.0){
					bottom_diff[idx] = sigmoid_output_data[idx] - 0;
					bottom_diff[idx] *= count_pos/(count_pos+count_neg);
				}else{
					bottom_diff[idx] = 0;
				}
			}else{
				bottom_diff[idx] = sigmoid_output_data[idx] - 0;
				bottom_diff[idx] *= count_pos/(count_pos+count_neg);
			}
      	}
      }
    }
	//

    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultiChannelBinSigmoidCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(MultiChannelBinSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(MultiChannelBinSigmoidCrossEntropyLoss);

}  // namespace caffe
