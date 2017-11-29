# Focal-Loss
loss layer of implementation.  
You can see "Focal Loss for Dense Object Detection" [arXiv](https://arxiv.org/abs/1708.02002) for more information.  

## Usage

```
// Focal Loss layer
optional FocalLossParameter focal_loss_param = 124;

// Focal Loss for Dense Object Detection
message FocalLossParameter {
  enum Type {
    ORIGIN = 0; // FL(p_t)  = -(1 - p_t) ^ gama * log(p_t), where p_t = p if y == 1 else 1 - p, whre p = sigmoid(x)
    LINEAR = 1; // FL*(p_t) = -log(p_t) / gama, where p_t = sigmoid(gama * x_t + beta), where x_t = x * y, y is the ground truth label {-1, 1}
  }
  optional Type type   = 1 [default = ORIGIN]; 
  optional float gamma = 2 [default = 2];
  // cross-categories weights to solve the imbalance problem
  optional float alpha = 3 [default = 0.25]; 
  optional float beta  = 4 [default = 1.0];
}

layer {
  name: "loss_cls"
  type: "FocalLoss"
  bottom: "cls_score"
  bottom: "labels"
  propagate_down: 1
  propagate_down: 0
  top: "loss_cls"
  include { phase: TRAIN }
  loss_weight: 1
  loss_param { ignore_label: -1 normalize: true }
  focal_loss_param { alpha: 0.25 gamma: 2 }
}
```

### Derivative
see https://github.com/zimenglan-sysu-512/paper-note/blob/master/focal_loss.pdf 

### Done
All categories share the same `alpha`.

### Sigmoid Form
Here use `softmax` instead of `sigmoid` function.  
If you want see how to use `sigmoid` to implement `Focal Loss`, please see https://github.com/sciencefans/Focal-Loss to get more information.  

### MXNet Repo
https://github.com/unsky/focal-loss
