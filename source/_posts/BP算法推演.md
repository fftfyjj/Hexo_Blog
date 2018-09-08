---
title: BP算法推演
typora-copy-images-to: BP算法推演
date: 2018-09-05 23:44:22
tags:
categories:
---

## BP算法的目的

计算 $ \partial J(\Theta) \over \partial \Theta_{ij}^{(l)} $  然后利用梯度下降法求解$min(J(\Theta))$ 时的 $\Theta$ 值

定义 $J(\Theta)$ 的计算公式如下：
$$
J(\Theta) = -{1 \over m}* \sum_{i=1}^m\sum_{k=1}^K[y_k^{(i)}*log((h_\Theta(x^{(i)}))_k) + (1-y_k^{(i)})*log(1-(h_\Theta(x^{(i)}))_k)] + {\lambda \over 2m} *\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}}(\Theta_{j,i}^{(l)})^2
$$

其中 $h_\Theta(x^{(i)})$ 是通过FP算法计算的：
$$
a^{(1)} = x^{(i)}
$$
$$
z_j^{(l)} = \sum_{i=0}^{s_{l-1}}a_i^{l-1}*\Theta_{j,i}^{(l-1)}
$$
$$
a_j^{l} = g(z_j^{(l)})
$$
$$
a_j^{L} = g(\sum_{i=0}^{s_{L-1}}a_i^{L-1}*\Theta_{j,i}^{(L-1)})
$$
$$
h_\Theta(x^{(i)}) = a^L
$$

如果将$J(\Theta)$中的$ h_\Theta(x^{(i)})_k $用FP算法中公式替换后求$ \partial J(\Theta) \over \partial \Theta_{ij}^{(l)} $ ，也是可以直接求解的，只是比较复杂。通过利用求导的链式法则，利用BP算法则可以高效的求导。



## 约定

### 定义$ \delta^{(l)} $ （这是一个向量）

为便于理解计算，定义$ \delta_i^{(l)} $ 为$l$层的第$i$个神经元节点的$\partial J(\Theta) \over \partial z_i$ 。



## 求解思路分析

### 求解模型

为便于理解，可以观察如下神经网络模型图：



观察图，可以得出如下结论：

通过该$\delta^{(l)}$则可以方便的计算出第$l-1$层对应的每个$\theta$的偏导 $\partial J(\Theta) \over \partial \Theta_{j,i}^{(l-1)}$ ，如果可以计算出$\delta^{(l-1)}$则可以依次计算出所有的$\theta$的偏导数。

### 如何求解$ \delta^{(l-1)}$ （这是一个向量）

$$
\delta_i^{(l-1)} = (\sum_{j=1}^{s^{(l)}}\delta_j^{(l)}{\partial z_j \over \partial a_i }) *g\ ^{'}(z_i) = (\sum_{j=1}^{s^{(l)}}\delta_j^{(l)}\Theta_{j,i}^{(l-1)}) *g\ ^{'}(z_i) \tag{1-1}
$$

向量版：
$$
\delta^{(l-1)} = (\Theta^{(l-1)})^T\delta^{(l)}.*g\ ^{'}(z^{(l-1)}) \tag{1-2}
$$
其中：
$$
g\ ^{'}(z^{(l-1)}) = g(z^{(l-1)}).*(1-g(z^{(l-1)})) = a^{(l-1)}.*(1-a^{(l-1)})
$$
综合即为：
$$
\delta^{(l-1)} = (\Theta^{(l-1)})^T\delta^{(l)}.*a^{(l-1)}.*(1-a^{(l-1)}) \tag{1-3}
$$
注意：此处向量运算时向量$a^{(l-1)} $中不包含标量$a_0^{(l-1)}$。



### 如何求解$ \partial J(\Theta) \over \partial \Theta_{j,i}^{(l)} $ （这是一个标量）

$$
{\partial J(\Theta) \over \partial \Theta_{j,i}^{(l)}} = \delta_j^{(l+1)}*{\partial z_j^{(l+1)}\over \partial \Theta_{j,i}^{(l)}} = \delta_j^{(l+1)}*a_i^{(l)}   \tag{2-1}
$$

其中$i$可以取0值。

### 如何求解$ \partial J(\Theta) \over \partial \Theta^{(l)} $  (这是一个矩阵)

$$
{\partial J(\Theta) \over \partial \Theta^{(l)}} = \delta^{(l+1)}*(a^{(l)})^T
$$



## 完整算法（常规版）VS (向量版)

### 常规版

对于样本空间中的第$t= \{1..m\}$个样本:

* 定义$a^{(1)} = x^{(t)} $。
* 利用FP算法，可以一直求解出$a^{(L)} = h_\Theta(x^{(t)}) = y^{(t)}$, $L$表示神经网络的输出层。
* 计算$\delta^{(L)}$， 得到 $ \delta^{(L)} = a^{(L)} - y^{(t)}$。
* 可以利用公式${\partial J(\Theta) \over \partial \Theta^{(l)}} = \delta^{(l+1)}\ast (a^{(l)})^T$ ，根据$\delta^{(l+1)}$计算出对应的该层的$\Theta$的偏导数${\partial J(\Theta) \over \partial \Theta^{(l)}}$。
* 同时可以利用公$\delta^{(l)} = (\Theta^{(l)})^T\delta^{(l+1)}.\ast a^{(l)}.\ast(1-a^{(l)})$推导出下层 $\delta$ 。
* 重复以上两步，即可求解出所有的偏导数。

重复针对每个样本都按以上步骤求值，相加每个样本的偏导数${\partial J(\Theta) \over \partial \Theta^{(l)}}$ 并除以样本数$m$，则为该神经网络模型第$l$层在整个样本空间内计算的平均偏导数$D^{(l)}$。
$$
D^{(l)} = {1\over m}*{\sum_{t=1}^m ({\partial J(\Theta) \over \partial \Theta^{(l)}})^{(t)}}
$$

### 向量版

计算步骤很简单：

定义m为样本的数量，$s_1$ 为nn模型中输入层的单元数(未考虑bias单元), $B$为加入bias单元的操作函数，$M$为去除bias单元的操作函数。

* 定义$A^{(1)} = X$，$X$为所有的样本构成的$m \times s_1$矩阵，$B_1(A^{(1)})$运算后得到一个$m\times(s_1+1)$的矩阵。
* 利用FP算法求解出$A^{(l)}$, 计算公式为：$A^{(l)} = g(B_1(A^{(l-1)})\ast \Theta^T)  $ ，得到一个$m\times s_l$的矩阵。
* 计算$\Delta^{(L)} = A^{(L)} - Y$ ，得到的是一个$m\times s_L$的矩阵。
* 利用BP思想可以计算：$ \Delta^{(l)} = \Delta^{(l+1)}\ast M(\Theta^{(l)}).\ast A^{(l)}.\ast (1-A^{(l)}) $ ，得到的是一个$ m\times s_l $的矩阵。
* 计算nn模型中第$l$层的样本空间平均偏导数：$ {1\over m} \ast ( \Delta^{(l)} )^T \ast {B_1}(A^{(l-1)}) $ , 得到的是一个 $s_l\times (s_{l-1} + 1)$矩阵。
* 如果是考虑了正则化，则nn模型中第$l$层的样本空间平均偏导数：$ \begin{gather}{1\over m}\ast ( \Delta^{(l)} )^T B_1(A^{(l-1)}) + {\lambda \over m}B_0(M(\Theta^{(l)})) \end{gather} $ , 得到的是一个$ s_l\times (s_{l-1} + 1) $矩阵。

## 实践

如下是吴恩达的《机器学习》课程中week5的练习题，效果还是不错：

```octave
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
mo = [1:num_labels]
o1 = ones(m,1);
o1t = o1';
o2 = ones(num_labels,1);
o2t = o2';
%Xt = [o1,X]
%X2t = [o1,(sigmoid(Xt*Theta1'))]
%h = sigmoid(X2t*Theta2');   此写法计算量超大
A1 = X;
A2 = sigmoid([ones(m,1), A1] * Theta1'); %5000*25
h = sigmoid([ones(m,1), A2] * Theta2'); %5000*10

Y = (y * o2t - o1 * mo) == 0
J = -1 * (o1t*((Y.*log(h) + (1-Y).*log(1 - h))*o2)) / m

T1 = Theta1(:,2:end)
T2 = Theta2(:,2:end)
reg = lambda * (sum((T1.^2)(:)) + sum((T2.^2)(:)))/(2*m)

J = J + reg;

A3 = h;
Z2 = [ones(m,1), A1] * Theta1'
Delta3 = A3 - Y;  %5000*10
Delta2 = (Delta3 * T2).*sigmoidGradient(Z2); %5000*25

rA2 = [ones(m,1), A2] %5000*26
rA1 = [ones(m,1), A1] %5000*401
DD2 = Delta3' * rA2  %10*26
DD1 = Delta2' * rA1  %25*401 

Theta1_grad = DD1/m
Theta2_grad = DD2/m

Theta1_reg = lambda*[zeros(hidden_layer_size,1), T1]/m
Theta2_reg = lambda*[zeros(num_labels,1), T2]/m
Theta1_grad = Theta1_grad + Theta1_reg;
Theta2_grad = Theta2_grad + Theta2_reg;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

```

TODO: 文章中有些地方表达不准确，不严谨，待进一步修正。