## 1.Multi-Layer Fully Connected Neural Networks

 **Inline Question 1: Suggested Answer**

Did you notice anything about the comparative difficulty of training the three-layer network vs. training the five-layer network? Which network seemed more sensitive to the initialization scale? Why?**

**回答:**
训练五层网络比训练三层网络更难，尤其是对初始化规模的敏感度更高。这是因为随着网络深度的增加，梯度消失或梯度爆炸的问题会更加严重。深层网络的参数初始化如果不合适，梯度在反向传播过程中可能会指数级地减小或增大，导致训练困难。因此，五层网络对初始化规模的要求更高，需要更精细的调整才能有效训练。

**Inline Question 2:** Jon notices that when he was training a network with AdaGrad that the updates became very small, and that his network was learning slowly. Using your knowledge of the AdaGrad update rule, why do you think the updates would become very small? Would Adam have the same issue?

**回答:**
AdaGrad的更新会变得非常小的原因是随着训练的进行，`cache`会不断累加梯度的平方，导致分母`np.sqrt(cache) + eps`越来越大。这使得学习率逐渐减小，最终更新变得非常小，网络的学习速度变慢。

Adam优化器通过引入动量和二阶动量的指数衰减来缓解这个问题。Adam在计算二阶动量时使用了指数衰减，避免了`cache`无限增大，因此Adam不会出现AdaGrad中更新变得非常小的问题。

### 补充代码：

init：

```python
    def __init__(
            self,
            hidden_dims,
            input_dim=3 * 32 * 32,
            num_classes=10,
            dropout_keep_ratio=1,
            normalization=None,
            reg=0.0,
            weight_scale=1e-2,
            dtype=np.float32,
            seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 获取所有层数的维度
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        # 初始化所有层的参数 (这里的层数是上面的layer_dims的长度减1,因此不会下标越界)
        for i in range(self.num_layers):
            self.params['W' + str(i + 1)] = np.random.normal(0, weight_scale, size=(layer_dims[i], layer_dims[i + 1]))
            self.params['b' + str(i + 1)] = np.zeros(layer_dims[i + 1])
            # 接下来添加batch normalization 层，注意最后一层不需要添加
            if self.normalization == 'batchnorm' and i < self.num_layers - 1:
                self.params['gamma' + str(i + 1)] = np.ones(layer_dims[i + 1])
                self.params['beta' + str(i + 1)] = np.zeros(layer_dims[i + 1])

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

```

loss：

```python 
    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 我们网络的结果是这样的 {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

        # 用一个变量保存上一层的输出
        layer_input = X
        caches = {}
        # 对前面 L - 1层进行操作，因为最后一层的操作和前面的不一样
        for i in range(1, self.num_layers):
            W = self.params['W' + str(i)]
            b = self.params['b' + str(i)]

            # 计算affine层的输出
            affine_out, affine_cache = affine_forward(layer_input, W, b)
            # 计算relu层的输出
            relu_out, relu_cache = relu_forward(affine_out)

            # 保存cache
            caches['affine_cache' + str(i)] = affine_cache
            caches['relu_cache' + str(i)] = relu_cache

            # 更新layer_input
            layer_input = relu_out

        # 最后一层的操作
        W = self.params['W' + str(self.num_layers)]
        b = self.params['b' + str(self.num_layers)]

        scores, affine_cache = affine_forward(layer_input, W, b)
        caches['affine_cache' + str(self.num_layers)] = affine_cache

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 计算loss
        loss, dscores = softmax_loss(scores, y)

        # 先计算最后一层的梯度
        W = self.params['W' + str(self.num_layers)]
        affine_cache = caches['affine_cache' + str(self.num_layers)]
        d_relu_out, dW, db = affine_backward(dscores, affine_cache)
        grads['W' + str(self.num_layers)] = dW + self.reg * W
        grads['b' + str(self.num_layers)] = db

        # 计算前面的梯度
        for i in range(self.num_layers - 1, 0, -1):
            W = self.params['W' + str(i)]
            affine_cache = caches['affine_cache' + str(i)]
            relu_cache = caches['relu_cache' + str(i)]

            # 先计算relu层的梯度
            d_affine_out = relu_backward(d_relu_out, relu_cache)
            # 再计算affine层的梯度
            d_relu_out, dW, db = affine_backward(d_affine_out, affine_cache)

            # 保存梯度
            grads['W' + str(i)] = dW + self.reg * W
            grads['b' + str(i)] = db
         
        # 加上正则化项
        for i in range(1, self.num_layers + 1):
            W = self.params['W' + str(i)]
            loss += 0.5 * self.reg * np.sum(W * W)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

```

sgd_momentum:

```python
def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v

    config["velocity"] = v

    return next_w, config

```

RMSProp:

```python
def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    cache = config["cache"]
    cache = config["decay_rate"] * cache + (1 - config["decay_rate"]) * dw ** 2
    next_w = w - config["learning_rate"] * dw / (np.sqrt(cache) + config["epsilon"])
    config["cache"] = cache


    return next_w, config

```

Adam:

```python
def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    t = config["t"] + 1
    m = config["beta1"] * config["m"] + (1 - config["beta1"]) * dw
    mt = m / (1 - config["beta1"] ** t)
    v = config["beta2"] * config["v"] + (1 - config["beta2"]) * dw ** 2
    vt = v / (1 - config["beta2"] ** t)
    next_w = w - config["learning_rate"] * mt / (np.sqrt(vt) + config["epsilon"])

    config["t"] = t
    config["m"] = m
    config["v"] = v


    return next_w, config

```

train you best model:

```python
best_model = None

################################################################################
# TODO: Train the best FullyConnectedNet that you can on CIFAR-10. You might   #
# find batch/layer normalization and dropout useful. Store your best model in  #
# the best_model variable.                                                     #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

best_val = -1
best_params = {}
# 重新定义一下训练数量
num_train = 500

small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
}

#随机训练30次，每次随机生成学习率和正则化强度，先用小数据集训练一下
for i in range(30):
    lr = 10 ** np.random.uniform(-5, -3)  # 学习速度
    ws = 10 ** np.random.uniform(-3, -1)  # 权重缩放
    reg = 10 ** np.random.uniform(-3, -1)  # 正则化强度

    # 创建一个四层模型
    model = FullyConnectedNet([256, 128, 64],
                              weight_scale=ws,
                              reg=reg)

    # 使用adam更新策略
    solver = Solver(model, small_data,
                    num_epochs=20, batch_size=256,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': lr,
                    },
                    verbose=False)
    solver.train()
    new_val = solver.best_val_acc

    if new_val > best_val:
        best_val = new_val
        best_model = model
        best_params['lr'] = lr
        best_params['reg'] = reg
        best_params['ws'] = ws
    print('lr: %e reg: %e ws: %e val accuracy: %f' % (
        lr, reg, ws, new_val))

print('best validation accuracy using small dataset: %f' % best_val)

# 拿效果最好的参数训练全部数据
best_model = FullyConnectedNet([256, 128, 64, 32],
                               weight_scale=best_params['ws'],
                               reg=best_params['reg'])

solver = Solver(best_model, data,
                num_epochs=20, batch_size=256,
                update_rule='adam',
                optim_config={
                    'learning_rate': best_params['lr'],
                },
                verbose=False)
solver.train()
print('best validation accuracy using full dataset: %f' % solver.best_val_acc)

```

## 2.batchnormalization

**Inline Question 1:**  Describe the results of this experiment. How does the weight initialization scale affect models with/without batch normalization differently, and why?

**答案：**

实验结果表明，当不使用批归一化时，权重初始化的尺度对模型训练影响很大：若初始权重过大，梯度容易爆炸；若过小，则容易消失，导致训练困难。而使用批归一化后，模型对权重初始化的敏感性显著降低，训练过程更加稳定。
 这是因为批归一化在每一层中标准化了激活值，使其均值为0、方差为1，从而缓解了由权重尺度引起的梯度消失或爆炸问题。

**Inline Question 2:** Describe the results of this experiment. What does this imply about the relationship between batch normalization and batch size? Why is this relationship observed?

**答案：**

实验表明，当批量大小较小时，批归一化的效果变差，模型收敛速度下降甚至训练不稳定。这说明批归一化在很大程度上依赖于足够的批量样本数来估计准确的均值和方差。
 因为批归一化是基于当前批次的统计量进行标准化的，当批次太小时，这些统计量波动较大，归一化的结果不稳定，影响模型的表现

**Inline Question 3:** # Which of these data preprocessing steps is analogous to batch normalization, and which is analogous to layer normalization?

Scaling each image in the dataset, so that the RGB channels for each row of pixels within an image sums up to 1. 

Scaling each image in the dataset, so that the RGB channels for all pixels within an image sums up to 1.   

Subtracting the mean image of the dataset from each image in the dataset. # 4. Setting all RGB values to either 0 or 1 depending on a given threshold.

**答案：**

类比于**批归一化**的是选项 **3**：从每张图像中减去整个数据集的均值图像，这是全局统计量的归一化，与BN中使用的批次统计类似。

类比于**层归一化**的是选项 **2**：对单个样本内部（整张图像）进行归一化，与LN在单个样本维度上归一化类似。

**nline Question 4:** When is layer normalization likely to not work well, and why? #  # 1. Using it in a very deep network # 2. Having a very small dimension of features # 3. Having a high regularization term

**答案：**

选项 2：特征维度很小 会导致层归一化表现不佳。
 因为层归一化是基于每个样本在特征维度上的均值和方差进行归一化的，当特征维度很小（如只有几维）时，统计结果不可靠，导致归一化效果不稳定，甚至会引入噪声，影响模型性能。

### 补充代码：

batchnorm_forward：

```python
def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_mean = np.mean(x, axis=0)  # 计算均值
        x_var = np.var(x, axis=0)  # 计算方差
        x_std = np.sqrt(x_var + eps)  # 计算标准差
        x_norm = (x - x_mean) / x_std  # 归一化
        out = gamma * x_norm + beta  # 计算输出

        cache = (x, x_mean, x_var, x_std, x_norm, out, gamma, beta)  # 保存中间变量

        # 更新running_mean和running_var
        running_mean = momentum * running_mean + (1 - momentum) * x_mean
        running_var = momentum * running_var + (1 - momentum) * x_var

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################

        x_norm = (x - running_mean) / np.sqrt(running_var + eps)  # 归一化
        out = gamma * x_norm + beta  # 计算输出

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache

```

batchnorm_backward:

```python
def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_mean, x_var, x_std, x_norm, out, gamma, beta, eps = cache
    m = x.shape[0]
    dx_hat = dout * gamma
    dvar = np.sum(dx_hat * (x - x_mean) * (-0.5) * np.power((x_var + eps), -1.5), axis=0)
    dmean = np.sum(dx_hat * (-1) / np.sqrt(x_var + eps), axis=0) + dvar * np.sum(-2 * (x - x_mean), axis=0) / m

    dx_1 = dout * gamma
    dx_2_b = np.sum((x - x_mean) * dx_1, axis=0)
    dx_2_a = ((x_var + eps) ** -0.5) * dx_1
    dx_3_b = -0.5 * ((x_var + eps) ** -1.5) * dx_2_b
    dx_4_b = dx_3_b * 1
    dx_5_b = np.ones_like(x) / m * dx_4_b
    dx_6_b = 2 * (x - x_mean) * dx_5_b
    dx_7_a = dx_6_b * 1 + dx_2_a * 1
    dx_7_b = dx_6_b * 1 + dx_2_a * 1
    dx_8_b = -1 * np.sum(dx_7_b, axis=0)
    dx_9_b = np.ones_like(x) / m * dx_8_b
    dx_10 = dx_9_b + dx_7_a

    dx = dx_10
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dx, dgamma, dbeta

```

batchnorm_backward_alt:

```python
def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################

    x, x_mean, x_var, x_std, x_norm, out, gamma, beta, eps = cache
    dgamma = np.sum(dout * x_norm, axis=0)  # 计算dgamma
    dbeta = np.sum(dout, axis=0)  # 计算dbeta

    dx_norm = dout * gamma  # 计算dx_norm
    dx_var = np.sum(dx_norm * (x - x_mean) * (-0.5) * np.power(x_var + eps, -1.5), axis=0)  # 计算dx_var
    dx_mean = np.sum(dx_norm * (-1) / x_std, axis=0) + dx_var * np.sum(-2 * (x - x_mean), axis=0) / x.shape[0]  # 计算dx_mean
    dx = dx_norm / x_std + dx_var * 2 * (x - x_mean) / x.shape[0] + dx_mean / x.shape[0]  # 计算dx


    return dx, dgamma, dbeta

```

fc_net.init:

```python
    def __init__(
            self,
            hidden_dims,
            input_dim=3 * 32 * 32,
            num_classes=10,
            dropout_keep_ratio=1,
            normalization=None,
            reg=0.0,
            weight_scale=1e-2,
            dtype=np.float32,
            seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 获取所有层数的维度
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        # 初始化所有层的参数 (这里的层数是上面的layer_dims的长度减1,因此不会下标越界)
        for i in range(self.num_layers):
            self.params['W' + str(i + 1)] = np.random.normal(0, weight_scale, size=(layer_dims[i], layer_dims[i + 1]))
            self.params['b' + str(i + 1)] = np.zeros(layer_dims[i + 1])
            # 接下来添加batch normalization 层，注意最后一层不需要添加
            if self.normalization == 'batchnorm' and i < self.num_layers - 1:
                self.params['gamma' + str(i + 1)] = np.ones(layer_dims[i + 1])
                self.params['beta' + str(i + 1)] = np.zeros(layer_dims[i + 1])

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

```

layer_utils:

```python
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    affine_out,affine_cache = affine_forward(x, w, b)
    bn_out,bn_cache = batchnorm_forward(affine_out, gamma, beta, bn_param)
    relu_out,relu_cache = relu_forward(bn_out)
    cache = (affine_cache, bn_cache, relu_cache)
    return relu_out, cache

def affine_bn_relu_backward(dout, cache):
    affine_cache, bn_cache, relu_cache = cache
    drelu_out = relu_backward(dout, relu_cache)
    dbn_out, dgamma, dbeta = batchnorm_backward(drelu_out, bn_cache)
    dx, dw, db = affine_backward(dbn_out, affine_cache)
    return dx, dw, db, dgamma, dbeta


```

fc_net.loss:

```python
    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 我们网络的结果是这样的 {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

        # 用一个变量保存上一层的输出
        layer_input = X
        caches = {}
        # 对前面 L - 1层进行操作，因为最后一层的操作和前面的不一样
        for i in range(1, self.num_layers):
            W = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            if self.normalization == 'batchnorm':
                gamma = self.params['gamma' + str(i)]
                beta = self.params['beta' + str(i)]
                layer_input, caches['layer' + str(i)] = affine_bn_relu_forward(layer_input, W, b, gamma, beta, self.bn_params[i - 1])
            else:
                layer_input, caches['layer' + str(i)] = affine_relu_forward(layer_input, W, b)


        # 最后一层的操作
        W = self.params['W' + str(self.num_layers)]
        b = self.params['b' + str(self.num_layers)]

        scores, affine_cache = affine_forward(layer_input, W, b)
        caches['layer' + str(self.num_layers)] = affine_cache

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 计算loss
        loss, dscores = softmax_loss(scores, y)

        # 先计算最后一层的梯度
        dx, dw, db = affine_backward(dscores, caches['layer' + str(self.num_layers)])
        grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db

        for i in range(self.num_layers - 1, 0, -1):
            if self.normalization == 'batchnorm':
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dx, caches['layer' + str(i)])
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta
            else:
                dx, dw, db = affine_relu_backward(dx, caches['layer' + str(i)])
            grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db

        # 加上正则化项
        for i in range(1, self.num_layers + 1):
            W = self.params['W' + str(i)]
            loss += 0.5 * self.reg * np.sum(W * W)


        return loss, grads

```

layernorm_forward:

```python
def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################

 	# 对输入数据求转置
    x = x.T
    gamma, beta = np.atleast_2d(gamma).T, np.atleast_2d(beta).T

    # 直接复用bn代码
    x_mean = np.mean(x, axis=0)
    x_var = np.var(x, axis=0)
    x_std = np.sqrt(x_var + eps)
    x_norm = (x - x_mean) / x_std
    out = gamma * x_norm + beta

    # 转置回来
    out = out.T

    cache = (x, x_mean, x_var, x_std, x_norm, out, gamma, beta, eps)

    return out, cache

```

layernorm_backward:

```python
def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################

    # 对输入数据求转置，直接复用bn代码
    x, x_mean, x_var, x_std, x_norm, out, gamma, beta, eps = cache

    dout = dout.T
    dgamma = np.sum(dout * x_norm, axis=1)  # 计算dgamma
    dbeta = np.sum(dout, axis=1)  # 计算dbeta

    dx_norm = dout * gamma  # 计算dx_norm
    dx_var = np.sum(dx_norm * (x - x_mean) * (-0.5) * np.power(x_var + eps, -1.5), axis=0)  # 计算dx_var
    dx_mean = np.sum(dx_norm * (-1) / x_std, axis=0) + dx_var * np.sum(-2 * (x - x_mean), axis=0) / x.shape[0]  # 计算dx_mean
    dx = dx_norm / x_std + dx_var * 2 * (x - x_mean) / x.shape[0] + dx_mean / x.shape[0]  # 计算dx

    dx = dx.T

    return dx, dgamma, dbeta

```

## 3.Dropout

**Inline Question 1:** What happens if we do not divide the values being passed through inverse dropout by `p` in the dropout layer? Why does that happen?

**答案：**

如果不将通过反向 dropout 的值除以保留概率 `p`，则在训练时，激活值的期望会变小（因为某些神经元被置为0，但其余未放大的激活值没有补偿），从而导致整个网络输出偏小。
 这种不对激活进行缩放的做法会造成训练和测试时的行为不一致，因为测试时不会进行 dropout，而训练时由于未缩放导致激活偏低，模型学习到的参数可能会对测试阶段不适用。
 因此，为了保持训练和测试阶段的激活分布一致，训练时需要将保留下来的激活值除以 `p`，即进行“反向 dropout”的缩放操作。

**Inline Question 2:** Compare the validation and training accuracies with and without dropout -- what do your results suggest about dropout as a regularizer?

**答案：**

不使用 dropout 时，训练准确率更高，但验证准确率可能较低或差异较大，容易出现过拟合。

使用 dropout 后，训练准确率下降（因为引入了噪声，使训练更难），但验证准确率更稳定、通常更高。

这说明 dropout 在训练时通过随机“关闭”部分神经元，防止网络对训练数据过拟合，从而提升模型在未见数据上的泛化能力。因此，dropout 是一种有效的正则化方法，能缓解过拟合问题，尤其适用于参数较多的神经网络。

### 补充代码：

dropout_forward:

```python
def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p  # 生成mask
        out = x * mask  # dropout操作

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################

        out = x  # 测试阶段不做任何操作


    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache

```

dropout_backward:

```python

def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################

        dx = dout * mask

    elif mode == "test":
        dx = dout
    return dx

```

fc_net loss:

```python
def loss(self, X, y=None):
    """Compute loss and gradient for the fully connected net.
    
    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
        scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
        names to gradients of the loss with respect to those parameters.
    """
    X = X.astype(self.dtype)
    mode = "test" if y is None else "train"

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.use_dropout:
        self.dropout_param["mode"] = mode
    if self.normalization == "batchnorm":
        for bn_param in self.bn_params:
            bn_param["mode"] = mode
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################


    # 我们网络的结果是这样的 {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    # 用一个变量保存上一层的输出
    layer_input = X
    caches = {}
    # 对前面 L - 1层进行操作，因为最后一层的操作和前面的不一样
    for i in range(1, self.num_layers):
        W = self.params['W' + str(i)]
        b = self.params['b' + str(i)]
        if self.normalization == 'batchnorm':
            gamma = self.params['gamma' + str(i)]
            beta = self.params['beta' + str(i)]
            layer_input, caches['layer' + str(i)] = affine_bn_relu_forward(layer_input, W, b, gamma, beta, self.bn_params[i - 1])
        else:
            layer_input, caches['layer' + str(i)] = affine_relu_forward(layer_input, W, b)
        if self.use_dropout:  # 如果使用dropout
            layer_input, caches['dropout' + str(i)] = dropout_forward(layer_input, self.dropout_param)

    # 最后一层的操作
    W = self.params['W' + str(self.num_layers)]
    b = self.params['b' + str(self.num_layers)]

    scores, affine_cache = affine_forward(layer_input, W, b)
    caches['layer' + str(self.num_layers)] = affine_cache


    # If test mode return early.
    if mode == "test":
        return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch/layer normalization, you don't need to regularize the   #
    # scale and shift parameters.                                              #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################


    # 计算loss
    loss, dscores = softmax_loss(scores, y)

    # 先计算最后一层的梯度
    dx, dw, db = affine_backward(dscores, caches['layer' + str(self.num_layers)])
    grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
    grads['b' + str(self.num_layers)] = db

    for i in range(self.num_layers - 1, 0, -1):
        if self.use_dropout:  # dropout层的梯度
            dx = dropout_backward(dx, caches['dropout' + str(i)])
        if self.normalization == 'batchnorm':
            dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dx, caches['layer' + str(i)])
            grads['gamma' + str(i)] = dgamma
            grads['beta' + str(i)] = dbeta
        else:
            dx, dw, db = affine_relu_backward(dx, caches['layer' + str(i)])
        grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
        grads['b' + str(i)] = db

    # 加上正则化项
    for i in range(1, self.num_layers + 1):
        W = self.params['W' + str(i)]
        loss += 0.5 * self.reg * np.sum(W * W)


    return loss, grads
```

## 4.Convolutional Neural Networks



### 补充代码：

conv_forward_naive：

```python
def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    # 先获取一些需要用到的数据
    N, C, H_input, W_input = x.shape  # N个样本，C个通道，H_input高，W_input宽
    F, C_w_, HH, WW = w.shape  # F个卷积核, C_w_个通道，HH高，WW宽
    stride = conv_param["stride"]  # 步长
    pad = conv_param["pad"]  # 填充数量

    # 计算卷积后的高和宽
    out_H = int(1 + (H_input + 2 * pad - HH) / stride)
    out_W = int(1 + (W_input + 2 * pad - WW) / stride)

    # 给x的上下左右填充上pad个0
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant", constant_values=0)
    # 将卷积核w转换成F * (C * HH * WW)的矩阵 (便于使用矩阵乘法)
    w_row = w.reshape(F, -1)
    # 生成空白输出便于后续循环填充
    out = np.zeros((N, F, out_H, out_W))

    # 开始卷积
    for n in range(N):  # 遍历样本
        for f in range(F):  # 遍历卷积核
            for i in range(out_H):  # 遍历高
                for j in range(out_W):  # 遍历宽
                    # 获取当前卷积窗口
                    window = x_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                    # 将卷积窗口拉成一行
                    window_row = window.reshape(1, -1)
                    # 计算当前卷积窗口和卷积核的卷积结果
                    out[n, f, i, j] = np.sum(window_row * w_row[f, :]) + b[f]
      
	 # 将pad后的x存入cache (省的反向传播的时候在计算一次)
    x = x_pad

    cache = (x, w, b, conv_param)
    return out, cache

```

conv_backward_naive:

```python
def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################

    # 获取一些需要用到的数据
    x, w, b, conv_param = cache
    N, C, H_input, W_input = x.shape  # N个样本，C个通道，H_input高，W_input宽
    F, C_w_, HH, WW = w.shape  # F个卷积核, C_w_个通道，HH高，WW宽
    stride = conv_param["stride"]  # 步长
    pad = conv_param["pad"]  # 填充数量

    # 计算卷积后的高和宽
    out_H = int(1 + (H_input - HH) / stride)
    out_W = int(1 + (W_input - WW) / stride)

    # 给dx,dw,db分配空间
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            for i in range(out_H):
                for j in range(out_W):
                    # 获取当前卷积窗口
                    window = x[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                    # 计算db
                    db[f] += dout[n, f, i, j]
                    # 计算dw
                    dw[f] += window * dout[n, f, i, j]
                    # 计算dx
                    dx[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += w[f] * dout[n, f, i, j]

    # 去掉dx的pad
    dx = dx[:, :, pad:H_input - pad, pad:W_input - pad]

    return dx, dw, db

```

max_pool_forward_naive:

```python
def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################

    # 获取一些需要用到的数据
    N, C, H, W = x.shape  # N个样本，C个通道，H高，W宽
    pool_height = pool_param["pool_height"]  # 池化核高
    pool_width = pool_param["pool_width"]  # 池化核宽
    stride = pool_param["stride"]  # 步长

    # 计算池化后的高和宽
    out_H = int(1 + (H - pool_height) / stride)
    out_W = int(1 + (W - pool_width) / stride)

    # 给out分配空间
    out = np.zeros((N, C, out_H, out_W))

    for n in range(N):
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    # 获取当前池化窗口
                    window = x[n, c, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width]
                    # 计算当前池化窗口的最大值
                    out[n, c, i, j] = np.max(window)


    cache = (x, pool_param)
    return out, cache

```

max_pool_backward_naive:

```python
def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################

    # 获取一些需要用到的数据
    x, pool_param = cache
    N, C, H, W = x.shape  # N个样本，C个通道，H高，W宽
    pool_height = pool_param["pool_height"]  # 池化核高
    pool_width = pool_param["pool_width"]  # 池化核宽
    stride = pool_param["stride"]  # 步长

    # 计算池化后的高和宽
    out_H = int(1 + (H - pool_height) / stride)
    out_W = int(1 + (W - pool_width) / stride)

    # 给dx分配空间
    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    # 获取当前池化窗口
                    window = x[n, c, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width]
                    # 计算当前池化窗口的最大值
                    max_index = np.argmax(window)
                    # 计算dx
                    dx[n, c, i * stride + max_index // pool_width, j * stride + max_index % pool_width] += dout[n, c, i, j]

    return dx

```

ThreeLayerConvNet:

init:

```python
    def __init__(
            self,
            input_dim=(3, 32, 32),
            num_filters=32,
            filter_size=7,
            hidden_dim=100,
            num_classes=10,
            weight_scale=1e-3,
            reg=0.0,
            dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        C, H, W = input_dim  # 获取输入数据的通道数，高度，宽度

        # 卷积层
        self.params["W1"] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params["b1"] = np.zeros(num_filters)

        # 全连接层
        self.params["W2"] = np.random.normal(0, weight_scale, (num_filters * H * W // 4, hidden_dim))
        self.params["b2"] = np.zeros(hidden_dim)

        # 全连接层
        self.params["W3"] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params["b3"] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

```

loss:

```python
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)  # 卷积层
        out2, cache2 = affine_relu_forward(out1, W2, b2)  # 全连接层
        scores, cache3 = affine_forward(out2, W3, b3)  # 全连接层

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 计算损失
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))  # L2正则化

        # 计算梯度
        dout, grads["W3"], grads["b3"] = affine_backward(dout, cache3)  # 全连接层
        dout, grads["W2"], grads["b2"] = affine_relu_backward(dout, cache2)  # 全连接层
        dout, grads["W1"], grads["b1"] = conv_relu_pool_backward(dout, cache1)  # 卷积层

        # 加上正则化项的梯度
        grads["W3"] += self.reg * W3
        grads["W2"] += self.reg * W2
        grads["W1"] += self.reg * W1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

```

spatial_batchnorm_forward:

```python
def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape  # N个样本，C个通道，H高，W宽
    x = np.moveaxis(x, 1, -1).reshape(-1, C)  # 将C通道放到最后，然后reshape成二维数组
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)  # 调用batchnorm_forward
    out = np.moveaxis(out.reshape(N, H, W, C), -1, 1)  # 将C通道放到第二维，然后reshape成四维数组

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache

```

spatial_batchnorm_backward:

```python
def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape  # N个样本，C个通道，H高，W宽
    dout = np.moveaxis(dout, 1, -1).reshape(-1, C)  # 将C通道放到最后，然后reshape成二维数组
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)  # 调用batchnorm_backward
    dx = np.moveaxis(dx.reshape(N, H, W, C), -1, 1)  # 将C通道放到第二维，然后reshape成四维数组

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

```

spatial_groupnorm_forward:

```python
def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape  # N个样本，C个通道，H高，W宽

    # 将C通道分成G组，每组有C//G个通道
    x = x.reshape(N, G, C // G, H, W)  # reshape成五维数组
    x_mean = np.mean(x, axis=(2, 3, 4), keepdims=True)  # 求均值
    x_var = np.var(x, axis=(2, 3, 4), keepdims=True)  # 求方差
    x_norm = (x - x_mean) / np.sqrt(x_var + eps)  # 归一化

    x_norm = x_norm.reshape(N, C, H, W)  # reshape成四维数组
    out = gamma * x_norm + beta  # 伸缩平移

    cache = (x, x_norm, x_mean, x_var, gamma, beta, G, eps)  # 缓存变量

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache

```

spatial_groupnorm_backward:

```python
def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_norm, x_mean, x_var, gamma, beta, G, eps = cache  # 从缓存中取出变量
    N, C, H, W = dout.shape  # N个样本，C个通道，H高，W宽

    # 计算dgamma和dbeta
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)  # 求dgamma
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)  # 求dbeta

    # 准备数据
    x = x.reshape(N, G, C // G, H, W)  # reshape成五维数组

    m = C // G * H * W
    dx_norm = (dout * gamma).reshape(N, G, C // G, H, W)
    dx_var = np.sum(dx_norm * (x - x_mean) * (-0.5) * np.power((x_var + eps), -1.5), axis=(2, 3, 4), keepdims=True)
    dx_mean = np.sum(dx_norm * (-1) / np.sqrt(x_var + eps), axis=(2, 3, 4), keepdims=True) + dx_var * np.sum(-2 * (x - x_mean), axis=(2, 3, 4),
                                                                                                             keepdims=True) / m
    dx = dx_norm / np.sqrt(x_var + eps) + dx_var * 2 * (x - x_mean) / m + dx_mean / m
    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta

```



## 5.PyTorch on CIFAR-10



### 补充代码：

three_layer_convnet：

```python
def three_layer_convnet(x, params):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?
    
    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    ################################################################################
    # TODO: Implement the forward pass for the three-layer ConvNet.                #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x = F.conv2d(x, conv_w1, bias=conv_b1, padding=2)
    x = F.relu(x)
    x = F.conv2d(x, conv_w2, bias=conv_b2, padding=1)
    x = F.relu(x)
    x = flatten(x)
    scores = x.mm(fc_w) + fc_b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################
    return scores

```

Training a ConvNet:

```python
learning_rate = 3e-3

channel_1 = 32
channel_2 = 16

conv_w1 = None
conv_b1 = None
conv_w2 = None
conv_b2 = None
fc_w = None
fc_b = None

################################################################################
# TODO: Initialize the parameters of a three-layer ConvNet.                    #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

conv_w1 = random_weight((channel_1, 3, 5, 5))
conv_b1 = zero_weight(channel_1)
conv_w2 = random_weight((channel_2, channel_1, 3, 3))
conv_b2 = zero_weight(channel_2)
fc_w = random_weight((channel_2 * 32 * 32, 10))
fc_b = zero_weight(10)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
train_part2(three_layer_convnet, params, learning_rate)

```

ThreeLayerConvNet:

```python
class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        ########################################################################
        # TODO: Set up the layers you need for a three-layer ConvNet with the  #
        # architecture defined above.                                          #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1)
        self.fc3 = nn.Linear(channel_2 * 32 * 32, num_classes)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def forward(self, x):
        scores = None
        ########################################################################
        # TODO: Implement the forward function for a 3-layer ConvNet. you      #
        # should use the layers you defined in __init__ and specify the        #
        # connectivity of those layers in forward()                            #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        scores = self.fc3(flatten(x))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return scores

```

Train a Three-Layer ConvNet:

```python
learning_rate = 3e-3
channel_1 = 32
channel_2 = 16

model = None
optimizer = None
################################################################################
# TODO: Instantiate your ThreeLayerConvNet model and a corresponding optimizer #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

model = ThreeLayerConvNet(in_channel=3, channel_1=channel_1, channel_2=channel_2, num_classes=10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

train_part34(model, optimizer)

```

Sequential API: Three-Layer ConvNet:

```python
channel_1 = 32
channel_2 = 16
learning_rate = 1e-2

model = None
optimizer = None

################################################################################
# TODO: Rewrite the 2-layer ConvNet with bias from Part III with the           #
# Sequential API.                                                              #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=channel_1, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=channel_1, out_channels=channel_2, kernel_size=3, padding=1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(channel_2 * 32 * 32, 10)
)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

train_part34(model, optimizer)

```

CIFAR-10 open-ended challenge:

```python
################################################################################
# TODO:                                                                        #         
# Experiment with any architectures, optimizers, and hyperparameters.          #
# Achieve AT LEAST 70% accuracy on the *validation set* within 10 epochs.      #
#                                                                              #
# Note that you can use the check_accuracy function to evaluate on either      #
# the test set or the validation set, by passing either loader_test or         #
# loader_val as the second argument to check_accuracy. You should not touch    #
# the test set until you have finished your architecture and  hyperparameter   #
# tuning, and only run the test set once at the end to report a final value.   #
################################################################################
model = None
optimizer = None

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    nn.Linear(128 * 4 * 4, 1024),
)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# You should get at least 70% accuracy.
# You may modify the number of epochs to any number below 15.
train_part34(model, optimizer, epochs=10)

```

