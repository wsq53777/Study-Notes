## 1. k-Nearest Neighbor (kNN) 练习

**Inline Question 1:** 注意距离矩阵中的结构化模式，其中一些行或列更亮。(注意，在默认的配色方案中，黑色表示低距离，白色表示高距离。)

- 数据中那些明显明亮的行背后的原因是什么?
- 是什么导致了这些列?

答案：

1. 该测试数据与所有的训练数据的距离大，说明该测试数据是异常值
2. 该训练数据与所有测试数据的距离大，说明该训练数据是异常值，坏点

 **Inline Question 2**：Which of the following preprocessing steps will **not** change the performance of a kNN classifier using L1 distance?

选项：

1、2、3正确

解释：

- **1**：全局减去一个常数对 L1 距离无影响，因为相对距离不变。
- **2**：每个像素都减去一个值，对L1距离图影响。
- **3**：相当于对每个值进行同样的缩放

**Inline Question 3**：Which statements are true for all `k`?

答案：**2 和 4**

1. 决策边界是非线性的，因为它依赖于训练数据的分布。
2. 1-NN 训练误差为 0（因为自己最近的就是自己），一定小于或等于更大的 k。
3. 不能保证，1-NN 更容易过拟合，可能导致测试误差更大。
4.  kNN 测试时必须计算到所有训练样本，时间随着训练集增大而增加。



### 补充代码：

KNN最近邻分类器实现：

```python

def compute_distances_two_loops(self, X):
"""
Compute the distance between each test point in X and each training point
in self.X_train using a nested loop over both the training data and the
test data.

Inputs:
- X: A numpy array of shape (num_test, D) containing test data.

Returns:
- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
  is the Euclidean distance between the ith test point and the jth training
  point.
"""
num_test = X.shape[0]
num_train = self.X_train.shape[0]
dists = np.zeros((num_test, num_train))
for i in range(num_test):
    for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension, nor use np.linalg.norm().          #
        #####################################################################

        dists[i,j] = np.sqrt(np.sum(np.power(X[i] - self.X_train[j],2)))

return dists

def compute_distances_one_loop(self, X):
"""
Compute the distance between each test point in X and each training point
in self.X_train using a single loop over the test data.

Input / Output: Same as compute_distances_two_loops
"""
num_test = X.shape[0]
num_train = self.X_train.shape[0]
dists = np.zeros((num_test, num_train))
for i in range(num_test):
    #######################################################################
    # TODO:                                                               #
    # Compute the l2 distance between the ith test point and all training #
    # points, and store the result in dists[i, :].                        #
    # Do not use np.linalg.norm().                                        #
    #######################################################################
    dists[i,:] = np.sqrt(np.sum(np.power(self.X_train - X[i],2),axis=1))
return dists

def compute_distances_no_loops(self, X):
"""
Compute the distance between each test point in X and each training point
in self.X_train using no explicit loops.

Input / Output: Same as compute_distances_two_loops
"""
num_test = X.shape[0]
num_train = self.X_train.shape[0]
dists = np.zeros((num_test, num_train))
#########################################################################
# TODO:                                                                 #
# Compute the l2 distance between all test points and all training      #
# points without using any explicit loops, and store the result in      #
# dists.                                                                #
#                                                                       #
# You should implement this function using only basic array operations; #
# in particular you should not use functions from scipy,                #
# nor use np.linalg.norm().                                             #
#                                                                       #
# HINT: Try to formulate the l2 distance using matrix multiplication    #
#       and two broadcast sums.                                         #
#########################################################################

tmp1 = np.sum(np.power(X,2),axis=1).reshape((X.shape[0],1))
tmp2 = np.sum(np.power(self.X_train,2),axis = 1).reshape((self.X_train.shape[0],1)).T
dists = np.sqrt(-2 * (X @ self.X_train.T) + tmp1 + tmp2)

return dists

def predict_labels(self, dists, k=1):
"""
Given a matrix of distances between test points and training points,
predict a label for each test point.

Inputs:
- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
  gives the distance betwen the ith test point and the jth training point.

Returns:
- y: A numpy array of shape (num_test,) containing predicted labels for the
  test data, where y[i] is the predicted label for the test point X[i].
"""
num_test = dists.shape[0]
y_pred = np.zeros(num_test)
for i in range(num_test):
    # A list of length k storing the labels of the k nearest neighbors to
    # the ith test point.
    closest_y = []
    #########################################################################
    # TODO:                                                                 #
    # Use the distance matrix to find the k nearest neighbors of the ith    #
    # testing point, and use self.y_train to find the labels of these       #
    # neighbors. Store these labels in closest_y.                           #
    # Hint: Look up the function numpy.argsort.                             #
    #########################################################################

    closest_y = self.y_train[np.argsort(dists[i])[:k]]

    #########################################################################
    # TODO:                                                                 #
    # Now that you have found the labels of the k nearest neighbors, you    #
    # need to find the most common label in the list closest_y of labels.   #
    # Store this label in y_pred[i]. Break ties by choosing the smaller     #
    # label.                                                                #
    #########################################################################

    y_pred[i] = np.argmax(np.bincount(closest_y))


return y_pred
```

Cross-validation交叉验证代码实现：

```python
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################

X_train_folds = np.array_split(X_train,num_folds)
y_train_folds = np.array_split(y_train,num_folds)
# print(X_train_folds)


# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################

pass
for k in k_choices:
  k_to_accuracies[k] = []
  for i in range(num_folds):
    #选择第i个分片为验证集，其他数据为训练数据
    temp_train_x = np.concatenate(np.compress([False if temp_i == i else True for temp_i in range(num_folds)],X_train_folds,axis=0))
    temp_train_y = np.concatenate(np.compress([False if temp_i == i else True for temp_i in range(num_folds)],y_train_folds,axis=0))

    # 训练数据
    classifier.train(temp_train_x,temp_train_y)

    # 获取预测
    temp_pred_y = classifier.predict(X_train_folds[i],k=k,num_loops=0)

    # 计算准确率
    correct_count = np.sum(temp_pred_y == y_train_folds[i])
    k_to_accuracies[k].append(correct_count / len(temp_pred_y))


# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
     
```



## 2.Training a Support Vector Machine SVM算法

**Inline Question 1**
 有时候梯度检查会在某些维度上不完全匹配。这种不一致可能是由什么原因造成的？这是否值得担忧？请举一个一维中梯度检查可能失败的简单例子。margin 的变化如何影响这种情况发生的频率？提示：SVM 损失函数从严格意义上讲是不可导的。

**回答：**
 这种不一致的原因通常是由于 SVM 损失函数在某些点上**不可导**，特别是在 margin 等于零时。在这种情况下，梯度的数值近似可能会受到扰动，因为有限差分法在不可导点附近不稳定。

例子：考虑函数 `f(x) = |x|`，它在 `x=0` 处不可导，左右两侧的导数分别是 `-1` 和 `1`。如果你在 `x=0` 处做梯度检查，数值梯度会因为有限差分的方向不同而产生不一致。

这种不一致不是严重问题，只要在大部分维度上，数值梯度和解析梯度是一致的，就可以认为你的实现是正确的。

margin 的值影响了这种情况发生的频率：

- 如果 margin 设置较大，那么更多的点可能落在不可导区域（因为被视为有“损失”），从而增加不一致的可能性；
- 如果 margin 较小，则更多点的梯度是 0，不太可能处于临界点，不一致的概率就小一些。

**Inline Question 2**
 描述你可视化出的 SVM 权重长什么样，并简要解释为什么会是这种样子。

**回答：**
 每一类的 SVM 权重可以被看作是一个图像，反映了模型学到的每个像素对该类别的判别性。在可视化结果中，权重图像通常呈现出每个类别的“原型图像”。

例如：对于“飞机”类别，图像可能背景较多是蓝色或灰色（天空），中间有水平结构（飞机机身）

### 补充代码：

计算梯度：

```python 
def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # 正确分类的梯度减上X[i]
                dW[:,y[i]] -= X[i].T
                # 错误分类的梯度加去X[i]
                dW[:,j] += X[i].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    # 梯度同样处理
    dW /= num_train
    # 正则项的梯度
    dW += 2 * reg * W


    return loss, dW

```

向量法计算Loss：

```python 
def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X @ W
    # 获取对于每个x而言正确分类的分数
    scores_correct = scores[range(num_train),y].reshape((scores.shape[0],1))
    # 对每个元素做max(0,scores_error - scores_correct + 1)操作，包括正确分类的元素
    # 统一操作后减少代码编写难度，只需要最后处理一下正确分类的分数，把他们变成0就行了
    margins = np.maximum(0,scores - scores_correct + 1)
    # 将正确分类的margins置为0
    margins[range(num_train),y] = 0
    loss += np.sum(margins) / num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################


    return loss, dW


```

LinearClassifier.train:

```python
def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################

            choice_idxs = np.random.choice(num_train,batch_size)
            X_batch = X[choice_idxs]
            y_batch = y[choice_idxs]


            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################


            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

```

预测：

```python
def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################


        scores = X @ self.W
        y_pred = np.argmax(scores,axis= 1)


        return y_pred


```

找到最好的hypeparameter:

```python
# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of about 0.39 (> 0.385) on the validation set.

# Note: you may see runtime/overflow warnings during hyper-parameter search.
# This may be caused by extreme values, and is not a bug.

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.

################################################################################
# TODO:                                                                        #
# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a linear SVM on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the LinearSVM object that achieves this  #
# accuracy in best_svm.                                                        #
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the SVMs don't take much time to train; once you are #
# confident that your validation code works, you should rerun the validation   #
# code with a larger value for num_iters.                                      #
################################################################################

# Provided as a reference. You may or may not want to change these hyperparameters
learning_rates = [2e-7, 0.75e-7,1.5e-7, 1.25e-7, 0.75e-7]
regularization_strengths = [3e4, 3.25e4, 3.5e4, 3.75e4, 4e4,4.25e4, 4.5e4,4.75e4, 5e4]

for lr in learning_rates:
  for reg in regularization_strengths:
    svm = LinearSVM()
    svm.train(X_train, y_train, learning_rate=lr, reg=reg,num_iters=1500, verbose=False)
    y_train_pred = svm.predict(X_train)
    y_train_accuracy = np.mean(y_train == y_train_pred)
    y_val_pred = svm.predict(X_val)
    y_val_accuracy = np.mean(y_val == y_val_pred)
    results[(lr,reg)] = (y_train_accuracy,y_val_accuracy)
    if(y_val_accuracy > best_val):
      best_val = y_val_accuracy
      best_svm = svm



# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

```



## 3.softmax

**Inline Question 1**：Why do we expect our loss to be close to -log(0.1)? Explain briefly.

**答案：**

因为softmax分类器的输出本质上是对每一类的概率预测。在权重初始化随机很小时，所有类别的得分基本一样，softmax后每个类的概率约为1/10。假设我们的标签是均匀分布的，那么交叉熵损失就是-log(1/10)。所以我们期望初始loss接近 -log(0.1)

**Inline Question 2**：Suppose the overall training loss is defined as the sum of the per-datapoint loss over all training examples. It is possible to add a new datapoint to a training set that would leave the SVM loss unchanged, but this is not the case with the Softmax classifier loss.True or False? 

**答案：**

True
SVM的损失（Hinge Loss）对于某些数据点，如果它已经被正确分类并且间隔大于1，则该数据点对总损失的贡献为0。如果我们向训练集中加入一个已经被正确大间隔分类的新样本，SVM总损失可能不会变。

但Softmax的损失无论样本多么容易分类，每一个样本都会有一定的损失（只要概率不是1，loss总是大于0），即使其概率接近1，损失也只是趋近于0。因此，任意加入一个新样本，其损失总会对总loss有（多少）贡献，所以loss一定会被改变。

### 补充代码：

softmax_loss_naive：

```python
def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################


    # 训练集的数量
    num_train = X.shape[0]
    # 分类的数量
    num_classes = W.shape[1]
    for i in range(num_train):
      scores = X[i] @ W
      # 对其求e的幂函数
      scores = np.exp(scores)
      # 求对于每一个分类的概率
      p = scores / np.sum(scores)
      # 求loss函数
      loss += -np.log(p[y[i]])

      # 求梯度
      for k in range(num_classes):
        # 获取当前分类的概率
        p_k = p[k]
        # 判断当前分类是否是正确分类
        if k == y[i]:
          dW[:,k] += (p_k - 1) * X[i] 
        else:
          dW[:,k] += (p_k) * X[i]


    # 处理正则项
    loss /= num_train
    dW /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    

    return loss, dW

```

softmax_loss_vectorized:

```python
def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # 训练集的数量
    num_train = X.shape[0]
    # 分类的数量
    num_classes = W.shape[1]

    # 先计算得分
    scores = X @ W
    # 再取e的幂函数
    scores = np.exp(scores)
    # 计算所有的概率
    p = scores / np.sum(scores,axis = 1,keepdims = True)
    # 计算loss函数
    loss += np.sum(-np.log(p[range(num_train),y]))

    # 计算梯度 根据上面的公式可以知道只要给正确分类的P - 1就可以得到dW
    p[range(num_train),y] -= 1
    dW = X.T @ p


    # 计算正则项
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

    return loss, dW

```

找到最好的hypeparameter:

```python
# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set.

from cs231n.classifiers import Softmax
results = {}
best_val = -1
best_softmax = None

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained softmax classifer in best_softmax.                          #
################################################################################

# Provided as a reference. You may or may not want to change these hyperparameters
learning_rates = [2e-7, 0.75e-7,1.5e-7, 1.25e-7, 0.75e-7]
regularization_strengths = [3e4, 3.25e4, 3.5e4, 3.75e4, 4e4,4.25e4, 4.5e4,4.75e4, 5e4]

for lr in learning_rates:
  for reg in regularization_strengths:
    softmax = Softmax()
    softmax.train(X_train, y_train, learning_rate=lr, reg=reg,num_iters=1500, verbose=False)
    y_train_pred = softmax.predict(X_train)
    y_train_accuracy = np.mean(y_train == y_train_pred)
    y_val_pred = softmax.predict(X_val)
    y_val_accuracy = np.mean(y_val == y_val_pred)
    results[(lr,reg)] = (y_train_accuracy,y_val_accuracy)
    if(y_val_accuracy > best_val):
      best_val = y_val_accuracy
      best_softmax = softmax


# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

```

## 4.two_layer_net

**Inline Question 1:**
常见的激活函数在反向传播时容易出现梯度为零（或者极小）的现象，这会导致训练停滞。请说明下面哪些激活函数存在这个问题，并说明在一维情况下，什么样的输入会导致该问题。

- Sigmoid
- ReLU
- Leaky ReLU

**回答：**

1. **Sigmoid**： 
   会出现梯度消失问题。对于输入绝对值很大的情况（特别是远离0时，例如x >> 0 或 x << 0），sigmoid 的输出接近0或1，导数接近0，导致梯度几乎为零，反向传播时几乎不会有有效梯度传回去。

2. **ReLU**： 
   也会出现梯度为零的情况。当输入小于等于0时，ReLU的输出为0，导数也为0，反向传播时不会有梯度流动（即“神经元死亡”现象）。

3. **Leaky ReLU**： 
   基本不会出现梯度完全为零的问题。即使输入小于0，输出仍然为一个很小的负斜率（如0.01x），梯度为常数（如0.01），不会完全阻断梯度流。所以Leaky ReLU能够减少“神经元死亡”的现象。

**Inline Question 2**： 
训练好的神经网络经常会出现测试准确率远低于训练准确率的现象。我们应该如何减小这种差距？请选择所有适用的方式，并进行解释。

- 1. 使用更大的数据集进行训练
- 2. 添加更多隐藏单元
- 3. 提高正则化强度
- 4. 以上都不对

**选择：** 
1 和 3

**解释：**  
- **选项1**：使用更大的数据集。更多的数据可以帮助模型更好地泛化，减小过拟合，从而缩小训练集与测试集之间的准确率差距。
- **选项2**：添加更多隐藏单元。这样会增大模型容量，容易导致过拟合，训练准确率变高但测试准确率可能变低，加大两者差距。因此不适用于减小gap。
- **选项3**：增加正则化强度。正则化（比如L2正则、dropout等）可以有效抑制过拟合，使模型在测试集上的表现更好，从而缩小差距。

### 补充代码：

affine_forward：

```python
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################

    out = x.reshape((x.shape[0],-1)) @ w + b

    cache = (x, w, b)
    return out, cache

```

affine_backward:

```python
def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################


    dx = (dout @ w.T).reshape(x.shape)
    dw = x.reshape((x.shape[0],-1)).T @ dout
    db = np.sum(dout,axis = 0)


    return dx, dw, db


```

relu_forward:

```python
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################

    out = np.maximum(0,x)

 ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


```

relu_backward:

```python
def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################


    dx = np.multiply(dout, (x > 0)) 

 ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

```

svm_loss:

```python
def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = 0, 0

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################

    # 样本数量
    num_train = x.shape[0]

    # 获取正确分类的分数
    scores_correct = x[range(num_train),y].reshape((x.shape[0],1))
    margins = np.maximum(0,x - scores_correct + 1)
    # 将正确分类的分数置为0
    margins[range(num_train),y] = 0
    loss += np.sum(margins)
    # 正则项
    loss /= num_train

    # 计算梯度
    margins[margins > 0] = 1
    row_sum = np.sum(margins,axis = 1)
    margins[range(num_train),y] = -row_sum
    dx = margins
    dx /= num_train
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx

```

softmax_loss:

```python
def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = 0, 0

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################

    # 训练数据的数量
    num_train = x.shape[0]

    # 先取e的幂函数
    scores = np.exp(x- x.max(axis=1, keepdims=True))
    # 上面有种增加数值稳定的函数，防止指数函数太大爆炸影响真正的效果
    # scores = np.exp(x- x.max(axis=1, keepdims=True)) 其中的-x.max（axis=1,keepdims=True）就是增加数值稳定的
    # 当然对于本题的数据，我们直接用np.exp(x)也可以的 ^_^
    # 计算所有的概率
    p = scores / np.sum(scores,axis = 1,keepdims = True)
    # 计算loss函数
    loss += np.sum(-np.log(p[range(num_train),y]))

    # 计算梯度 根据公式可以知道只要给正确分类的P - 1就可以得到dW
    p[range(num_train),y] -= 1
    dx += p


    # 计算正则项
    loss /= num_train
    dx /= num_train
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx

```

fc_net:

```python
    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################


        self.params.W1 = np.random.normal(loc=0.0,scale=weight_scale,size=(input_dim,hidden_dim))
        self.params.W2 = np.random.normal(loc=0.0,scale=weight_scale,size=(hidden_dim,num_classes))
        self.params.b1 = np.zeros((hidden_dim,))
        self.params.b2 = np.zeros((num_classes,))
```

loss：

```python
def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

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
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']


        affine_1_out,affine_1_cache = affine_forward(X,W1,b1)
        relu_out,relu_cache = relu_forward(affine_1_out)
        affine_2_out,affine_2_cache = affine_forward(relu_out,W2,b2)
        # 这里不走softmax层了，因为softmax层是计算loss值了，而这里我们只需要scores
        scores = affine_2_out


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        # 用softmax计算loss和grad
        loss,d_affine_2_out = softmax_loss(affine_2_out,y)
        # loss 只需要算到这里，接下来加上正则项
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2)
)
        # 接下来算梯度
        d_relu_out,dW_2,dB_2 = affine_backward(d_affine_2_out,affine_2_cache)
        d_affine_1_out = relu_backward(d_relu_out,relu_cache)
        dX,dW_1,dB_1 = affine_backward(d_affine_1_out,affine_1_cache)

        dW_1 += self.reg * W1
        dW_2 += self.reg * W2

        # 保存梯度
        grads['W1'] = dW_1
        grads['W2'] = dW_2
        grads['b1'] = dB_1
        grads['b2'] = dB_2


        return loss, grads

```

Solver:

```python
input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
model = TwoLayerNet(input_size, hidden_size, num_classes)
solver = None

##############################################################################
# TODO: Use a Solver instance to train a TwoLayerNet that achieves about 36% #
# accuracy on the validation set.                                            #
##############################################################################


solver = Solver(model, data, optim_config={'learning_rate': 1e-3})
solver.train()


```

Tune your hyperparameters:

```python
best_model = None


#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_model.                                                          #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on thexs previous exercises.                          #
#################################################################################
results = {}
best_val = -1

learning_rates = np.geomspace(3e-4, 3e-2, 3)
regularization_strengths = np.geomspace(1e-6, 1e-2, 5)

import itertools

for lr, reg in itertools.product(learning_rates, regularization_strengths):
    # Create Two Layer Net and train it with Solver
    model = TwoLayerNet(hidden_dim=128, reg=reg)
    solver = Solver(model, data, optim_config={'learning_rate': lr}, num_epochs=10, verbose=False)
    solver.train()
    
    # Compute validation set accuracy and append to the dictionary
    results[(lr, reg)] = solver.best_val_acc

    # Save if validation accuracy is the best
    if results[(lr, reg)] > best_val:
        best_val = results[(lr, reg)]
        best_model = model

# Print out results.
for lr, reg in sorted(results):
    val_accuracy = results[(lr, reg)]
    print('lr %e reg %e val accuracy: %f' % (lr, reg, val_accuracy))

```



## 5. Higher Level Representations: Image Features

**Inline question 1:**  Describe the misclassification results that you see. Do they make sense?

**回答：**

从错误分类结果的可视化中可以看到，很多被错误分类的图像本身就具有一定的模糊性。例如，一些被错误识别为“飞机”的图片其实是具有天空背景的“鸟”，或者是颜色、形状和“飞机”类别相似的其他物体。这反映出特征提取方法（如HOG和色彩直方图）只能提取部分关于图像纹理和颜色的信息，却难以捕捉更高层次的语义特征。

类似地，被误分类为“汽车”的图片，可能确实包含了和汽车颜色、轮廓类似的物体。这说明分类器主要依赖于这些低层次特征进行判别，而对于复杂场景下的物体，容易出现混淆。

这些结果是合理的，因为机器只能依赖输入的特征，特征选择有限时就会造成某些类别之间不容易被区分，尤其是本身在视觉和色彩上就容易混淆的类别。

### 补充代码：

Train SVM on features:

```python
# Use the validation set to tune the learning rate and regularization strength

from cs231n.classifiers.linear_classifier import LinearSVM

learning_rates = [1e-9, 1e-8, 1e-7, 1e-4, 1e-3, 1e-2]
regularization_strengths = [1e-2, 1, 3]

results = {}
best_val = -1
best_svm = None

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained classifer in best_svm. You might also want to play          #
# with different numbers of bins in the color histogram. If you are careful    #
# you should be able to get accuracy of near 0.44 on the validation set.       #
################################################################################


import itertools

for lr, reg in itertools.product(learning_rates, regularization_strengths):
    # Create SVM and train it 
    svm = LinearSVM()
    svm.train(X_train_feats, y_train, lr, reg, num_iters=1500)

    # Compute training and validation sets accuracies and append to the dictionary
    y_train_pred, y_val_pred = svm.predict(X_train_feats), svm.predict(X_val_feats)
    results[(lr, reg)] = np.mean(y_train == y_train_pred), np.mean(y_val == y_val_pred)

    # Save if validation accuracy is the best
    if results[(lr, reg)][1] > best_val:
        best_val = results[(lr, reg)][1]
        best_svm = svm


# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved: %f' % best_val)

```

Neural Network on image features:

```python
from cs231n.classifiers.fc_net import TwoLayerNet
from cs231n.solver import Solver

input_dim = X_train_feats.shape[1]
hidden_dim = 500
num_classes = 10

data = {
    'X_train': X_train_feats,
    'y_train': y_train,
    'X_val': X_val_feats,
    'y_val': y_val,
    'X_test': X_test_feats,
    'y_test': y_test,
}

net = TwoLayerNet(input_dim, hidden_dim, num_classes)
best_net = None

################################################################################
# TODO: Train a two-layer neural network on image features. You may want to    #
# cross-validate various parameters as in previous sections. Store your best   #
# model in the best_net variable.                                              #
################################################################################

learning_rates = np.linspace(1e-2, 2.75e-2, 4)
regularization_strengths = np.geomspace(1e-6, 1e-4, 3)

results = {}
best_val = -1

import itertools

for lr, reg in itertools.product(learning_rates, regularization_strengths):
    # Create Two Layer Net and train it with Solver
    model = TwoLayerNet(input_dim, hidden_dim, num_classes,reg = reg)
    solver = Solver(model, data, optim_config={'learning_rate': lr}, num_epochs=15, verbose=False)
    solver.train()

    # Compute validation set accuracy and append to the dictionary
    results[(lr, reg)] = solver.best_val_acc

    # Save if validation accuracy is the best
    if results[(lr, reg)] > best_val:
        best_val = results[(lr, reg)]
        best_net = model

# Print out results.
for lr, reg in sorted(results):
    val_accuracy = results[(lr, reg)]
    print('lr %e reg %e val accuracy: %f' % (lr, reg, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val)

```

