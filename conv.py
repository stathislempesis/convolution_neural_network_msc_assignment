# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh

import numpy

import logging
from mlp.layers import Layer, max_and_argmax
from numpy.lib.stride_tricks import as_strided as ast
from numpy import newaxis

logger = logging.getLogger()
logger.setLevel(logging.INFO)

'''
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("mlp/cconv.pyx"),
)
'''
"""
You have been given some very initial skeleton below. Feel free to build on top of it and/or
modify it according to your needs. Just notice, you can factor out the convolution code out of
the layer code, and just pass (possibly) different conv implementations for each of the stages
in the model where you are expected to apply the convolutional operator. This will allow you to
keep the layer implementation independent of conv operator implementation, and you can easily
swap it layer, for example, for more efficient implementation if you came up with one, etc.
"""

def my1_conv2d(image, kernels, strides=(1, 1)):
    '''
    Implements a 2d valid convolution of kernels with the image
    Note: filter means the same as kernel and convolution (correlation) of those with the input space
    produces feature maps (sometimes refereed to also as receptive fields). Also note, that
    feature maps are synonyms here to channels, and as such num_inp_channels == num_inp_feat_maps
    :param image: 4D tensor of sizes (batch_size, num_input_channels, img_shape_x, img_shape_y)
    :param filters: 4D tensor of filters of size (num_inp_feat_maps, num_out_feat_maps, kernel_shape_x, kernel_shape_y)
    :param strides: a tuple (stride_x, stride_y), specifying the shift of the kernels in x and y dimensions
    :return: 4D tensor of size (batch_size, num_out_feature_maps, feature_map_shape_x, feature_map_shape_y)
    '''
    
    # http://cs231n.github.io/convolutional-networks/
    
    batch_size, num_input_channels, img_shape_x, img_shape_y = image.shape
    num_inp_feat_maps, num_out_feat_maps, kernel_shape_x, kernel_shape_y = kernels.shape
    stride_x, stride_y = strides
    feature_map_shape_x = (img_shape_x-kernel_shape_x)/stride_x+1
    feature_map_shape_y = (img_shape_y-kernel_shape_y)/stride_y+1
    
    convolution = numpy.zeros((batch_size, num_out_feat_maps, feature_map_shape_x, feature_map_shape_y))
    
    newshape = (batch_size, num_inp_feat_maps, feature_map_shape_x, feature_map_shape_y, kernel_shape_x, kernel_shape_y)
    
    newstrides = image.itemsize*numpy.array([num_inp_feat_maps*img_shape_x*img_shape_y,img_shape_x*img_shape_y,img_shape_y,1,img_shape_y,1])
    
    strided = ast(image,shape = newshape,strides = newstrides)
    
    convolution = numpy.tensordot(strided, kernels, axes=[[1, 4, 5], [0, 2, 3]])
    
    convolution = numpy.transpose(convolution, (0,3,1,2))
    
    '''
    for mx in xrange(0, feature_map_shape_x):
        # the slice on the x dimension
        batch_slice = image[:,:,mx:mx+kernel_shape_x,:]
        for my in xrange(0, feature_map_shape_y):
            # calculates the result of convolution using the einstein summation
            # which using the tensor it calculates the element wise multiplication
            # between the batch slice of the image and the kernels and
            # then it calculates the sum of the elements of the previous multiplication
            mult_sum = numpy.einsum('lkni,kjni->lj', batch_slice[:,:,:,my:my+kernel_shape_y], kernels)
            # stores the result of the convolution
            convolution[:,:,mx,my] = mult_sum
    '''
    
    return convolution

class ConvLinear(Layer):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my1_conv2d,
                 conv_bck=my1_conv2d,
                 conv_grad=my1_conv2d):
        """

        :param num_inp_feat_maps: int, a number of input feature maps (channels)
        :param num_out_feat_maps: int, a number of output feature maps (channels)
        :param image_shape: tuple, a shape of the image
        :param kernel_shape: tuple, a shape of the kernel
        :param stride: tuple, shift of kernels in both dimensions
        :param irange: float, initial range of the parameters
        :param rng: RandomState object, random number generator
        :param conv_fwd: handle to a convolution function used in fwd-prop
        :param conv_bck: handle to a convolution function used in backward-prop
        :param conv_grad: handle to a convolution function used in pgrads
        :return:
        """

        super(ConvLinear, self).__init__(rng=rng)

        self.num_inp_feat_maps = num_inp_feat_maps
        self.num_out_feat_maps = num_out_feat_maps
        self.image_shape = image_shape
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.irange = irange
        self.rng = rng
        self.conv_fwd = conv_fwd
        self.conv_bck = conv_bck
        self.conv_grad = conv_grad
        self.W = self.rng.uniform(
            -irange, irange,
            (self.num_inp_feat_maps, self.num_out_feat_maps, self.kernel_shape[0], self.kernel_shape[1]))
        self.b = numpy.ones((self.num_out_feat_maps, ), dtype=numpy.float64)

    def fprop(self, inputs):
        
        # run convolution for forward propagation
        conv = self.conv_fwd(inputs.astype(numpy.float64), kernels=self.W)
        
        # stores feature map's dimensions
        self.feature_map_shape_x = conv.shape[2]
        self.feature_map_shape_y = conv.shape[3]
        
        # initialize the output of forward propagation
        # (batch_size, number of output feature maps, x dimension of feature map, y dimension of feature map)
        output = numpy.zeros((conv.shape))
        
        # adds the result of convolution with biases for each feature map
        for i in range(conv.shape[1]):
            output[:, i, :, :] = conv[:, i, :, :] + self.b[i]
        
        return output

    def bprop(self, h, igrads):
        
        #input comes from 2D convolutional tensor, reshape to 4D
        if igrads.ndim == 2:
            igrads = igrads.reshape(inputs.shape[0], self.W.shape[1], numpy.sqrt(igrads.shape[1], self.W.shape[1]), -1)
            
        kernel, bias = self.get_params()
        
        # initialize an tensor with all elements equals to zero
        # (deltas' size, number of output feature maps, input's dimension x + (kernel's dimension x -1), input's dimension y + (kernel's dimension y -1)
        zero_pad = numpy.zeros((igrads.shape[0], bias.shape[0], igrads.shape[2]+(kernel.shape[2]-1)*2, igrads.shape[3]+(kernel.shape[3]-1)*2))
        
        # initialize the slice where we'll put the values of deltas in the tensor with zeros
        # (kernel's dimension x -1, input's dimension x)
        # (kernel's dimension y -1, input's dimension y)
        slice_x = slice(kernel.shape[2]-1, igrads.shape[2]+(kernel.shape[2]-1))
        slice_y = slice(kernel.shape[3]-1, igrads.shape[3]+(kernel.shape[3]-1))
        
        # put the deltas in the tensor with the zero values
        zero_pad[:, :, slice_x, slice_y] = igrads.astype(numpy.float64)
        
        rotated_W = numpy.zeros((kernel.shape))
        
        # rotates the weight tensor 180o
        for i in xrange(0,kernel.shape[0]):
            for j in xrange(0,kernel.shape[1]):
                rotated_W[i,j] = numpy.rot90(self.W[i,j],2)
                             
        rotated_W = numpy.transpose(rotated_W, (1,0,2,3))
        
        # run the convolution for backward propagation
        ograds = self.conv_bck(zero_pad, rotated_W, self.stride)
        
        return igrads, ograds

    def bprop_cost(self, h, igrads, cost):
        if cost is None or cost.get_name() == 'mse':
            # for linear layer and mean square error cost,
            # cost back-prop is the same as standard back-prop
            return self.bprop(h, igrads)
        else:
            raise NotImplementedError('convlinear.bprop_cost method not implemented '
                                      'for the %s cost' % cost.get_name())

    def pgrads(self, inputs, deltas, l1_weight=0, l2_weight=0):
        """
        Return gradients w.r.t parameters

        :param inputs, input to the i-th layer
        :param deltas, deltas computed in bprop stage up to -ith layer
        :param kwargs, key-value optional arguments
        :return list of grads w.r.t parameters dE/dW and dE/db in *exactly*
                the same order as the params are returned by get_params()
                
        Note: deltas here contain the whole chain rule leading
        from the cost up to the the i-th layer, i.e.
        dE/dy^L dy^L/da^L da^L/dh^{L-1} dh^{L-1}/da^{L-1} ... dh^{i}/da^{i}
        and here we are just asking about
          1) da^i/dW^i and 2) da^i/db^i
        since W and b are only layer's parameters
        """

        #input comes from 2D convolutional tensor, reshape to 4D
        if deltas.ndim == 2:
            deltas = deltas.reshape(deltas.shape[0], self.W.shape[1], numpy.sqrt(deltas.shape[1]/self.W.shape[1]), -1)

        #you could basically use different scalers for biases
        #and weights, but it is not implemented here like this
        l2_W_penalty, l2_b_penalty = 0, 0
        if l2_weight > 0:
            l2_W_penalty = l2_weight*self.W
            l2_b_penalty = l2_weight*self.b

        l1_W_penalty, l1_b_penalty = 0, 0
        if l1_weight > 0:
            l1_W_penalty = l1_weight*numpy.sign(self.W)
            l1_b_penalty = l1_weight*numpy.sign(self.b)
        
        # transpose the first with the second dimension so that
        # we can have the correct results at the convolution
        inputs1 = numpy.transpose(inputs, (1,0,2,3))
        
        # run the convolution for the gradients
        grad_W = self.conv_grad(inputs1, deltas, self.stride) + l2_W_penalty + l1_W_penalty
        grad_b = numpy.sum(deltas, axis=(0, 2, 3)) + l2_b_penalty + l1_b_penalty

        return [grad_W, grad_b]

    def get_params(self):
        return [self.W, self.b]

    def set_params(self, params):
        #we do not make checks here, but the order on the list
        #is assumed to be exactly the same as get_params() returns
        self.W = params[0]
        self.b = params[1]

    def get_name(self):
        return 'convlinear'

class ConvSigmoid(ConvLinear):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my1_conv2d,
                 conv_bck=my1_conv2d,
                 conv_grad=my1_conv2d):
        """

        :param num_inp_feat_maps: int, a number of input feature maps (channels)
        :param num_out_feat_maps: int, a number of output feature maps (channels)
        :param image_shape: tuple, a shape of the image
        :param kernel_shape: tuple, a shape of the kernel
        :param stride: tuple, shift of kernels in both dimensions
        :param irange: float, initial range of the parameters
        :param rng: RandomState object, random number generator
        :param conv_fwd: handle to a convolution function used in fwd-prop
        :param conv_bck: handle to a convolution function used in backward-prop
        :param conv_grad: handle to a convolution function used in pgrads
        :return:
        """

        super(ConvSigmoid, self).__init__(
        num_inp_feat_maps,
        num_out_feat_maps,
        image_shape,
        kernel_shape,
        stride,
        irange,
        rng,
        conv_fwd,
        conv_bck,
        conv_grad)

    def fprop(self, inputs):
        
        return sigmoid(super(Sigmoid, self).fprop(inputs))

    def bprop(self, h, igrads):
        
        igrads, ograds = super(Sigmoid, self).bprop(h, igrads)
        
        deltas = igrads * sigmoidDerivative(h)
        
        return deltas, ograds
    
    def get_name(self):
        return 'convsigmoid'
    
def sigmoid(x):
        
    return 1.0/(1.0+numpy.exp(-x))
    
def sigmoidDerivative(x):

    return x*(1.0-x)

class ConvRelu(ConvLinear):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my1_conv2d,
                 conv_bck=my1_conv2d,
                 conv_grad=my1_conv2d):
        """

        :param num_inp_feat_maps: int, a number of input feature maps (channels)
        :param num_out_feat_maps: int, a number of output feature maps (channels)
        :param image_shape: tuple, a shape of the image
        :param kernel_shape: tuple, a shape of the kernel
        :param stride: tuple, shift of kernels in both dimensions
        :param irange: float, initial range of the parameters
        :param rng: RandomState object, random number generator
        :param conv_fwd: handle to a convolution function used in fwd-prop
        :param conv_bck: handle to a convolution function used in backward-prop
        :param conv_grad: handle to a convolution function used in pgrads
        :return:
        """

        super(ConvRelu, self).__init__(
        num_inp_feat_maps,
        num_out_feat_maps,
        image_shape,
        kernel_shape,
        stride,
        irange,
        rng,
        conv_fwd,
        conv_bck,
        conv_grad)

    def fprop(self, inputs):
        
        a = super(ConvRelu, self).fprop(inputs)
        h = numpy.clip(a, 0, 20.0)
        
        return h

    def bprop(self, h, igrads):
  
        deltas = (h > 0)*igrads
        ___, ograds = super(ConvRelu, self).bprop(h=None, igrads=deltas)
        
        return deltas, ograds
    
    def get_name(self):
        return 'convrelu'

class ConvMaxPool2D(Layer):
    def __init__(self,
                 num_feat_maps,
                 conv_shape,
                 pool_shape=(2, 2),
                 pool_stride=(2, 2)):
        """

        :param conv_shape: tuple, a shape of the lower convolutional feature maps output
        :param pool_shape: tuple, a shape of pooling operator
        :param pool_stride: tuple, a strides for pooling operator
        :return:
        """

        super(ConvMaxPool2D, self).__init__(rng=None)
        
        self.num_feat_maps = num_feat_maps
        self.conv_shape = conv_shape
        self.pool_shape = pool_shape
        self.pool_stride = pool_stride

    def fprop(self, inputs):
        
        batch_size, num_input_feat_maps, input_shape_x, input_shape_y = inputs.shape
        pool_shape_x, pool_shape_y = self.pool_shape
        stride_x, stride_y = self.pool_stride
        
        # I calculated the shape of the pool using this web site
        # http://cs231n.github.io/convolutional-networks/
        output_shape_x = (input_shape_x - pool_shape_x) / stride_x + 1
        output_shape_y = (input_shape_y - pool_shape_y) / stride_y + 1
        
        # initialize the pseudo-weight tensor
        # (batch_size, number of input feature maps, pool's x dimension, pool's y dimension)
        Gvalue = numpy.zeros((batch_size, num_input_feat_maps, output_shape_x, output_shape_y))
        
        # initialize a tensor similar to the weight matrix but we stores the index
        # from where we take the max value
        # (batch_size, number of input feature maps, pool's x dimension, pool's y dimension)
        self.Gindex = numpy.zeros(Gvalue.shape, dtype=int)
        
        newshape = (batch_size, num_input_feat_maps, output_shape_x, output_shape_y, pool_shape_x, pool_shape_y)
    
        newstrides = inputs.itemsize*numpy.array([num_input_feat_maps*input_shape_x*input_shape_y,input_shape_x*input_shape_y,input_shape_x*stride_x,stride_x,input_shape_y,1])

        strided = ast(inputs,shape = newshape,strides = newstrides)

        Gvalue, self.Gindex = max_and_argmax(strided, axes=(4, 5))
        
        '''
        for mx in xrange(0, output_shape_x):
            # the slice on the x dimension
            slice_x = slice(mx*stride_x, mx*stride_x + pool_shape_x)
            for my in xrange(0, output_shape_y):
                # the slice on the y dimension
                slice_y = slice(my*stride_y, my*stride_y + pool_shape_y)
                
                # stores the max value and the index of the max value
                Gvalue[:,:,mx,my], self.Gindex[:,:,mx,my] = max_and_argmax(inputs[:, :, slice_x, slice_y], axes=(2, 3))
        '''    
        return Gvalue

    def bprop(self, h, igrads):
        
        batch_size, num_output_feat_maps, output_shape_x, output_shape_y = self.Gindex.shape
        
        #input comes from 2D convolutional tensor, reshape to 4D
        if igrads.ndim == 2:
            igrads = igrads.reshape((batch_size, num_output_feat_maps, output_shape_x, output_shape_y))
            
        pool_shape_x, pool_shape_y = self.pool_shape
        stride_x, stride_y = self.pool_stride
        
        input_shape_x, input_shape_y = self.conv_shape
        
        # initialize the tensor where we store the output gradients
        # (batch_size, number of input feature maps, pool's x dimension, pool's y dimension)
        ograds = numpy.zeros((batch_size, num_output_feat_maps, input_shape_x, input_shape_y))

        for mx in xrange(0, output_shape_x):
            # the slice on the x dimension
            slice_x = slice(mx*stride_x, mx*stride_x + pool_shape_x)
            for my in xrange(0, output_shape_y):
                # the slice on the y dimension
                slice_y = slice(my*stride_y, my*stride_y + pool_shape_y)
                
                # initialize the tensor where we store the gradients of the max values
                # (batch_size, number of input feature maps, pool's x dimension * pool's y dimension)
                gradients_index = numpy.zeros((batch_size, num_output_feat_maps, pool_shape_x*pool_shape_y))
                
                for i in xrange(0, num_output_feat_maps):
                    index = self.Gindex[:,:,mx,my]
                    # finds the gradients
                    gradients_index[:,i,index[0,i]] = igrads[:,i,mx,my]
                
                # stores the gradients using the gradients_index
                ograds[:, :, slice_x, slice_y] = gradients_index.reshape((batch_size, num_output_feat_maps, pool_shape_x, pool_shape_y))    
         
        return igrads, ograds

    def get_params(self):
        return []

    def pgrads(self, inputs, deltas, **kwargs):
        return []

    def set_params(self, params):
        pass

    def get_name(self):
        return 'convmaxpool2d'