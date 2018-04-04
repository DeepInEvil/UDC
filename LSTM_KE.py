import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.tensor as T
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
import warnings
import math
from torch.nn.utils.rnn import PackedSequence
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend


class RNNCellBase(Module):

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def LSTMCell_func(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, topic=None, topic_i_w=None, topic_f_w=None, drop=None, recurrent_drop=None):
    # if input.is_cuda:
    #     igates = F.linear(input, w_ih)
    #     hgates = F.linear(hidden[0], w_hh)
    #     state = fusedBackend.LSTMFused.apply
    #     return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    #print topic.size(), topic_i_w.size(), ingate.size()
    #ingate = F.sigmoid(ingate + F.linear(topic, topic_i_w))
    #print ingate.size()
    ingate = F.sigmoid(ingate)

    #forgetgate = F.sigmoid(forgetgate + F.linear(topic, topic_f_w))
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)
    hy = drop(hy)

    return hy, cy


def LSTMtopicCell_func(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, topic=None, topic_i_w=None, topic_f_w=None, drop=None, recurrent_drop=None):
    # if input.is_cuda:
    #     igates = F.linear(input, w_ih)
    #     hgates = F.linear(hidden[0], w_hh)
    #     state = fusedBackend.LSTMFused.apply
    #     return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    #print topic.size(), topic_i_w.size(), ingate.size()
    #print (topic.size(), topic_i_w.size(), ingate.size())
    ingate = F.sigmoid(ingate + F.linear(topic, topic_i_w))
    #print ingate.size()
    #ingate = F.sigmoid(ingate)

    forgetgate = F.sigmoid(forgetgate + F.linear(topic, topic_f_w))
    #forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)
    hy = drop(hy)

    return hy, cy


class LSTMCell(RNNCellBase):
    r"""A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, (h_0, c_0)
        - **input** (batch, input_size): tensor containing input features
        - **h_0** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
        - **c_0** (batch, hidden_size): tensor containing the initial cell state
          for each element in the batch.

    Outputs: h_1, c_1
        - **h_1** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
        - **c_1** (batch, hidden_size): tensor containing the next cell state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size x input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`

    Examples::

        >>> rnn = nn.LSTMCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> cx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=False, drop=0.0):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_drop = nn.Dropout(p=drop)

        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        #self.weight_ih = self.weight_drop(self.weight_ih)
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        #self.weight_hh = self.recurrent_drop(self.weight_hh)
        #self.topic_w_i = Parameter(torch.Tensor(hidden_size, topic_size))
        #self.topic_w_f = Parameter(torch.Tensor(hidden_size, topic_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        #self.check_forward_input(input, topic)
        #self.check_forward_hidden(input, hx[0], '[0]')
        #self.check_forward_hidden(input, hx[1], '[1]')
        # return LSTMCell_func(
        #     input, hx,
        #     self.weight_ih, self.weight_hh,
        #     self.bias_ih, self.bias_hh, topic, self.topic_w_i, self.topic_w_f, self.weight_drop
        # )
        return LSTMCell_func(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh, drop=self.weight_drop
        )


class LSTMKECell(RNNCellBase):
    r"""A long short-term memory (LSTM) cell with topic vectors as additional inputs.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, (h_0, c_0)
        - **input** (batch, input_size): tensor containing input features
        - **h_0** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
        - **c_0** (batch, hidden_size): tensor containing the initial cell state
          for each element in the batch.

    Outputs: h_1, c_1
        - **h_1** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
        - **c_1** (batch, hidden_size): tensor containing the next cell state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size x input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`

    Examples::

        >>> rnn = nn.LSTMCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> cx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
    """

    def __init__(self, input_size, hidden_size, topic_size, bias=False, drop=0.0):
        super(LSTMKECell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_drop = nn.Dropout(p=drop)

        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        #self.weight_ih = self.weight_drop(self.weight_ih)
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        #self.weight_hh = self.recurrent_drop(self.weight_hh)
        self.topic_w_i = Parameter(torch.Tensor(hidden_size, topic_size))
        self.topic_w_f = Parameter(torch.Tensor(hidden_size, topic_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, topic, hx):
        #print topic.size()
        #self.check_forward_input(input, topic)
        #self.check_forward_hidden(input, hx[0], '[0]')
        #self.check_forward_hidden(input, hx[1], '[1]')
        return LSTMtopicCell_func(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh, topic, self.topic_w_i, self.topic_w_f, self.weight_drop
        )
        # return LSTMCell_func(
        #     input, hx,
        #     self.weight_ih, self.weight_hh,
        #     self.bias_ih, self.bias_hh,
        # )