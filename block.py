from tensorflow.keras.layers import Conv1D, SeparableConv1D, MaxPooling1D, BatchNormalization, Activation
from tensorflow.keras.layers import GlobalAveragePooling1D, Dropout, Dense, Input, Reshape, multiply, add


def MBConv(activation="relu", drop_rate=0., kernel_size=3, filters_in=32, filters_out=16, strides=1, expand_ratio=1,
           se_ratio=0., id_skip=True, name=""):
    """
    Mobile inverted Bottleneck Convolution layer

    :param activation: activation function
    :param drop_rate: float between 0 and 1, fraction of the input units to drop
    :param kernel_size: integer, the dimension of the convolution windo
    :param filters_in: integer, the number of input filters
    :param filters_out: integer, the number of output filters
    :param strides: integer, the stride of the convolution
    :param expand_ratio: integer, scaling coefficient for the input filters
    :param se_ratio: float between 0 and 1, fraction to squeeze the input filters
    :param id_skip: boolean
    :param name: string, block label
    :return:
    """
    def __mbconv(x):
        inputs = x
        # Expansion phase
        filters = filters_in * expand_ratio
        if expand_ratio != 1:
            x = Conv1D(
                filters,
                1,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                name=name + "expand_conv"
            )(x)
            x = BatchNormalization(name=name + "expand_bn")(x)
            x = Activation(activation, name=name + "expand_activation")(x)
        else:
            x = inputs

        # Depthwise Convolution
        conv_pad = 'same'
        x = SeparableConv1D(
            int(x.shape[-1]),
            kernel_size,
            strides,
            padding=conv_pad,
            use_bias=False,
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            name=name + "dwconv"
        )(x)
        x = BatchNormalization(name=name + "bn")(x)
        x = Activation(activation, name=name + "activation")(x)

        # Squeeze and Excitation phase
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))
            se = GlobalAveragePooling1D(name=name + "se_squeeze")(x)
            se = Reshape((1, filters), name=name + "se_reshape")(se)
            se = Conv1D(
                filters_se,
                1,
                padding="same",
                activation=activation,
                kernel_initializer="he_normal",
                name=name + "se_reduce"
            )(se)
            se = Conv1D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
                kernel_initializer="he_normal",
                name=name + "se_expand"
            )(se)
            x = multiply([x, se], name=name + "se_excite")

        # Output phase
        x = Conv1D(
            filters_out,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "project_conv"
        )(x)
        x = BatchNormalization(name=name + "project_bn")(x)

        if id_skip and strides == 1 and filters_in == filters_out:
            if drop_rate > 0:
                x = Dropout(drop_rate, name=name + "drop")(x)
            x = add([x, inputs], name=name + "add")

        return x

    return __mbconv


def MBConvBlock(repeats, kernel_size, filters_in, filters_out, expand_ratio, skip_op, strides, se_ratio, block_id=1):
    """
    MBConv Block

    :param repeats: integer, the number of convolutional layers
    :param kernel_size: integer, the dimension of the convolution window
    :param filters_in: integer, the number of input filters
    :param filters_out: integer, the number of output filters
    :param expand_ratio: integer, scaling coefficient for the input filters
    :param skip_op: One of None, "pool", "identity", skip operation
    :param strides: integer, the stride ot the convolution
    :param se_ratio: float between 0 and 1, fraction to squeeze the input filters
    :param block_id: integer larger than 1, the block id
    :return:
    """
    # check inputs
    assert skip_op in [None, "pool", "identity"], "{} is not valid skip_op".format(skip_op)

    def __block(x):
        for i in range(repeats):
            # The first block needs to take care of stride and filter size increase
            x = MBConv(activation="relu", drop_rate=0.2,
                       name="block{}{}_".format(block_id, chr(i + 97)),
                       kernel_size=kernel_size,
                       filters_in=filters_in if i == 0 else filters_out,
                       filters_out=filters_out,
                       expand_ratio=expand_ratio,
                       id_skip=True if skip_op == "identity" else False,
                       strides=strides if i == 1 else 1,
                       se_ratio=se_ratio
                       )(x)

            if skip_op == "pool":
                x = MaxPooling1D(name="block{}{}_pool".format(block_id, chr(i + 97)), padding="same")(x)

        return x

    return __block


def ConvBlock(repeats, kernel_size, filters, skip_op, strides, se_ratio, block_id=1):
    """
    Conv Block
    :param repeats: integer, the number of convolutional layers
    :param kernel_size: integer, the dimension of the convolution window
    :param filters: integer, the number of filters
    :param skip_op: One of None, "pool", "identity", skip operation
    :param strides: integer, the stride ot the convolution
    :param se_ratio: float between 0 and 1, fraction to squeeze the input filters
    :param block_id: block_id: integer larger than 1, the block id
    :return:
    """
    # check inputs
    assert skip_op in [None, "pool", "identity"], "{} is not valid skip_op".format(skip_op)

    def __block(x):
        inputs = x

        for i in range(repeats):
            x = Conv1D(filters,
                       kernel_size,
                       strides,
                       padding='same',
                       activation="relu",
                       kernel_initializer="he_normal",
                       name="block{}{}_conv".format(block_id, chr(i + 97))
                       )(x)

        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters * se_ratio))
            se = GlobalAveragePooling1D(name="block{}_se_squeeze".format(block_id))(x)
            se = Reshape((1, filters), name="block{}_se_reshape".format(block_id))(se)
            se = Conv1D(
                filters_se,
                1,
                padding='same',
                activation="relu",
                kernel_initializer="he_normal",
                name="block{}_se_reduce".format(block_id)
            )(se)
            se = Conv1D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
                kernel_initializer="he_normal",
                name="block{}_se_expand".format(block_id)
            )(se)
            x = multiply([x, se], name="block{}_se_excite".format(block_id))

        # skip operation
        if skip_op == "pool":
            x = MaxPooling1D(name="block{}_pool".format(block_id), padding="same")(x)
        elif skip_op == "identity":
            if strides == 1:
                shortcut = inputs
                if int(inputs.shape[-1]) != int(x.shape[-1]):
                    shortcut = Conv1D(int(x.shape[-1]), 1, strides=strides,
                                      kernel_initializer="he_normal", padding="valid",
                                      name="block{}_shortcut".format(block_id))(x)
                x = add([x, shortcut], name="block{}_add".format(block_id))

        return x

    return __block


def SeparableConvBlock(repeats, kernel_size, skip_op, strides, se_ratio, block_id=1):
    """
    Separable Conv Block
    :param repeats: integer, the number of convolutional layers
    :param kernel_size: integer, the dimension of the convolution window
    :param skip_op: One of None, "pool", "identity", skip operation
    :param strides: integer, the stride ot the convolution
    :param se_ratio: float between 0 and 1, fraction to squeeze the input filters
    :param block_id: block_id: integer larger than 1, the block id
    :return:
    """
    # check inputs
    assert skip_op in [None, "pool", "identity"], "{} is not valid skip_op".format(skip_op)

    def __block(x):
        inputs = x
        filters = int(x.shape[-1])

        for i in range(repeats):
            x = SeparableConv1D(filters,
                                kernel_size,
                                strides,
                                padding='same',
                                activation="relu",
                                depthwise_initializer="he_normal",
                                pointwise_initializer="he_normal",
                                name="block{}{}_conv".format(block_id, chr(i + 97))
                                )(x)

        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters * se_ratio))
            se = GlobalAveragePooling1D(name="block{}_se_squeeze".format(block_id))(x)
            se = Reshape((1, filters), name="block{}_se_reshape".format(block_id))(se)
            se = Conv1D(
                filters_se,
                1,
                padding='same',
                activation="relu",
                kernel_initializer="he_normal",
                name="block{}_se_reduce".format(block_id)
            )(se)
            se = Conv1D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
                kernel_initializer="he_normal",
                name="block{}_se_expand".format(block_id)
            )(se)
            x = multiply([x, se], name="block{}_se_excite".format(block_id))

        # skip operation
        if skip_op == "pool":
            x = MaxPooling1D(name="block{}_pool".format(block_id), padding="same")(x)
        elif skip_op == "identity":
            if strides == 1:
                shortcut = inputs
                if int(inputs.shape[-1]) != int(x.shape[-1]):
                    shortcut = Conv1D(int(x.shape[-1]), 1, strides=strides,
                                      kernel_initializer="he_normal", padding="valid",
                                      name="block{}_shortcut".format(block_id))(x)
                x = add([x, shortcut], name="block{}_add".format(block_id))

        return x

    return __block


if __name__ == "__main__":
    from tensorflow.keras.models import Model

    inputs = Input(shape=(256 * 3, 1))
    # MBConvBlock
    outputs = MBConvBlock(
        repeats=3,
        kernel_size=2,
        filters_in=16,
        filters_out=32,
        expand_ratio=1,
        skip_op="identity",
        strides=1,
        se_ratio=0.25,
        block_id=1
    )(inputs)

    # ConvBlock
    outputs = ConvBlock(
        repeats=2,
        kernel_size=2,
        filters=64,
        skip_op="pool",
        strides=1,
        se_ratio=0.,
        block_id=2
    )(outputs)

    # SeparableConvBlock
    outputs = SeparableConvBlock(
        repeats=2,
        kernel_size=2,
        skip_op=None,
        strides=1,
        se_ratio=0.25,
        block_id=3
    )(outputs)

    outputs = Dense(6, "softmax")(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
"""
model.summary()

Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 768, 1)]     0                                            
__________________________________________________________________________________________________
block1a_dwconv (SeparableConv1D (None, 24, 1)        3           input_1[0][0]                    
__________________________________________________________________________________________________
block1a_bn (BatchNormalization) (None, 24, 1)        4           block1a_dwconv[0][0]             
__________________________________________________________________________________________________
block1a_activation (Activation) (None, 24, 1)        0           block1a_bn[0][0]                 
__________________________________________________________________________________________________
block1a_se_squeeze (GlobalAvera (None, 1)            0           block1a_activation[0][0]         
__________________________________________________________________________________________________
block1a_se_reshape (Reshape)    (None, 1, 16)        0           block1a_se_squeeze[0][0]         
__________________________________________________________________________________________________
block1a_se_reduce (Conv1D)      (None, 1, 4)         68          block1a_se_reshape[0][0]         
__________________________________________________________________________________________________
block1a_se_expand (Conv1D)      (None, 1, 16)        80          block1a_se_reduce[0][0]          
__________________________________________________________________________________________________
block1a_se_excite (Multiply)    (None, 24, 16)       0           block1a_activation[0][0]         
                                                                 block1a_se_expand[0][0]          
__________________________________________________________________________________________________
block1a_project_conv (Conv1D)   (None, 24, 32)       512         block1a_se_excite[0][0]          
__________________________________________________________________________________________________
block1a_project_bn (BatchNormal (None, 24, 32)       128         block1a_project_conv[0][0]       
__________________________________________________________________________________________________
block1b_dwconv (SeparableConv1D (None, 24, 32)       1088        block1a_project_bn[0][0]         
__________________________________________________________________________________________________
block1b_bn (BatchNormalization) (None, 24, 32)       128         block1b_dwconv[0][0]             
__________________________________________________________________________________________________
block1b_activation (Activation) (None, 24, 32)       0           block1b_bn[0][0]                 
__________________________________________________________________________________________________
block1b_se_squeeze (GlobalAvera (None, 32)           0           block1b_activation[0][0]         
__________________________________________________________________________________________________
block1b_se_reshape (Reshape)    (None, 1, 32)        0           block1b_se_squeeze[0][0]         
__________________________________________________________________________________________________
block1b_se_reduce (Conv1D)      (None, 1, 8)         264         block1b_se_reshape[0][0]         
__________________________________________________________________________________________________
block1b_se_expand (Conv1D)      (None, 1, 32)        288         block1b_se_reduce[0][0]          
__________________________________________________________________________________________________
block1b_se_excite (Multiply)    (None, 24, 32)       0           block1b_activation[0][0]         
                                                                 block1b_se_expand[0][0]          
__________________________________________________________________________________________________
block1b_project_conv (Conv1D)   (None, 24, 32)       1024        block1b_se_excite[0][0]          
__________________________________________________________________________________________________
block1b_project_bn (BatchNormal (None, 24, 32)       128         block1b_project_conv[0][0]       
__________________________________________________________________________________________________
block1b_drop (Dropout)          (None, 24, 32)       0           block1b_project_bn[0][0]         
__________________________________________________________________________________________________
block1b_add (Add)               (None, 24, 32)       0           block1b_drop[0][0]               
                                                                 block1a_project_bn[0][0]         
__________________________________________________________________________________________________
block1c_dwconv (SeparableConv1D (None, 24, 32)       1088        block1b_add[0][0]                
__________________________________________________________________________________________________
block1c_bn (BatchNormalization) (None, 24, 32)       128         block1c_dwconv[0][0]             
__________________________________________________________________________________________________
block1c_activation (Activation) (None, 24, 32)       0           block1c_bn[0][0]                 
__________________________________________________________________________________________________
block1c_se_squeeze (GlobalAvera (None, 32)           0           block1c_activation[0][0]         
__________________________________________________________________________________________________
block1c_se_reshape (Reshape)    (None, 1, 32)        0           block1c_se_squeeze[0][0]         
__________________________________________________________________________________________________
block1c_se_reduce (Conv1D)      (None, 1, 8)         264         block1c_se_reshape[0][0]         
__________________________________________________________________________________________________
block1c_se_expand (Conv1D)      (None, 1, 32)        288         block1c_se_reduce[0][0]          
__________________________________________________________________________________________________
block1c_se_excite (Multiply)    (None, 24, 32)       0           block1c_activation[0][0]         
                                                                 block1c_se_expand[0][0]          
__________________________________________________________________________________________________
block1c_project_conv (Conv1D)   (None, 24, 32)       1024        block1c_se_excite[0][0]          
__________________________________________________________________________________________________
block1c_project_bn (BatchNormal (None, 24, 32)       128         block1c_project_conv[0][0]       
__________________________________________________________________________________________________
block1c_drop (Dropout)          (None, 24, 32)       0           block1c_project_bn[0][0]         
__________________________________________________________________________________________________
block1c_add (Add)               (None, 24, 32)       0           block1c_drop[0][0]               
                                                                 block1b_add[0][0]                
__________________________________________________________________________________________________
block2a_conv (Conv1D)           (None, 24, 64)       4160        block1c_add[0][0]                
__________________________________________________________________________________________________
block2b_conv (Conv1D)           (None, 24, 64)       8256        block2a_conv[0][0]               
__________________________________________________________________________________________________
block2_pool (MaxPooling1D)      (None, 12, 64)       0           block2b_conv[0][0]               
__________________________________________________________________________________________________
block3a_conv (SeparableConv1D)  (None, 12, 64)       4288        block2_pool[0][0]                
__________________________________________________________________________________________________
block3b_conv (SeparableConv1D)  (None, 12, 64)       4288        block3a_conv[0][0]               
__________________________________________________________________________________________________
block3_se_squeeze (GlobalAverag (None, 64)           0           block3b_conv[0][0]               
__________________________________________________________________________________________________
block3_se_reshape (Reshape)     (None, 1, 64)        0           block3_se_squeeze[0][0]          
__________________________________________________________________________________________________
block3_se_reduce (Conv1D)       (None, 1, 16)        1040        block3_se_reshape[0][0]          
__________________________________________________________________________________________________
block3_se_expand (Conv1D)       (None, 1, 64)        1088        block3_se_reduce[0][0]           
__________________________________________________________________________________________________
block3_se_excite (Multiply)     (None, 12, 64)       0           block3b_conv[0][0]               
                                                                 block3_se_expand[0][0]           
__________________________________________________________________________________________________
dense (Dense)                   (None, 12, 6)        390         block3_se_excite[0][0]           
==================================================================================================
Total params: 30,145
Trainable params: 29,823
Non-trainable params: 322
__________________________________________________________________________________________________
None
"""
