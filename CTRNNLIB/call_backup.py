def call_correct_implementation(self, inputs, states, training=None):
    # v = states[0]  # previous memory state # v value
    z = states[1]  # previous carry state # z value
    # tf.print('input')
    # tf.print(inputs, output_stream=sys.stdout)

    # print("alpha: ", self.alpha)
    # print("displacement: ", self.displacement)
    # print("stopper: ", self.stopper)
    # print("z_factor: ", self.z_factor)
    # print("other_factor: ", self.v_factor)

    z_factor_ = tf.constant(self.z_factor)
    a = tf.constant(self.v_factor)  # define as hyper-paramter in the future
    d = tf.constant(self.displacement)  # define as hyper-paramter in the future
    stopper_ = tf.constant(self.stopper)
    alpha_ = tf.constant(self.alpha)  # alpha of the sigmiod to make it sharper (approximate to unit step)
    d_minus_stopper = tf.math.subtract(d, stopper_)
    # tf.print(d_minus_stopper, output_stream=sys.stdout)

    z_mul_z_factor = tf.math.multiply(z, z_factor_)
    d_minus_z = tf.math.subtract(d, z)
    d_minus_z_squared = backend.square(d_minus_z)
    second_term_factor = tf.math.divide(a, d_minus_z_squared)
    # print("implementaionx",self.implementation)

    inputs_i = inputs

    k_i, _, _, _ = tf.split(
        self.kernel, num_or_size_splits=4, axis=1)
    w_mul_v = backend.dot(inputs_i, k_i)  # W*V
    b_i, _, _, _ = tf.split(
        self.bias, num_or_size_splits=4, axis=0)
    w_mul_v_add_b = backend.bias_add(w_mul_v, b_i)  # W*V+b
    x_i_squared = backend.square(w_mul_v_add_b)  # build squared function backend.square(w_mul_v_add_b)
    term2 = tf.math.multiply(second_term_factor, x_i_squared)  # element wise mulitpilcation
    new_z = tf.math.add(z_mul_z_factor, term2)

    new_z_minus_d_minus_stopper = tf.math.subtract(new_z, d_minus_stopper)
    d_minuse_new_z = tf.math.subtract(d, new_z)
    # tf.print(d_minuse_new_z)

    gain = tf.math.pow(10.0, 13)

    activation_function_ = tf.math.divide(self.eps_MUL_A, d_minuse_new_z)

    activation_function_ = tf.math.multiply(activation_function_, gain)
    # tf.print(activation_function_)

    new_v = tf.math.multiply(activation_function_, w_mul_v)
    sigmiod__new_z_minus_d = tf.keras.activations.sigmoid(
        tf.math.multiply(new_z_minus_d_minus_stopper, self.alpha))  # 5 after five zeros it stop learnig
    new_v = sigmiod__new_z_minus_d
    # new_v=tf.math.multiply(sigmiod__new_z_minus_d, gain)
    # new_v=tf.math.multiply(sigmiod__new_z_minus_d, w_mul_v)

    # tf.print(new_v, output_stream=sys.stdout)
    # tf.print(k_i, output_stream=sys.stdout)
    # tf.print(new_z, output_stream=sys.stdout)
    # tf.print("*"*10, output_stream=sys.stdout)
    # tf.print(new_z_minus_d_minus_stopper, output_stream=sys.stdout)

    # new_v=backend.bias_add(new_v, b_i)
    # tf.print('output')

    # tf.print(new_v, output_stream=sys.stdout)
    # tf.print(new_z, output_stream=sys.stdout)

    return new_v, [new_v, new_z]


def call_without_VW(self, inputs, states, training=None):
    # v = states[0]  # previous memory state # v value
    z = states[1]  # previous carry state # z value
    # tf.print('input')
    # tf.print(inputs, output_stream=sys.stdout)

    # print("alpha: ", self.alpha)
    # print("displacement: ", self.displacement)
    # print("stopper: ", self.stopper)
    # print("z_factor: ", self.z_factor)
    # print("other_factor: ", self.v_factor)

    z_factor_ = tf.constant(self.z_factor)
    a = tf.constant(self.v_factor)  # define as hyper-paramter in the future
    d = tf.constant(self.displacement)  # define as hyper-paramter in the future
    stopper_ = tf.constant(self.stopper)
    alpha_ = tf.constant(self.alpha)  # alpha of the sigmiod to make it sharper (approximate to unit step)
    d_minus_stopper = tf.math.subtract(d, stopper_)
    # tf.print(d_minus_stopper, output_stream=sys.stdout)

    z_mul_z_factor = tf.math.multiply(z, z_factor_)
    d_minus_z = tf.math.subtract(d, z)
    d_minus_z_squared = backend.square(d_minus_z)
    second_term_factor = tf.math.divide(a, d_minus_z_squared)
    # print("implementaionx",self.implementation)

    inputs_i = inputs

    k_i, _, _, _ = tf.split(
        self.kernel, num_or_size_splits=4, axis=1)
    w_mul_v = backend.dot(inputs_i, k_i)  # W*V
    b_i, _, _, _ = tf.split(
        self.bias, num_or_size_splits=4, axis=0)
    w_mul_v_add_b = backend.bias_add(w_mul_v, b_i)  # W*V+b
    x_i_squared = backend.square(w_mul_v_add_b)  # build squared function backend.square(w_mul_v_add_b)
    term2 = tf.math.multiply(second_term_factor, x_i_squared)  # element wise mulitpilcation
    new_z = tf.math.add(z_mul_z_factor, term2)

    new_z_minus_d_minus_stopper = tf.math.subtract(new_z, d_minus_stopper)
    d_minuse_new_z = tf.math.subtract(d, new_z)
    # tf.print(d_minuse_new_z)

    gain = tf.math.pow(10.0, 13)

    activation_function_ = tf.math.divide(self.eps_MUL_A, d_minuse_new_z)

    activation_function_ = tf.math.multiply(activation_function_, gain)
    new_v = activation_function_
    # tf.print(activation_function_)

    # new_v=tf.math.multiply(activation_function_, w_mul_v)
    # sigmiod__new_z_minus_d=tf.keras.activations.sigmoid(tf.math.multiply(new_z_minus_d_minus_stopper, self.alpha)) #5 after five zeros it stop learnig
    # new_v=tf.math.multiply(sigmiod__new_z_minus_d, w_mul_v)

    # tf.print(new_v, output_stream=sys.stdout)
    # tf.print(k_i, output_stream=sys.stdout)
    # tf.print(new_z, output_stream=sys.stdout)
    # tf.print("*"*10, output_stream=sys.stdout)
    # tf.print(new_z_minus_d_minus_stopper, output_stream=sys.stdout)

    # new_v=backend.bias_add(new_v, b_i)
    # tf.print('output')

    # tf.print(new_v, output_stream=sys.stdout)
    # tf.print(new_z, output_stream=sys.stdout)

    return new_v, [new_v, new_z]

  def call_orginal(self, inputs, states, training=None):
    #v = states[0]  # previous memory state # v value
    z = states[1]  # previous carry state # z value
    #tf.print('input')
    #tf.print(inputs, output_stream=sys.stdout)

    # print("alpha: ", self.alpha)
    # print("displacement: ", self.displacement)
    # print("stopper: ", self.stopper)
    # print("z_factor: ", self.z_factor)
    # print("other_factor: ", self.v_factor)


    z_factor_=tf.constant(self.z_factor)
    a= tf.constant(self.v_factor) # define as hyper-paramter in the future
    d= tf.constant(self.displacement)# define as hyper-paramter in the future
    stopper_=tf.constant(self.stopper)
    alpha_=tf.constant(self.alpha) # alpha of the sigmiod to make it sharper (approximate to unit step)
    d_minus_stopper = tf.math.subtract(d, stopper_)
    #tf.print(d_minus_stopper, output_stream=sys.stdout)



    z_mul_z_factor= tf.math.multiply(z , z_factor_)
    d_minus_z=tf.math.subtract(d,z)
    d_minus_z_squared=backend.square(d_minus_z)
    second_term_factor=tf.math.divide(a, d_minus_z_squared)
    #print("implementaionx",self.implementation)

    inputs_i = inputs

    k_i, _, _, _ = tf.split(
          self.kernel, num_or_size_splits=4, axis=1)
    w_mul_v = backend.dot(inputs_i, k_i)#W*V
    b_i, _, _, _ = tf.split(
        self.bias, num_or_size_splits=4, axis=0)
    w_mul_v_add_b = backend.bias_add(w_mul_v, b_i) #W*V+b
    x_i_squared=backend.square(w_mul_v_add_b) #build squared function backend.square(w_mul_v_add_b)
    term2=tf.math.multiply(second_term_factor,x_i_squared)# element wise mulitpilcation
    new_z=tf.math.add(z_mul_z_factor,term2)

    new_z_minus_d_minus_stopper=tf.math.subtract(new_z,d_minus_stopper)
    d_minuse_new_z=tf.math.subtract(d, new_z)
    #tf.print(d_minuse_new_z)

    gain=tf.math.pow(10.0,13)

    activation_function_ = tf.math.divide(self.eps_MUL_A,d_minuse_new_z)

    activation_function_ = tf.math.multiply(activation_function_, gain)
    #tf.print(activation_function_)

    new_v=tf.math.multiply(activation_function_, w_mul_v)
    # sigmiod__new_z_minus_d=tf.keras.activations.sigmoid(tf.math.multiply(new_z_minus_d_minus_stopper, self.alpha)) #5 after five zeros it stop learnig
    # new_v=tf.math.multiply(sigmiod__new_z_minus_d, w_mul_v)

    #tf.print(new_v, output_stream=sys.stdout)
    # tf.print(k_i, output_stream=sys.stdout)
    # tf.print(new_z, output_stream=sys.stdout)
    # tf.print("*"*10, output_stream=sys.stdout)
    # tf.print(new_z_minus_d_minus_stopper, output_stream=sys.stdout)


    #new_v=backend.bias_add(new_v, b_i)
    #tf.print('output')

    #tf.print(new_v, output_stream=sys.stdout)
    #tf.print(new_z, output_stream=sys.stdout)

    return new_v, [new_v, new_z]
