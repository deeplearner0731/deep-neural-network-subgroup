import math
from keras import backend as K
from keras import Model
from keras.layers import Layer, Softmax, Input
from keras.callbacks import EarlyStopping
from keras.initializers import Constant, glorot_normal
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class_weight = {0: 1.,
                1: 50.}



@tf.custom_gradient
def dr_loss(y_true, y_pred, trt, pi):
    """
    A‑learning style loss with an explicit gradient.
    Shapes
    -------
    y_true : (B, 1)
    y_pred : (B, 1)
    trt    : (B,)   or  (B, 1)   (will be flattened)
    pi     : (B,)   or  (B, 1)
    """
    # ---- make sure trt and pi are 1‑D ----
    trt = tf.reshape(trt, (-1,))
    pi  = tf.reshape(pi,  (-1,))

    c = (trt + 1.0)/2.0 - pi          # (B,)
    c = tf.expand_dims(c, -1)         # (B, 1)   ← **rank 2**, not 3

    diff  = y_true - c * y_pred       # (B, 1)
    value = tf.sqrt(tf.reduce_mean(tf.square(diff)))

    def grad(dy):                     # dy is the upstream scalar
        g = -c * diff                 # (B, 1)  ← still rank‑2
        denom = tf.sqrt(tf.reduce_mean(tf.square(diff))) \
                * tf.cast(tf.size(diff), tf.float32)
        g = g / denom                 # (B, 1)

        # return grads w.r.t: y_true, y_pred, trt, pi
        return dy * g, dy * g, None, None

    return value, grad



class ConcreteSelect(Layer):
    
    def __init__(self, output_dim, start_temp = 10.0, min_temp = 0.1, alpha = 0.99999, **kwargs):
        self.output_dim = output_dim
        self.start_temp = start_temp
        
        self.min_temp = K.constant(min_temp)
        self.alpha = K.constant(alpha)
        super(ConcreteSelect, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.temp = self.add_weight(name = 'temp', shape = [], initializer = Constant(self.start_temp), trainable = False)
        self.logits = self.add_weight(name = 'logits', shape = [self.output_dim, input_shape[1]], initializer = glorot_normal(), trainable = True)
        super(ConcreteSelect, self).build(input_shape)
        
    def call(self, X, training = None):
        uniform = K.random_uniform(self.logits.shape, K.epsilon(), 1.0)
        gumbel = -K.log(-K.log(uniform))
        temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha))
        noisy_logits = (self.logits + gumbel) / temp
        samples = K.softmax(noisy_logits)
        
        discrete_logits = K.one_hot(K.argmax(self.logits), self.logits.shape[1])
        
        self.selections = K.in_train_phase(samples, discrete_logits, training)
        Y = K.dot(X, K.transpose(self.selections))
        
        return Y
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
class StopperCallback(EarlyStopping):
    
    def __init__(self, mean_max_target = 0.998):
        self.mean_max_target = mean_max_target
        super(StopperCallback, self).__init__(monitor = '', patience = float('inf'), verbose = 1, mode = 'max', baseline = self.mean_max_target)
    
    def on_epoch_begin(self, epoch, logs = None):
        
        #print('mean max of probabilities:', self.get_monitor_value(logs), '- temperature', K.get_value(self.model.get_layer('concrete_select').temp))
        return
        
        
        #print( K.get_value(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis = -1)))
        #print(K.get_value(K.max(self.model.get_layer('concrete_select').selections, axis = -1)))
    
    def get_monitor_value(self, logs):
        monitor_value = K.get_value(K.mean(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis = -1)))
        return monitor_value


class ConcreteAutoencoderFeatureSelector():
    
    def __init__(self, K, output_function, num_epochs = 300, batch_size = None, learning_rate = 0.005, start_temp = 10.0, min_temp = 0.05, tryout_limit = 1,loss_name='A',trt= None,pi= None,ver=0):
        self.K = K
        self.output_function = output_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.tryout_limit = tryout_limit
        self.loss_name=loss_name
        self.trt=trt
        self.pi=pi
        self.show=ver
        
    
    def _loss_a(self, trt_full, pi_full):
        """
        Returns a closure loss(y_true, y_pred) that captures trt and pi.
        Assumes rows are fed in the same order as trt_full / pi_full.
        """
        trt_const = tf.constant(trt_full.reshape(-1), dtype=tf.float32)
        pi_const  = tf.constant(pi_full.reshape(-1),  dtype=tf.float32)

        def loss(y_true, y_pred):
            bsz = tf.shape(y_true)[0]
            trt_batch = trt_const[:bsz]
            pi_batch  = pi_const[:bsz]
            return dr_loss(y_true, y_pred, trt_batch, pi_batch)

        return loss


        
        
    def fit(self, X, Y = None, val_X = None, val_Y = None):
        if Y is None:
            Y = X
        assert len(X) == len(Y)
        validation_data = None
        if val_X is not None and val_Y is not None:
            assert len(val_X) == len(val_Y)
            validation_data = (val_X, val_Y)
        
        if self.batch_size is None:
            
            self.batch_size = max(len(X) // 256, 16)
        
        num_epochs = self.num_epochs
        steps_per_epoch = (len(X) + self.batch_size - 1) // self.batch_size
        
        for i in range(self.tryout_limit):
            
            K.set_learning_phase(1)
            
            inputs = Input(shape = X.shape[1:])

            alpha = math.exp(math.log(self.min_temp / self.start_temp) / (num_epochs * steps_per_epoch))
            
            self.concrete_select = ConcreteSelect(self.K, self.start_temp, self.min_temp, alpha, name = 'concrete_select')

            selected_features = self.concrete_select(inputs)

            outputs = self.output_function(selected_features)

            self.model = Model(inputs, outputs)
            
        
                
            #self.model.compile(Adam(self.learning_rate), loss = self.custom_loss(self.trt,self.pi))
            if self.loss_name == "A":
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                    loss=self._loss_a(self.trt, self.pi),
                    run_eagerly=False         # eager not needed; keeps it fast
                )
            stopper_callback = StopperCallback()

            hist = self.model.fit(X, Y, self.batch_size, num_epochs, verbose = self.show, callbacks = [stopper_callback], validation_data = validation_data)
            
            
            
            
            
            if K.get_value(K.mean(K.max(K.softmax(self.concrete_select.logits, axis = -1)))) >= stopper_callback.mean_max_target:
                break
            
            num_epochs *= 2
        
        self.probabilities = K.get_value(K.softmax(self.model.get_layer('concrete_select').logits))
        self.indices = K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))
        self.outputs=outputs
        return self
    
    def get_indices(self):
        return K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))
    
    def get_mask(self):
        return K.get_value(K.sum(K.one_hot(K.argmax(self.model.get_layer('concrete_select').logits), self.model.get_layer('concrete_select').logits.shape[1]), axis = 0))
    
    def transform(self, X):
        return X[self.get_indices()]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self, indices = False):
        return self.get_indices() if indices else self.get_mask()
    
    def get_params(self):
        return self.model