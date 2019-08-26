import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision
from mxnet import gluon



class BASE(mx.gluon.HybridBlock):
    def __init__(self, num_of_pts=98, **kwargs):
        super(BASE, self).__init__(**kwargs)
        self.pts_num = num_of_pts
        self.lmks_net = mx.gluon.nn.HybridSequential()
        self.lmks_net.add(

            nn.Conv2D(channels=16, kernel_size=(3,3), strides=(2,2), padding=(1,1)),
            nn.BatchNorm(),
            nn.Activation('relu'),

            nn.Conv2D(channels=32, kernel_size=(3,3), strides=(1,1), padding=(1,1)),
            nn.BatchNorm(),
            nn.Activation('relu'),
            
            nn.Conv2D(channels=32, kernel_size=(3,3), strides=(2,2), padding=(1,1)),
            nn.BatchNorm(),
            nn.Activation('relu'),

            nn.Conv2D(channels=32, kernel_size=(3,3), strides=(1,1), padding=(1,1)),
            nn.BatchNorm(),
            nn.Activation('relu'),

            nn.Conv2D(channels=32, kernel_size=(3,3), strides=(2,2), padding=(1,1)),
            nn.BatchNorm(),
            nn.Activation('relu'),

            nn.Conv2D(channels=64, kernel_size=(3,3), strides=(1,1), padding=(1,1)),
            nn.BatchNorm(),
            nn.Activation('relu'),

            nn.Conv2D(channels=64, kernel_size=(3,3), strides=(2,2), padding=(1,1)),
            nn.BatchNorm(),
            nn.Activation('relu'),

            nn.Conv2D(channels=128, kernel_size=(3,3), strides=(1,1), padding=(1,1)),
            nn.BatchNorm(),
            nn.Activation('relu'),

            nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),

            nn.Conv2D(channels=num_of_pts*2, kernel_size=(3,3), strides=(1,1), padding=(0,0))
        )

    def hybrid_forward(self, F, x):
        return self.lmks_net(x)


if __name__ == '__main__':
	x   = mx.nd.random.uniform(0.0, 1.0, shape=(1, 3, 96, 96))
	net = BASE(num_of_pts=98)
	net.initialize(init=mx.initializer.Xavier())
	net.summary(x)
