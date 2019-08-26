import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision
from mxnet import gluon



class MSBASE(mx.gluon.HybridBlock):
    def __init__(self, num_of_pts=98, **kwargs):
        super(MSBASE, self).__init__(**kwargs)
        self.pts_num = num_of_pts
        
        self.s1_feature = nn.HybridSequential()
        self.s2_feature = nn.HybridSequential()
        self.s3_feature = nn.HybridSequential()

        self.s1_post = nn.HybridSequential()
        self.s2_post = nn.HybridSequential()
        self.s3_post = nn.HybridSequential()

        self.lmks_out   = mx.gluon.nn.HybridSequential()

        self.s1_feature.add(

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
        )

        self.s2_feature.add(
            nn.Conv2D(channels=64, kernel_size=(3,3), strides=(2,2), padding=(1,1)),
            nn.BatchNorm(),
            nn.Activation('relu'),
        )

        self.s3_feature.add(
            nn.Conv2D(channels=64, kernel_size=(3,3), strides=(2,2), padding=(1,1)),
            nn.BatchNorm(),
            nn.Activation('relu'),
        )

        self.s1_avg = nn.AvgPool2D(pool_size=(2,2), strides=(2,2))

        self.s2_avg = nn.AvgPool2D(pool_size=(2,2), strides=(2,2))

        self.lmks_out.add(
            nn.Conv2D(channels=num_of_pts*2, kernel_size=(3,3), strides=(1,1), padding=(0,0))
        )

    def hybrid_forward(self, F, x):

        s1_f = self.s1_feature(x)
        s2_f = mx.sym.add_n(
            self.s2_feature(s1_f), self.s1_avg(s1_f)
        )
        s3_f = mx.sym.add_n(
            self.s3_feature(s2_f), self.s2_avg(s2_f)
        )
        lmks = self.lmks_out(s3_f)

        return lmks


if __name__ == '__main__':
	x   = mx.nd.random.uniform(0.0, 1.0, shape=(1, 3, 96, 96))
	net = MSBASE(num_of_pts=98)
	net.initialize(init=mx.initializer.Xavier())
	net.summary(x)
