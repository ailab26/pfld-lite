import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision
from mxnet import gluon



class M1BASE(mx.gluon.HybridBlock):
    def __init__(self, num_of_pts=98, **kwargs):
        super(M1BASE, self).__init__(**kwargs)
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

            ## 1-st down-scale
            nn.Conv2D(channels=16, kernel_size=(3,3), strides=(1,1), groups=16, padding=(1,1)),
            nn.BatchNorm(),
            nn.Conv2D(channels=32, kernel_size=(1,1), strides=(1,1), groups=1),
            nn.Activation('relu'),
            
            nn.Conv2D(channels=32, kernel_size=(3,3), strides=(1,1), groups=32, padding=(1,1)),
            nn.BatchNorm(),
            nn.Conv2D(channels=32, kernel_size=(1,1), strides=(1,1), groups=1),
            nn.Activation('relu'),

            nn.Conv2D(channels=32, kernel_size=(3,3), strides=(1,1), groups=32, padding=(1,1)),
            nn.BatchNorm(),
            nn.Conv2D(channels=32, kernel_size=(1,1), strides=(1,1), groups=1),
            nn.Activation('relu'),

            nn.Conv2D(channels=32, kernel_size=(3,3), strides=(2,2), groups=32, padding=(1,1)),
            nn.BatchNorm(),
            nn.Conv2D(channels=32, kernel_size=(1,1), strides=(1,1), groups=1),
            nn.Activation('relu'),

            ## 2-nd down-scale
            nn.Conv2D(channels=32, kernel_size=(3,3), strides=(1,1), groups=32, padding=(1,1)),
            nn.BatchNorm(),
            nn.Conv2D(channels=64, kernel_size=(1,1), strides=(1,1), groups=1),
            nn.Activation('relu'),

            nn.Conv2D(channels=64, kernel_size=(3,3), strides=(1,1), groups=64, padding=(1,1)),
            nn.BatchNorm(),
            nn.Conv2D(channels=64, kernel_size=(1,1), strides=(1,1), groups=1),
            nn.Activation('relu'),

            nn.Conv2D(channels=64, kernel_size=(3,3), strides=(1,1), groups=64, padding=(1,1)),
            nn.BatchNorm(),
            nn.Conv2D(channels=64, kernel_size=(1,1), strides=(1,1), groups=1),
            nn.Activation('relu'),

            nn.Conv2D(channels=64, kernel_size=(3,3), strides=(2,2), groups=64, padding=(1,1)),
            nn.BatchNorm(),
            nn.Conv2D(channels=64, kernel_size=(1,1), strides=(1,1), groups=1),
            nn.Activation('relu'),

            ## 3-rd keep-scale
            nn.Conv2D(channels=64, kernel_size=(3,3), strides=(1,1), groups=64, padding=(1,1)),
            nn.BatchNorm(),
            nn.Conv2D(channels=128, kernel_size=(1,1), strides=(1,1), groups=1),
            nn.Activation('relu'),

            nn.Conv2D(channels=128, kernel_size=(3,3), strides=(1,1), groups=128, padding=(1,1)),
            nn.BatchNorm(),
            nn.Conv2D(channels=128, kernel_size=(1,1), strides=(1,1), groups=1),
            nn.Activation('relu'),

            nn.Conv2D(channels=128, kernel_size=(3,3), strides=(1,1), groups=128, padding=(1,1)),
            nn.BatchNorm(),
            nn.Conv2D(channels=128, kernel_size=(1,1), strides=(1,1), groups=1),
            nn.Activation('relu'),
        )

        self.s2_feature.add(
            nn.Conv2D(channels=128, kernel_size=(3,3), strides=(2,2), padding=(1,1)),
            nn.BatchNorm(),
            nn.Activation('relu'),
        )

        self.s3_feature.add(
            nn.Conv2D(channels=256, kernel_size=(3,3), strides=(2,2), padding=(1,1)),
            nn.BatchNorm(),
            nn.Activation('relu'),
        )

        self.s1_post.add(
            nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            nn.Flatten(),
        )

        self.s2_post.add(
            nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            nn.Flatten(),
        )

        self.s3_post.add(
            nn.MaxPool2D(pool_size=(3,3), strides=(3,3)),
            nn.Flatten(),
        )

        self.lmks_out.add(
            nn.Dense(units=512, activation=None),
            nn.Dense(units=num_of_pts*2, activation=None),
        )

    def hybrid_forward(self, F, x):

        s1_f = self.s1_feature(x)
        s2_f = self.s2_feature(s1_f)
        s3_f = self.s3_feature(s2_f)

        s1_f = self.s1_post(s1_f)
        s2_f = self.s2_post(s2_f)
        s3_f = self.s3_post(s3_f)

        ms_f = mx.symbol.Concat(s1_f, s2_f, s3_f)
        #ms_f = mx.nd.Concat(s1_f, s2_f, s3_f)
        lmks = self.lmks_out(ms_f)

        return lmks


if __name__ == '__main__':
	x   = mx.nd.random.uniform(0.0, 1.0, shape=(1, 3, 96, 96))
	net = M1BASE(num_of_pts=98)
	net.initialize(init=mx.initializer.Xavier())
	net.summary(x)
