import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision
from mxnet import gluon

def _add_conv(out, channels=1, kernel=1, stride=1, pad=0, num_group=1, relu=True):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=True))
    out.add(nn.BatchNorm(scale=True))
    if relu:
        out.add(nn.Activation('relu'))


def _add_conv_dw(out, dw_channels, channels, stride, relu=True):
    _add_conv(out, channels=dw_channels, kernel=3, stride=stride, pad=1, num_group=dw_channels, relu=relu)
    _add_conv(out, channels=channels, relu=relu)


class LinearBottleneck(nn.HybridBlock):
    def __init__(self, in_channels, channels, t, alpha, stride, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)

        self.use_shortcut = stride == 1 and in_channels == channels

        expand_channels = int(in_channels * t * alpha)
        with self.name_scope():
            self.out = nn.HybridSequential()
            _add_conv(self.out, expand_channels, relu=True)
            _add_conv(self.out, expand_channels, kernel=3, stride=stride, pad=1, num_group=expand_channels,relu=True)
            _add_conv(self.out, channels, relu=True)

    def hybrid_forward(self, F, x):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out


class CPFLD(mx.gluon.HybridBlock):
    def __init__(self, num_of_pts=98, alpha=1.0, **kwargs):
        super(CPFLD, self).__init__(**kwargs)
        self.pts_num = num_of_pts
        self.feature_shared = mx.gluon.nn.HybridSequential()
        self.lmks_net = mx.gluon.nn.HybridSequential()
        self.angs_net  = mx.gluon.nn.HybridSequential()
        
        ##------shared feature-----##
        ## shadow feature extraction
        _add_conv(self.feature_shared, channels=64, kernel=3, stride=2, pad=1, num_group=1)
        _add_conv_dw(self.feature_shared, dw_channels=64, channels=64, stride=1)
        ## mobilenet-v2, t=2, c=64, n=5, s=2
        self.feature_shared.add(
            LinearBottleneck(in_channels=64, channels=64, t=2, alpha=alpha, stride=2),
            LinearBottleneck(in_channels=64, channels=64, t=2, alpha=alpha, stride=1),
            LinearBottleneck(in_channels=64, channels=64, t=2, alpha=alpha, stride=1),
            LinearBottleneck(in_channels=64, channels=64, t=2, alpha=alpha, stride=1),
            LinearBottleneck(in_channels=64, channels=64, t=2, alpha=alpha, stride=1)
        )

        ##------landmark regression-----##
        ## mobilenet-v2, t=2, c=128, n=1, s=2
        self.lmks_net.add(
            LinearBottleneck(in_channels=64, channels=128, t=2, alpha=alpha, stride=2)
        )
        ## mobilenet-v2, t=4, c=128, n=6, s=1
        self.lmks_net.add(
            LinearBottleneck(in_channels=128, channels=128, t=4, alpha=alpha, stride=1),
            LinearBottleneck(in_channels=128, channels=128, t=4, alpha=alpha, stride=1),
            LinearBottleneck(in_channels=128, channels=128, t=4, alpha=alpha, stride=1),
            LinearBottleneck(in_channels=128, channels=128, t=4, alpha=alpha, stride=1),
            LinearBottleneck(in_channels=128, channels=128, t=4, alpha=alpha, stride=1),
            LinearBottleneck(in_channels=128, channels=128, t=4, alpha=alpha, stride=1)
        )
        ## mobilenet-v2, t=2, c=16, n=1, s=1
        self.lmks_net.add(
            LinearBottleneck(in_channels=128, channels=16, t=2, alpha=alpha, stride=1),
        )
        ## landmarks regression: base line
        self.s2_conv = nn.Conv2D(channels=32, kernel_size=(3,3), strides=(2,2), padding=(1,1), activation=None, use_bias=True)
        self.s2_bn   = nn.BatchNorm(scale=True)
        self.s2_act  = nn.Activation('relu')
        self.s3_conv = nn.Conv2D(channels=128, kernel_size=(3,3), strides=(2,2), padding=(1,1), activation=None, use_bias=True)
        self.s3_bn   = nn.BatchNorm(scale=True)
        self.s3_act  = nn.Activation('relu')
        
        self.s1_avg = nn.AvgPool2D(pool_size=(5,5), strides=(4,4), padding=(2,2))
        self.s2_avg = nn.AvgPool2D(pool_size=(3,3), strides=(2,2), padding=(1,1))

        self.lmks_out = nn.HybridSequential()
        self.lmks_out.add(
            nn.Conv2D(channels=num_of_pts*2, kernel_size=(3,3), strides=(1,1), padding=(0,0)),
            #nn.Flatten()
        )

    def hybrid_forward(self, F, x):
        x  = self.feature_shared(x)
        
	    ## regress facial landmark: base-line
        s1  = self.lmks_net(x)

        s2  = self.s2_conv(s1) 
        s2  = self.s2_bn(s2)
        s2  = self.s2_act(s2)

        s3  = self.s3_conv(s2)
        s3  = self.s3_bn(s3)
        s3  = self.s3_act(s3)

        
        s1  = self.s1_avg(s1)
        s2  = self.s2_avg(s2)
     
        lmk = F.concat(s1, s2, s3,  dim=1)
        lmk = self.lmks_out(lmk)
        return lmk


if __name__ == '__main__':
	x   = mx.nd.random.uniform(0.0, 1.0, shape=(1, 3, 96, 96))
	net = CPFLD(num_of_pts=98, alpha=0.25)
	net.initialize(init=mx.initializer.Xavier())
	net.summary(x)
