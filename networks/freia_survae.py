import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm


def xavier_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.xavier_normal_(m.weight.data)

def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                        nn.Linear(512,  c_out))
def subnet_conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 512,   3, padding=1), nn.ReLU(),
                        nn.Conv2d(512,  c_out, 3, padding=1))

def subnet_conv_1x1(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 512,   1), nn.ReLU(),
                        nn.Conv2d(512,  c_out, 1))

def simple_flow(args):
    nc, h, w = args.num_channels, args.size, args.size
    # a simple chain of operations is collected by ReversibleSequential
    inn = Ff.SequenceINN(nc * h * w)
    for k in range(12):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc)
    return inn



def base_flow(args):
    """Basic example from documentation: https://github.com/VLL-HD/FrEIA#quick-start-guide"""
    nc, h, w = args.num_channels, args.size, args.size
    nodes = [Ff.InputNode(nc, h, w, name='input')]
    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))
    ndim_x = nc * h * w

    # Higher resolution convolutional part
    for k in range(4):
        nodes.append(Ff.Node(nodes[-1],
                             Fm.GLOWCouplingBlock,
                             {'subnet_constructor':subnet_conv, 'clamp':1.2},
                             name=F'conv_high_res_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':k},
                             name=F'permute_high_res_{k}'))

    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

    # Lower resolution convolutional part
    for k in range(12):
        if k%2 == 0:
            subnet = subnet_conv_1x1
        else:
            subnet = subnet_conv

        nodes.append(Ff.Node(nodes[-1],
                             Fm.GLOWCouplingBlock,
                             {'subnet_constructor':subnet, 'clamp':1.2},
                             name=F'conv_low_res_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':k},
                             name=F'permute_low_res_{k}'))

    # Make the outputs into a vector, then split off 1/4 of the outputs for the
    # fully connected part
    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))
    split_node = Ff.Node(nodes[-1],
                        Fm.Split,
                        {'section_sizes':(ndim_x // 4, 3 * ndim_x // 4), 'dim':0},
                        name='split')
    nodes.append(split_node)

    # Fully connected part
    for k in range(12):
        nodes.append(Ff.Node(nodes[-1],
                             Fm.GLOWCouplingBlock,
                             {'subnet_constructor':subnet_fc, 'clamp':2.0},
                             name=F'fully_connected_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':k},
                             name=F'permute_{k}'))

    # Concatenate the fully connected part and the skip connection to get a single output
    nodes.append(Ff.Node([nodes[-1].out0, split_node.out1],
                        Fm.Concat1d, {'dim':0}, name='concat'))
    nodes.append(Ff.OutputNode(nodes[-1], name='output'))

    inn = Ff.GraphINN(nodes, verbose=args.verbose)
    inn.apply(xavier_init)
    return inn


from survae.flows import Flow
from survae.transforms import VAE, AffineCouplingBijection, ActNormBijection2d, Conv1x1
from survae.transforms import UniformDequantization, Augment, Squeeze2d, Slice
from survae.distributions import StandardNormal, ConditionalNormal, ConditionalBernoulli, StandardUniform
from survae.nn.layers import ElementwiseParams2d
from survae.nn.nets import MLP, DenseNet

def vae(args):
    latent_size = 20
    num_dims = args.size*args.size
    encoder = ConditionalNormal(MLP(num_dims, 2*latent_size,
                                    hidden_units=[512,256],
                                    activation='relu',
                                    in_lambda=lambda x: 2 * x.view(x.shape[0], num_dims).float() - 1))
    decoder = ConditionalBernoulli(MLP(latent_size, num_dims,
                                       hidden_units=[512,256],
                                       activation='relu',
                                       out_lambda=lambda x: x.view(x.shape[0], 1, args.size, args.size)))

    model = Flow(base_dist=StandardNormal((latent_size,)),
                 transforms=[VAE(encoder=encoder, decoder=decoder)])
    return model

def net(channels):
    return nn.Sequential(DenseNet(in_channels=channels//2, out_channels=channels,
                                  num_blocks=1, mid_channels=64, depth=8,
                                  growth=16, dropout=0.0, gated_conv=True,
                                  zero_init=True),
                        ElementwiseParams2d(2))

def flow(args):
    c = args.num_channels
    s = args.size
    model = Flow(base_dist=StandardNormal((c*8, s//4, s//4)), transforms=[
                 UniformDequantization(num_bits=8),
                 Augment(StandardUniform((c, s, s)), x_size=c),
                 AffineCouplingBijection(net(c*2)), ActNormBijection2d(c*2), Conv1x1(c*2),
                 AffineCouplingBijection(net(c*2)), ActNormBijection2d(c*2), Conv1x1(c*2),
                 AffineCouplingBijection(net(c*2)), ActNormBijection2d(c*2), Conv1x1(c*2),
                 AffineCouplingBijection(net(c*2)), ActNormBijection2d(c*2), Conv1x1(c*2),
                 Squeeze2d(), Slice(StandardNormal((c*4, s//2, s//2)), num_keep=c*4),
                 AffineCouplingBijection(net(c*4)), ActNormBijection2d(c*4), Conv1x1(c*4),
                 AffineCouplingBijection(net(c*4)), ActNormBijection2d(c*4), Conv1x1(c*4),
                 AffineCouplingBijection(net(c*4)), ActNormBijection2d(c*4), Conv1x1(c*4),
                 AffineCouplingBijection(net(c*4)), ActNormBijection2d(c*4), Conv1x1(c*4),
                 Squeeze2d(), Slice(StandardNormal((c*8, s//4, s//4)), num_keep=c*8),
                 AffineCouplingBijection(net(c*8)), ActNormBijection2d(c*8), Conv1x1(c*8),
                 AffineCouplingBijection(net(c*8)), ActNormBijection2d(c*8), Conv1x1(c*8),
                 AffineCouplingBijection(net(c*8)), ActNormBijection2d(c*8), Conv1x1(c*8),
                 AffineCouplingBijection(net(c*8)), ActNormBijection2d(c*8), Conv1x1(c*8),])
    return model

        # loss = -model.log_prob(x.to(device)).sum() / (math.log(2) * x.numel())
