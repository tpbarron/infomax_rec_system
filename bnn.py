
import numpy as np

import torch
import torch.autograd as autograd
from torch.autograd import Variable, Function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_default_tensor_type('torch.FloatTensor')

USE_REPARAMETRIZATION_TRICK = True

def square(a):
    return torch.pow(a, 2.)

eps = np.finfo(float).eps

class RBF(nn.Module):

    def __init__(self):
        super(RBF, self).__init__()

    def forward(self, x):
        return torch.exp(-(torch.pow(x, 2.)))


class BayesianLayer(nn.Module):
    """Probabilistic layer that uses Gaussian distributed weights.

    Each weight has two parameters: mean and standard deviation.
    """

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 nonlinearity=F.relu,
                 prior_sd=0.1,
                 **kwargs):
        super(BayesianLayer, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.nonlinearity = nonlinearity
        self.prior_sd = prior_sd

        prior_rho = float(self.std_to_log(self.prior_sd).numpy())
        # print ("prior_rho: ", prior_rho)

        self.W = torch.Tensor(self.num_inputs, self.num_outputs).normal_(0.0, self.prior_sd)
        self.b = torch.zeros(self.num_outputs)

        # set the priors
        self.mu = nn.Parameter(torch.FloatTensor(self.num_inputs, \
            self.num_outputs).normal_(0.0, 1.))
        self.rho = nn.Parameter(torch.FloatTensor(self.num_inputs, \
            self.num_outputs).fill_(prior_rho))
        # bias priors
        self.b_mu = nn.Parameter(torch.FloatTensor(self.num_outputs).normal_(0., 1.))
        self.b_rho = nn.Parameter(torch.FloatTensor(self.num_outputs).fill_(prior_rho))

        # backups
        self.mu_old = torch.FloatTensor(self.num_inputs, \
            self.num_outputs).normal_(0., 1.)
        self.rho_old= torch.FloatTensor(self.num_inputs, \
            self.num_outputs).fill_(prior_rho)
        # bias priors
        self.b_mu_old = torch.FloatTensor(self.num_outputs).normal_(0., 1.)
        self.b_rho_old = torch.FloatTensor(self.num_outputs).fill_(prior_rho)

    def get_W(self):
        # Here we generate random epsilon values from a normal distribution
        epsilon = Variable(torch.FloatTensor(self.num_inputs, self.num_outputs).normal_(0.0, 1.0))
        # Here we calculate weights based on shifting and rescaling according
        # to mean and variance (paper step 2)
        W = self.mu + self.log_to_std(self.rho) * epsilon
        self.W = W
        return W

    def get_b(self):
        # Here we generate random epsilon values from a normal distribution
        epsilon = Variable(torch.FloatTensor(self.num_outputs).normal_(0.0, 1.0))
        b = self.b_mu + self.log_to_std(self.b_rho) * epsilon
        self.b = b
        return b

    def log_to_std(self, rho):
        return torch.log(1 + torch.exp(rho))

    def std_to_log(self, sigma):
        if not isinstance(sigma, torch.FloatTensor):
            sigma = torch.FloatTensor([sigma])
        return torch.log(torch.exp(sigma) - 1.)

    def save_old_params(self):
        """Save old parameter values for KL calculation."""
        self.mu_old.copy_(self.mu.data)
        self.rho_old.copy_(self.rho.data)
        self.b_mu_old.copy_(self.b_mu.data)
        self.b_rho_old.copy_(self.b_rho.data)

    def reset_to_old_params(self):
        """Reset to old parameter values for KL calculation."""
        self.mu.data.copy_(self.mu_old)
        self.rho.data.copy_(self.rho_old)
        self.b_mu.data.copy_(self.b_mu_old)
        self.b_rho.data.copy_(self.b_rho_old)

    def kl_div_p_q(self, p_mean, p_std, q_mean, q_std):
        """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian

        Args:
            p_mean: torch.autograd.Variable(torch.FloatTensor)
            p_std: torch.autograd.Variable(torch.FloatTensor)
            q_mean: torch.autograd.Variable(torch.FloatTensor)
            q_std: torch.autograd.Variable(torch.FloatTensor)
        """

        if not isinstance(p_mean, Variable) and not isinstance(p_mean, nn.Parameter):
            raise TypeError("arg p_mean must be torch.autograd.Variable")
        if not isinstance(p_std, Variable) and not isinstance(p_std, nn.Parameter):
            raise TypeError("arg p_std must be torch.autograd.Variable")
        if not isinstance(q_mean, Variable) and not isinstance(q_mean, nn.Parameter):
            raise TypeError("arg q_mean must be torch.autograd.Variable")
        if not isinstance(q_std, Variable) and not isinstance(q_std, nn.Parameter):
            raise TypeError("arg q_std must be torch.autograd.Variable")

        q_mean = q_mean.expand_as(p_mean)
        q_std = q_std.expand_as(p_std)
        numerator = square(p_mean - q_mean) + \
            square(p_std) - square(q_std)
        denominator = 2. * square(q_std) + eps
        return torch.sum(numerator / denominator + torch.log(q_std) - torch.log(p_std))

    def kl_div_new_prior(self):
        kl_div = self.kl_div_p_q(
            self.mu,
            self.log_to_std(self.rho),
            Variable(torch.FloatTensor([0.])),
            Variable(torch.FloatTensor([self.prior_sd])))
        kl_div += self.kl_div_p_q(self.b_mu,
                                  self.log_to_std(self.b_rho),
                                  Variable(torch.FloatTensor([0.])),
                                  Variable(torch.FloatTensor([self.prior_sd])))
        return kl_div

    def kl_div_old_new(self):
        # print ("KL div old new", self.mu_old, self.rho_old, self.mu, self.rho)
        kl_div = self.kl_div_p_q(
            Variable(self.mu_old),
            Variable(self.log_to_std(self.rho_old)),
            self.mu,
            self.log_to_std(self.rho))
        # if kl_div.data[0] < 0:
        #     print ("Dkl1: ", kl_div)
        kl_div += self.kl_div_p_q(Variable(self.b_mu_old),
                                  Variable(self.log_to_std(self.b_rho_old)),
                                  self.b_mu,
                                  self.log_to_std(self.b_rho))
        # if kl_div.data[0] < 0:
        #     print ("Dkl2: ", kl_div)
        return kl_div

    def get_output_for_reparametrization(self, input):
        """Implementation of the local reparametrization trick.

        This essentially leads to a speedup compared to the naive implementation case.
        Furthermore, it leads to gradients with less variance.

        References
        ----------
        Kingma et al., "Variational Dropout and the Local Reparametrization Trick", 2015
        """
        input = input.view(input.size()[0], -1)
        gamma = torch.addmm(self.b_mu.expand(input.size()[0], self.mu.size()[1]), input, self.mu)
        delta = torch.addmm(square(self.log_to_std(self.b_rho)).expand(input.size()[0], self.rho.size()[1]), \
            square(input), square(self.log_to_std(self.rho)))
        epsilon = Variable(torch.Tensor(self.num_outputs).normal_(0., 1.))
        activation = gamma + torch.sqrt(delta) * epsilon.expand_as(delta)
        if self.nonlinearity is not None:
            activation = self.nonlinearity(activation)

        return activation

    def get_output_for_default(self, input):
        input = input.view(input.size()[0], -1)
        W = self.get_W()
        b = self.get_b()
        activation = torch.addmm(b.expand(input.size()[0], W.size()[1]), input, W)

        if self.nonlinearity is not None:
            activation = self.nonlinearity(activation)

        return activation

    def forward(self, input):
        if USE_REPARAMETRIZATION_TRICK:
            return self.get_output_for_reparametrization(input)
        else:
            return self.get_output_for_default(input)


class BNN(nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_hidden=128,
                 nonlinearity=F.relu,
                 lr=0.001,
                 n_samples=10,
                 likelihood_sd=0.1,
                 nonlin=False):
        super(BNN, self).__init__()
        print ("Ins/outs: ", n_inputs, n_outputs)
        if nonlin:
            self.bl1 = BayesianLayer(n_inputs, n_hidden, nonlinearity=nonlinearity)
            self.bl3 = BayesianLayer(n_hidden, n_outputs, nonlinearity=F.sigmoid)
            self.bls = nn.ModuleList([self.bl1, self.bl3])
        else:
            self.bl = BayesianLayer(n_inputs, n_outputs, nonlinearity=F.sigmoid)
            self.bls = nn.ModuleList([self.bl])

        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.n_samples = n_samples
        self.likelihood_sd = likelihood_sd
        self.n_batches = 5. # same as original

    def save_old_params(self):
        for m in self.modules():
            if isinstance(m, BayesianLayer):
                m.save_old_params()

    def reset_to_old_params(self):
        for m in self.modules():
            if isinstance(m, BayesianLayer):
                m.reset_to_old_params()

    def info_gain(self):
        layers = []
        for m in self.modules():
            if isinstance(m, BayesianLayer):
                layers.append(m)
        return sum(l.kl_div_old_new() for l in layers)

    def _log_prob_normal(self, inp, mu=0., sigma=1.):
        if not isinstance(sigma, torch.FloatTensor) and not isinstance(sigma, Variable):
            sigma = Variable(torch.FloatTensor([sigma]))

        # print ("inp: ", inp)
        # print ("Sig: ", sigma)
        sigma = sigma.expand_as(inp)
        # print ("Sig: ", sigma)
        two_pi = Variable(torch.Tensor([2 * np.pi])).expand_as(inp)
        log_normal = - torch.log(sigma) - \
            torch.log(torch.sqrt(two_pi)) - \
            square(inp - mu) / (2. * square(sigma))
        return torch.sum(log_normal)

    def log_p_w_q_w_kl(self):
        """KL divergence KL[q_\phi(w)||p(w)]"""
        layers = []
        for m in self.modules():
            if isinstance(m, BayesianLayer):
                layers.append(m)
        return sum(l.kl_div_new_prior() for l in layers)

    def loss(self, inputs, targets):
        # MC samples.
        _log_p_D_given_w = []
        for _ in range(self.n_samples):
            # print ("Loss sample..")
            # Make prediction.
            prediction = self(inputs)
            # print ("prediction: ", prediction)
            # Calculate model likelihood log(P(D|w)).
            _log_p_D_given_w.append(self._log_prob_normal(targets, prediction, self.likelihood_sd))

        log_p_D_given_w = sum(_log_p_D_given_w)

        # Calculate variational posterior log(q(w)) and prior log(p(w)).
        kl = self.log_p_w_q_w_kl()

        # Calculate loss function.
        return kl / self.n_batches - log_p_D_given_w / self.n_samples

    def train(self, inputs, targets, use_cuda=False):
        self.opt.zero_grad()
        # print ("inputs: ", inputs.shape)
        # print ("targets: ", targets.shape)
        L = self.loss(Variable(torch.from_numpy(inputs).float()), Variable(torch.from_numpy(targets).float()))
        L.backward()
        self.opt.step()
        return L.data.cpu()[0]

    def loss_last_sample(self, input, target):
        """The difference with the original loss is that we only update based on the latest sample.
        This means that instead of using the prior p(w), we use the previous approximated posterior
        q(w) for the KL term in the objective function: KL[q(w)|p(w)] becomems KL[q'(w)|q(w)].
        """
        # MC samples.
        # _log_p_D_given_w = []
        _log_p_D_given_w = 0.
        for _ in range(self.n_samples):
            # Make prediction.
            prediction = self.forward(input)
            # Calculate model likelihood log(P(sample|w)).
            # _log_p_D_given_w.append(self._log_prob_normal(target, prediction, self.likelihood_sd))
            _log_p_D_given_w += self._log_prob_normal(target, prediction, self.likelihood_sd)

        # log_p_D_given_w = torch.sum(_log_p_D_given_w)
        # Calculate loss function.
        # self.kl_div() should be zero when taking second order step
        # info_gain() == kl_div()
        return self.info_gain() - _log_p_D_given_w / self.n_samples

    def fast_kl_div(self, inputs, targets, step_size=0.1, use_cuda=False):
        """
        Approximate KL div by curvature at origin. Ref VIME.
        """
        # save old parameters
        self.save_old_params()
        # compute gradients
        self.opt.zero_grad()
        inputs = Variable(torch.from_numpy(inputs)).float()
        targets = Variable(torch.from_numpy(targets)).float()
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        loss = self.loss_last_sample(inputs, targets)
        loss.backward()
        # now variables should have populated gradients

        kl_component = []
        for m in self.modules():
            if isinstance(m, BayesianLayer):
                # compute kl for mu
                mu = m.mu.data
                mu_grad = m.mu.grad.data
                rho_old = m.rho_old
                invH = torch.log(1 + torch.exp(rho_old)).pow(2.)
                # print (type(mu_grad), type(invH))
                kl_component.append((step_size**2. * mu_grad.pow(2.) * invH).sum())

                # compute kl for rho
                rho = m.rho.data
                rho_grad = m.rho.grad.data
                rho_old = m.rho_old
                # print (type(rho_grad))
                H = 2. * (torch.exp(2 * rho)) / (1. + torch.exp(rho)).pow(2.) / (torch.log(1. + torch.exp(rho)).pow(2.))
                invH = 1. / H
                # print (type(invH))
                kl_component.append((step_size**2. * rho_grad.pow(2.) * invH).sum())

                # compute kl for b_mu
                b_mu = m.b_mu.data
                b_mu_grad = m.b_mu.grad.data
                b_rho_old = m.b_rho_old
                invH = torch.log(1 + torch.exp(b_rho_old)).pow(2.)
                kl_component.append((step_size**2. * b_mu_grad.pow(2.) * invH).sum())

                # compute kl for rho
                b_rho = m.b_rho.data
                b_rho_grad = m.b_rho.grad.data
                b_rho_old = m.b_rho_old
                # print (type(rho_grad))
                H = 2. * (torch.exp(2 * b_rho)) / (1. + torch.exp(b_rho)).pow(2.) / (torch.log(1. + torch.exp(b_rho)).pow(2.))
                invH = 1. / H
                # print (type(invH))
                kl_component.append((step_size**2. * b_rho_grad.pow(2.) * invH).sum())

        # print (sum(kl_component))
        self.reset_to_old_params()
        return sum(kl_component)

    def kl_given_sample(self, inputs, targets):
        inputs = Variable(torch.from_numpy(inputs)).float()
        targets = Variable(torch.from_numpy(targets)).float()
        loss = self.loss_last_sample(inputs, targets)
        loss.backward()
        self.opt.step()
        return self.info_gain()

    def forward(self, inputs):
        x = inputs
        for bl in self.bls:
            # print (x)
            x = bl(x)
        # x = self.bl1(inputs)
        # x = self.bl2(x)
        # x = self.bl3(x)
        return x

def build_toy_dataset(n_data=400, noise_std=0.1):
    D = 1
    rs = np.random.RandomState(0)
    inputs = np.linspace(-5, 5, n_data)
    # inputs  = np.concatenate([np.linspace(0, 2, num=n_data/2),
    #                           np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    # inputs = (inputs - 4.0) / 4.0
    inputs  = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    return inputs, targets



# class BBSVI(nn.Module):
#
#     def __init__(self,
#                  model,
#                  n_samples=20,
#                  n_batches=1,
#                  likelihood_sd=0.1):
#         super(BBSVI, self).__init__()
#         self.model = model
#         self.n_samples = n_samples
#         self.n_batches = n_batches
#         self.likelihood_sd = likelihood_sd
#         self.num_param = 0
#
#         for m in self.model.modules():
#             if isinstance(m, BayesianLayer):
#                 self.num_param += m.mu.numel()
#                 self.num_param += m.b.numel()
#
#     # def log_prob(self, target, pred):
#     #     log_prior = 0.0
#     #     wts_sqrd = 0.0
#     #     noise_variance = 0.01
#     #     for m in self.model.modules():
#     #         if isinstance(m, BayesianLayer):
#     #             wts_sqrd += torch.sum(square(m.W))
#     #             wts_sqrd += torch.sum(square(m.b))
#     #
#     #     log_prior = -0.1 * wts_sqrd
#     #     log_lik = -torch.sum(square(pred - target)) / noise_variance
#     #     return log_prior + log_lik
#     #
#     # def gaussian_entropy(self):
#     #     log_std = 0.0
#     #     for m in self.model.modules():
#     #         if isinstance(m, BayesianLayer):
#     #             log_std += torch.sum(m.rho)
#     #             log_std += torch.sum(m.b_rho)
#     #
#     #     return 0.5 * self.num_param * (1.0 + torch.log(Variable(torch.DoubleTensor([2. * np.pi])))) + log_std
#     #
#     # def forward(self, inp, target):
#     #     # compute SVI loss
#     #     _log_p_D_given_w = []
#     #     for samp in range(self.n_samples):
#     #         # get prediction
#     #         pred = self.model(inp)
#     #         # calc model likelihood log(P(D|w))
#     #         _log_p_D_given_w.append(self.log_prob(target, pred))
#     #     log_p_D_given_w = sum(_log_p_D_given_w)
#     #     # cal variational posterior log(q(w)) and prior log(p(w))
#     #     ent = self.gaussian_entropy()
#     #     #kl = self.log_p_w_q_w_kl()
#     #     # calc loss
#     #     return ent - log_p_D_given_w / self.n_samples
#
#     def log_prob_normal(self, inp, mu=0., sigma=1.):
#         if not isinstance(sigma, torch.FloatTensor) and not isinstance(sigma, Variable):
#             sigma = Variable(torch.DoubleTensor([sigma]))
#
#         sigma = sigma.expand_as(inp)
#         two_pi = Variable(torch.Tensor([2 * np.pi])).expand_as(inp)
#         log_normal = - torch.log(sigma) - \
#             torch.log(torch.sqrt(two_pi)) - \
#             square(inp - mu) / (2. * square(sigma))
#         return torch.sum(log_normal)
#
#     def log_p_w_q_w_kl(self):
#         """KL divergence KL[q_\phi(w)||p(w)]"""
#         layers = []
#         for m in self.model.modules():
#             if isinstance(m, BayesianLayer):
#                 layers.append(m)
#         return sum(l.kl_div_new_prior() for l in layers)
#
#     def forward(self, inp, target):
#         # compute SVI loss
#         _log_p_D_given_w = []
#         for samp in range(self.n_samples):
#             # get prediction
#             pred = self.model(inp)
#             # calc model likelihood log(P(D|w))
#             _log_p_D_given_w.append(self.log_prob_normal(target, pred, self.likelihood_sd))
#         log_p_D_given_w = sum(_log_p_D_given_w)
#         # cal variational posterior log(q(w)) and prior log(p(w))
#         kl = self.log_p_w_q_w_kl()
#         # calc loss
#         return kl / self.n_batches - log_p_D_given_w / self.n_samples

# import matplotlib.pyplot as plt
#
# # Set up figure.
# fig = plt.figure(figsize=(12, 8), facecolor='white')
# ax = fig.add_subplot(111, frameon=False)
# plt.ion()
# plt.show(block=False)

def plot(model, inp, tar, itr, loss):
    print("Iteration {} lower bound {}".format(itr, -loss))

    plot_inputs = np.linspace(-8, 8, num=400)
    plot_inputs_var = Variable(torch.from_numpy(plot_inputs).float(), volatile=False)

    n_passes = 10
    outs = []
    for i in range(n_passes):
        outputs = model(plot_inputs_var.view(-1, 1))
        outputs = outputs.data.numpy()
        outs.append(outputs)

    # Plot data and functions.
    plt.cla()
    ax.plot(inp.ravel(), tar.ravel(), 'bx')
    for i in range(n_passes):
        ax.plot(plot_inputs, outs[i][:,0])
    ax.set_ylim([-2, 3])
    plt.draw()
    plt.pause(1.0/60.0)

def train(model, optimizer):
    model.train()
    inp, tar = build_toy_dataset()
    data, target = Variable(torch.from_numpy(inp).float()), Variable(torch.from_numpy(tar).float())

    for i in range(1000):
        optimizer.zero_grad()
        loss = model(data, target)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            plot(model.model, inp, tar, i, loss.data[0])
        print('Train loss: ', loss.data[0])

if __name__ == '__main__':
    model = BNN(1, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    bbsvi = BBSVI(model)
    train(bbsvi, optimizer)
