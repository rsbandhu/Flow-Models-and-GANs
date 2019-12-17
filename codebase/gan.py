import torch
from torch.nn import functional as F
import pdb

def loss_nonsaturating(g, d, x_real, *, device):
    '''
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating discriminator loss
    - g_loss (torch.Tensor): nonsaturating generator loss
    '''
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - F.binary_cross_entropy_with_logits
    #   - F.logsigmoid

    #Calculate discriminator output for real data
    d_real_out = d(x_real)

    #generate fake images
    g_fake = g(z)

    # Calculate discriminator output for fake data
    d_fake_out = d(g_fake)

    d_loss = 0
    #Calculate discriminator loss function
    d_loss -= F.logsigmoid(d_real_out).mean()
    d_loss -= torch.log(1-torch.sigmoid(d_fake_out)).mean()

    #Calculate Generator loss function
    g_loss = -F.logsigmoid(d_fake_out).mean()

    #raise NotImplementedError
    # YOUR CODE ENDS HERE

    return d_loss, g_loss

def conditional_loss_nonsaturating(g, d, x_real, y_real, *, device):
    '''
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating conditional discriminator loss
    - g_loss (torch.Tensor): nonsaturating conditional generator loss
    '''


    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    y_fake = y_real  # use the real labels as the fake labels as well

    # YOUR CODE STARTS HERE

    # Calculate discriminator output for real data with label conditioning
    d_real_out = d(x_real, y_real)

    #Generate fake images conditioned on labels
    g_fake = g(z, y_fake)

    # Calculate discriminator output for fake data with label conditioning
    d_fake_out = d(g_fake, y_fake)

    d_loss = 0

    #Calculate discriminator loss
    d_loss -= F.logsigmoid(d_real_out).mean()
    d_loss -= torch.log(1 - torch.sigmoid(d_fake_out)).mean()

    #Calculate generator loss
    g_loss = -F.logsigmoid(d_fake_out).mean()

    #raise NotImplementedError
    # YOUR CODE ENDS HERE

    return d_loss, g_loss

def loss_wasserstein_gp(g, d, x_real, *, device):
    '''
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): wasserstein discriminator loss
    - g_loss (torch.Tensor): wasserstein generator loss
    '''
    #pdb.set_trace()
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - torch.rand
    #   - torch.autograd.grad(..., create_graph=True)

    # Calculate discriminator output for real data
    d_real_out = d(x_real )
    g_fake = g( z )

    # Calculate discriminator output for fake data
    d_fake_out = d(g_fake )

    d_loss = 0
    #Discriminator loss of regular GAN objectives
    #d_loss -= F.logsigmoid( d_real_out ).mean()
    #d_loss -= torch.log( 1 - torch.sigmoid( d_fake_out ) ).mean()

    d_loss -= d_real_out.mean()
    d_loss += d_fake_out.mean()


    l2_reg = 10.0  # lambda for L2 regularization of gradient of D
    #choose a random number uniformly  in [0,1]
    alpha_rand = torch.rand_like(x_real, device=device)

    #Create a variable by mixing real and fake data
    r_xz = torch.tensor((alpha_rand*g_fake + (1-alpha_rand)*x_real),requires_grad = True)

    #Calculate output of the Discriminator for the mixed data and its gradient
    d_real_fake_out = d(r_xz).mean()
    gradient = torch.autograd.grad(outputs=d_real_fake_out, inputs=r_xz, only_inputs=True, create_graph=True)[0]
    #Calculate loss due to gradient
    grads_loss = l2_reg*((gradient.norm(2, dim= -1)-1)**2).mean()

    #Update discriminator loss with regularized gradients
    d_loss += grads_loss

    #Generator loss
    #g_loss = -F.logsigmoid( d_fake_out ).mean()
    g_loss = -d_fake_out.mean()

    #raise NotImplementedError
    # YOUR CODE ENDS HERE

    return d_loss, g_loss
