import torch
import torch.nn as nn
import torch.nn.functional as F

def random_noise_attack(model, device, dat, eps):
    # Add uniform random noise in [-eps,+eps]
    x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    # Clip the perturbed datapoints to ensure we are in bounds [0,1]
    x_adv = torch.clamp(x_adv.clone().detach(), 0., 1.)
    # Return perturbed samples
    return x_adv

# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model,device,data,lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()


def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start):
    # TODO: Implement the PGD attack
    # - dat and lbl are tensors
    # - eps and alpha are floats
    # - iters is an integer
    # - rand_start is a bool

    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach()

    # If rand_start is True, add uniform noise to the sample within [-eps,+eps],
    # else just copy x_nat
    if rand_start:
        noise = torch.rand(x_nat.size()) * 2 * eps - eps
        noise = noise.to(device)
        x_noise = x_nat + noise
    else:
        x_noise = x_nat

    # Make sure the sample is projected into original distribution bounds [0,1]
    x_noise = torch.clamp(x_noise, min=0, max=1)
    
    # Iterate over iters
    for i in range(iters):
        # Compute gradient w.r.t. data (we give you this function, but understand it)
        grad = gradient_wrt_data(model, device, x_noise, lbl)
        # Perturb the image using the gradient
        pert = alpha * grad.sign()
        x_noise += pert
        # Clip the perturbed datapoints to ensure we still satisfy L_infinity constraint
        pert = torch.clamp(x_noise - x_nat, min=-eps, max=eps)
        x_noise  = x_nat + pert
        # Clip the perturbed datapoints to ensure we are in bounds [0,1]
        x_noise = torch.clamp(x_noise, min=0, max=1)
    # Return the final perturbed samples
    return x_noise


def FGSM_attack(model, device, dat, lbl, eps):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float

    # HINT: FGSM is a special case of PGD
    x_nat = dat.clone().detach()
    grad = gradient_wrt_data(model, device, x_nat, lbl)
    x_noise = x_nat + eps * grad.sign()
    x_noise = torch.clamp(x_noise, min=0, max=1)

    return x_noise


def rFGSM_attack(model, device, dat, lbl, eps):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float

    # HINT: rFGSM is a special case of PGD
    x_nat = dat.clone().detach()
    noise = torch.rand(x_nat.size()) * 2 * eps - eps
    noise = noise.to(device)
    x_noise = x_nat + noise
    x_noise = torch.clamp(x_noise, min=0, max=1)
    grad = gradient_wrt_data(model, device, x_noise, lbl)
    pert = alpha * grad.sign()
    x_noise += pert
    pert = torch.clamp(x_noise - x_nat, min=-eps, max=eps)
    x_noise  = x_nat + pert
    x_noise = torch.clamp(x_noise, min=0, max=1)

    return x_noise


def FGM_L2_attack(model, device, dat, lbl, eps):
    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach()

    # Compute gradient w.r.t. data
    grad = gradient_wrt_data(model, device, x_nat, lbl)
    # Compute sample-wise L2 norm of gradient (L2 norm for each batch element)
    l2_of_grad = grad.view(grad.size(0), -1).norm(dim=-1)
    # HINT: Flatten gradient tensor first, then compute L2 norm

    # Perturb the data using the gradient
    # HINT: Before normalizing the gradient by its L2 norm, use
    torch.clamp(l2_of_grad, min=1e-12) #to prevent division by 0
    for i in range(grad.size(0)):
        grad[i] /= l2_of_grad[i]
    # Add perturbation the data
    x_nat += eps * grad
    # Clip the perturbed datapoints to ensure we are in bounds [0,1]
    x_nat = torch.clamp(x_nat, min=0, max=1)
    # Return the perturbed samples
    return x_nat
