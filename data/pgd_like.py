import torch
import numpy as np







def poison_gradients(model, model_poison, k_percent):

    history_w = [param.data.clone().detach() for param in model_poison.parameters() if param.data is not None]
    history_grad = [param.grad.clone().detach() for param in model_poison.parameters() if param.grad is not None]

    for param, poison_w, poison_grad in zip(model.parameters(), history_w, history_grad):
        if param.grad is not None:

            current_grad = param.grad.view(-1)
            num_params = current_grad.numel()
            k = int(num_params * k_percent)

            _, top_k_indices = torch.topk(current_grad.abs(), k, largest=True)

            attack_w = param.data.view(-1).clone()
            attack_w[top_k_indices] = poison_w.view(-1)[top_k_indices]

            param.data.data.copy_(attack_w.view_as(param.data))


            attack_grad = param.grad.view(-1).clone()
            attack_grad[top_k_indices] = poison_grad.view(-1)[top_k_indices]

            param.grad.data.copy_(attack_grad.view_as(param.grad))




    return param.grad



def poison_gradients_change(model, model_poison, k_percent):
    for param, poison_param in zip(model.parameters(), model_poison.parameters()):
        if poison_param.grad is not None:

            poison_grad = poison_param.grad.view(-1)
            num_params = poison_grad.numel()
            k = int(num_params * k_percent)
            _, top_k_indices = torch.topk(poison_grad.abs(), k, largest=True)


            param_date_flat = param.data.view(-1).clone()
            constraint_set = param_date_flat[top_k_indices]


            poison_data_flat = poison_param.data.view(-1)
            for idx in top_k_indices:
                poison_value = poison_data_flat[idx]

                # param_date_flat[idx] = poison_value


                closest_value = constraint_set[(constraint_set - poison_value).abs().argmin()]
                param_date_flat[idx] = closest_value


            param.data.copy_(param_date_flat.view_as(param.data))

    return param.grad


