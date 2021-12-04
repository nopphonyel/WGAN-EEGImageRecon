import torch
from sklearn.metrics import f1_score


def micro_macro_f1(pred: torch.Tensor, real: torch.Tensor, num_classes):
    """
    Calculate the macro and micro f1 score (In the multiclass way)
    :param pred:
    :param real:
    :param num_classes:
    :return:
    """
    label_set = list(range(num_classes))
    pred = pred.detach().cpu().numpy()
    real = real.detach().cpu().numpy()
    micro = f1_score(y_true=real, y_pred=pred, labels=label_set, average='micro').item()
    macro = f1_score(y_true=real, y_pred=pred, labels=label_set, average='macro').item()
    return micro, macro


if __name__ == "__main__":
    import numpy as np

    BS = 10
    pred = torch.tensor(np.random.randint(0, 5, size=(BS)))
    real = torch.tensor(np.random.randint(0, 5, size=(BS)))

    micro, macro = micro_macro_f1(pred, real, 6)
    print(micro, macro)
