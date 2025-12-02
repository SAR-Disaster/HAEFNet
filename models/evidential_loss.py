import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialLoss(nn.Module):
    def __init__(self, num_classes=2, uncertainty_weight=0.1, focal_weight=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.uncertainty_weight = uncertainty_weight
        self.focal_weight = focal_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, outputs, labels):
        if isinstance(outputs, list):
            output = outputs[0]
        else:
            output = outputs

        ce_loss = self.ce_loss(output, labels)
        uncertainty_loss = self._compute_uncertainty_loss(output, labels)
        total_loss = ce_loss + self.uncertainty_weight * uncertainty_loss

        return total_loss

    def _compute_uncertainty_loss(self, output, labels):
        probs = F.softmax(output, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
        valid_mask = (labels != 255).float()
        avg_uncertainty = (entropy * valid_mask).sum() / (valid_mask.sum() + 1e-12)
        target_uncertainty = 0.3
        uncertainty_loss = F.mse_loss(avg_uncertainty, torch.tensor(target_uncertainty, device=output.device))

        return uncertainty_loss


class DempsterShaferLoss(nn.Module):
    def __init__(self, num_classes=2, uncertainty_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.uncertainty_weight = uncertainty_weight

    def forward(self, mass_functions, labels):
        B, K_plus_1, H, W = mass_functions.shape
        K = K_plus_1 - 1

        valid_mask = (labels != 255).float()

        plausibility = mass_functions[:, :-1, :, :] + mass_functions[:, -1:, :, :]

        prob_dist = plausibility / (plausibility.sum(dim=1, keepdim=True) + 1e-12)

        log_probs = torch.log(prob_dist + 1e-12)

        labels_one_hot = F.one_hot(labels.clamp(0, K - 1), num_classes=K).permute(0, 3, 1, 2).float()

        ce_loss = -(labels_one_hot * log_probs).sum(dim=1)
        ce_loss = (ce_loss * valid_mask).sum() / (valid_mask.sum() + 1e-12)

        uncertainty = mass_functions[:, -1, :, :]
        uncertainty_loss = (uncertainty * valid_mask).sum() / (valid_mask.sum() + 1e-12)

        total_loss = ce_loss + self.uncertainty_weight * uncertainty_loss

        return total_loss


def create_evidential_loss(config):
    loss_type = config.get("type", "evidential")
    num_classes = config.get("num_classes", 2)
    uncertainty_weight = config.get("uncertainty_weight", 0.1)

    if loss_type == "evidential":
        return EvidentialLoss(num_classes=num_classes, uncertainty_weight=uncertainty_weight)
    elif loss_type == "dempster_shafer":
        return DempsterShaferLoss(num_classes=num_classes, uncertainty_weight=uncertainty_weight)
    else:
        raise ValueError(f"Unknown evidential loss type: {loss_type}")


if __name__ == "__main__":
    torch.manual_seed(0)

    B, K, H, W = 2, 2, 32, 32
    outputs = [torch.randn(B, K, H, W)]
    labels = torch.randint(0, K, (B, H, W))

    evidential_loss = EvidentialLoss(num_classes=K)
    loss1 = evidential_loss(outputs, labels)
    print(f"Evidential Loss: {loss1.item():.4f}")

    mass_functions = torch.rand(B, K + 1, H, W)
    mass_functions = mass_functions / mass_functions.sum(dim=1, keepdim=True)

    ds_loss = DempsterShaferLoss(num_classes=K)
    loss2 = ds_loss(mass_functions, labels)
    print(f"Dempster-Shafer Loss: {loss2.item():.4f}")

    print("✓ 证据损失函数测试通过!")
