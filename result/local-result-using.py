import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Создаём архитектуру модели
model = fasterrcnn_resnet50_fpn(pretrained=False)

# Загружаем твои обученные веса
model.load_state_dict(torch.load('fasterrcnn_model.pth', map_location='cpu'))

# Переводим модель в режим оценки
model.eval()